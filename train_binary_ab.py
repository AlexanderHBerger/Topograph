import json
import os
import random
from argparse import ArgumentParser

import numpy as np
import torch.multiprocessing as mp
import torch.nn as nn
import yaml
from sklearn.model_selection import KFold

import wandb


from metrics.betti_error import BettiNumberMetric
from metrics.cldice import ClDiceMetric
from metrics.topograph import TopographMetric
from utils import train_utils

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
from glob import glob
from shutil import copyfile

import monai
import torch
from monai.data import list_data_collate
from monai.metrics import DiceMetric
from torch.utils.data import DataLoader
from tqdm import tqdm

from evaluation import evaluate_model



os.environ['KMP_DUPLICATE_LIB_OK']='True'


parser = ArgumentParser()
parser.add_argument('--config',
                    default=None,
                    help='config file (.yaml) containing the hyper-parameters for training and dataset specific info.')
parser.add_argument('--pretrained', default=None, help='checkpoint of the pretrained model')
parser.add_argument('--resume', default=None, help='checkpoint of the last epoch of the model')
parser.add_argument('--cuda_visible_device', nargs='*', type=int, default=[0],
                        help='list of index where skip conn will be made')
parser.add_argument('--disable_wandb', default=False, action='store_true',
                    help='disable wandb logging')
parser.add_argument('--perceptual_network', default=False, action='store_true',
                    help='If a perceptual network for the loss should be trained')
parser.add_argument('--folds', default=1, help='Number of folds for cross-validation')
parser.add_argument('--fold_no', default=-1, help='Which fold to use for training, validation, and testing')
parser.add_argument('--sweep', default=False, action='store_true',
                    help='If the training is part of a sweep')
parser.add_argument("--log_train_images", default=False, action='store_true',
                    help="Log training images to wandb")
parser.add_argument("--integrate_test", default=False, action='store_true',
                    help="Integrate test into training")
parser.add_argument("--no_c", default=True, help="Whether to not use the efficient c implementation")

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)

def main(args, config):
    # fixing seed for reproducibility
    random.seed(config.TRAIN.SEED)
    np.random.seed(config.TRAIN.SEED)
    torch.random.manual_seed(config.TRAIN.SEED)
    monai.utils.set_determinism(seed=config.TRAIN.SEED) # type: ignore
    
    if args.resume and args.pretrained:
        raise Exception('Do not use pretrained and resume at the same time.')
    
    if int(args.folds) > 1 and int(args.fold_no) == -1:
        raise Exception('Please specify which fold to use for training, validation, and testing')
    
    if int(args.folds) > 1 and int(args.fold_no) >= int(args.folds):
        raise Exception('Fold number must be less than the number of folds')
    
    # create UNet, DiceLoss and Adam optimizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Running on", device)

    if args.sweep:
        run = wandb.init(project="binary_topograph")
        exp_id = run.name + "_" + run.id
        config.TRAIN.LR = wandb.config.lr
        config.LOSS.ALPHA = wandb.config.alpha
        config.MODEL.CHANNELS = wandb.config.channels
        config.MODEL.NUM_RES_UNITS = wandb.config.num_res_units
        config.LOSS.ALPHA_WARMUP_EPOCHS = wandb.config.alpha_warmup_epochs
        config.TRAIN.BATCH_SIZE = wandb.config.batch_size
        config.LOSS.DICE_TYPE = wandb.config.dice_type
        config.LOSS.CLDICE_ALPHA = wandb.config.cldice_alpha
        config.LOSS.PUSH_UNMATCHED_TO_1_0 = wandb.config.push_unmatched_to_1_0
        config.LOSS.BARCODE_LENGTH_THRESHOLD = wandb.config.barcode_length_threshold
        config.LOSS.USE_LOSS = wandb.config.loss_method
        #config.LOSS.ERROR_WEIGHTING = wandb.config.error_weighting
        config.LOSS.AGGREGATION_TYPE = getattr(wandb.config, 'aggregation_type', 'mean')


        config.LOSS.THRES_DISTR = getattr(wandb.config, 'thres_distr', "none")
        config.LOSS.THRES_VAR = getattr(wandb.config, 'thres_var', 0.0)
        config.LOSS.AGGREGATION_TYPE = getattr(wandb.config, 'aggregation_type', 'mean')
        config.TRAIN.OPTIMIZER = getattr(wandb.config, 'optimizer', 'adam')
        config.TRAIN.WEIGHT_DECAY = getattr(wandb.config, 'weight_decay', 0.0)
        config.TRAIN.LR_SCHEDULE = getattr(wandb.config, 'lr_schedule', 'constant')

    else:
        # set default values for new parameters to keep compatibility with old config files
        config.LOSS.THRES_DISTR = getattr(config.LOSS, 'THRES_DISTR', "none")
        config.LOSS.THRES_VAR = getattr(config.LOSS, 'THRES_VAR', 0.0)
        config.LOSS.AGGREGATION_TYPE = getattr(config.LOSS, 'AGGREGATION_TYPE', 'mean')
        config.TRAIN.OPTIMIZER = getattr(config.TRAIN, 'OPTIMIZER', 'adam')
        config.TRAIN.WEIGHT_DECAY = getattr(config.LOSS, 'WEIGHT_DECAY', 0.0)
        config.TRAIN.LR_SCHEDULE = getattr(config.TRAIN, 'LR_SCHEDULE', 'constant')
    
    # Background is always set to false for the binary datasets
    config.DATA.INCLUDE_BACKGROUND = False
    config.LOSS.TOPOLOGY_WEIGHTS = (1,1) 

    if args.integrate_test:

        # Load the dataset
        train_dataset, val_dataset, test_dataset = train_utils.binary_dataset_selection(config, args)

        test_loader = DataLoader(
            test_dataset,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=config.TRAIN.NUM_WORKERS,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
            sampler=None,
            drop_last=False
        ) 
    
    else:
        # Load the dataset
        train_dataset, val_dataset = train_utils.binary_dataset_selection(config, args)
    
    # Initialize the k-fold cross validation
    folds = int(args.folds)
    if folds > 1:
        kf = KFold(n_splits=folds, shuffle=True, random_state=config.TRAIN.SEED)
        iterator = kf.split(train_dataset) # type: ignore
        iterator = [list(iterator)[int(args.fold_no)]]
    else:
        ds_range = torch.randperm(len(train_dataset))
        # Randomly select 80% of the ids in ds_range for training and the rest for validation
        train_idx = ds_range[:int(0.8*len(train_dataset))]
        val_idx = ds_range[int(0.8*len(train_dataset)):]

        iterator = [(train_idx, val_idx)]

    for fold, (train_idx, val_idx) in enumerate(iterator):
        print("train idx:", train_idx)
        print("val idx:", val_idx)
        # In datasets, where we want a patient-wise split, we need to create a new dataset for each fold
        if config.DATA.DATASET in ['roads', 'cremi', 'elegans', 'drive', 'buildings']:
            train_ds = train_dataset
            val_ds = val_dataset

            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            shuffle = False
        else:
            train_ds = train_dataset
            val_ds = train_dataset

            train_sampler = torch.utils.data.SubsetRandomSampler(train_idx)
            val_sampler = torch.utils.data.SubsetRandomSampler(val_idx)
            shuffle = False
            
        # Create dataloaders accordingly
        train_loader = DataLoader(
            train_ds,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=shuffle,
            num_workers=config.TRAIN.NUM_WORKERS,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
            sampler=train_sampler,
            drop_last=True
        )

        val_loader = DataLoader(
            val_ds,
            batch_size=config.TRAIN.BATCH_SIZE,
            shuffle=False,
            num_workers=config.TRAIN.NUM_WORKERS,
            collate_fn=list_data_collate,
            pin_memory=torch.cuda.is_available(),
            sampler=val_sampler
        )
        
        dice_metric = DiceMetric(include_background=True,
                                reduction="mean",
                                get_not_nans=False)
        clDice_metric = ClDiceMetric(
            ignore_background=not config.DATA.INCLUDE_BACKGROUND,
        )
        betti_number_metric = BettiNumberMetric(
            num_processes=16,
            ignore_background=not config.DATA.INCLUDE_BACKGROUND,
            eight_connectivity= config.LOSS.EIGHT_CONNECTIVITY
        )
        topograph_metric = TopographMetric(
            num_processes=16,
            ignore_background=not config.DATA.INCLUDE_BACKGROUND,
            sphere=False
        )

        # Create model
        if config.MODEL.NAME and config.MODEL.NAME == 'SwinUNETR':
            model = monai.networks.nets.SwinUNETR(
                img_size=(160,160),
                in_channels=config.DATA.IN_CHANNELS,
                out_channels=config.DATA.OUT_CHANNELS,
                spatial_dims=2,
                depths=(2,2,2,2),
                num_heads=(3,6,12, 24),
            ).to(device)
        elif config.MODEL.NAME and config.MODEL.NAME == 'UNet':
            model = monai.networks.nets.UNet(
                spatial_dims=2,
                in_channels=config.DATA.IN_CHANNELS,
                out_channels=config.DATA.OUT_CHANNELS,
                channels=config.MODEL.CHANNELS,
                strides=[2] + [1 for _ in range(len(config.MODEL.CHANNELS) - 2)],
                num_res_units=config.MODEL.NUM_RES_UNITS,
            ).to(device)
        else:
            raise Exception('ERROR: Model not implemented')
        
        # Loss function choice
        loss_function, exp_name = train_utils.loss_selection(config, args)

        if args.sweep:
            exp_name = 'sweep_' + exp_id + "_" + exp_name
        # Create wandb run
        elif not args.disable_wandb and not args.sweep:
            # start a new wandb run to track this script
            wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project="binary_topograph",
                
                # track hyperparameters and run metadata
                config={
                    "learning_rate": float(config.TRAIN.LR),
                    "epochs": config.TRAIN.MAX_EPOCHS,
                    "batch_size": config.TRAIN.BATCH_SIZE,
                    "datapath": config.DATA.DATA_PATH,
                    "exp_name": exp_name,
                    "loss_type": config.LOSS.USE_LOSS,
                    "alpha": config.LOSS.ALPHA,
                }
            )
            # get run id
            exp_name += "_" + wandb.run.id
            
        # Copy config files and verify if files exist
        exp_path = './models/'+config.DATA.DATASET+'/'+exp_name
        if os.path.exists(exp_path) and args.resume == None:
            raise Exception('ERROR: Experiment folder exist, please delete folder or check config file')
        else:
            try:
                os.makedirs(exp_path)
                copyfile(args.config, os.path.join(exp_path, "config.yaml"))
            except:
                pass
            
        optimizer = torch.optim.Adam(model.parameters(), config.TRAIN.LR)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1000000, gamma=0.1)   #always check that the step size is high enough
        
        # Resume training
        last_epoch = 0
        if args.resume:
            dic = torch.load(args.resume)
            model.load_state_dict(dic['model'])
            optimizer.load_state_dict(dic['optimizer'])
            scheduler.load_state_dict(dic['scheduler'])
            last_epoch = int(scheduler.last_epoch/len(train_loader))
            
        # Start from pretrained model
        if args.pretrained:
            dic = torch.load(args.pretrained)
            model.load_state_dict(dic['model'])

        # Variables for model selection
        best_combined_val_metric = -1
        best_metric_epoch = -1

        # Variables for early stopping
        best_dice = -1
        best_bm = -1

        # start a typical PyTorch training
        for epoch in tqdm(range(last_epoch, config.TRAIN.MAX_EPOCHS)):
            model.train()
            step = 0
            alpha = 0
            num_iterations = len(train_loader)
            # Iteration loop
            print("starting training loop with num_iterations:", num_iterations)
            for iteration, batch_data in enumerate(tqdm(train_loader)):
                step += 1
                optimizer.zero_grad()
                inputs, labels = batch_data["img"].to(device), batch_data["seg"].to(device)

                # convert meta tensor back to normal tensor
                if isinstance(inputs, monai.data.meta_tensor.MetaTensor): # type: ignore
                    inputs = inputs.as_tensor()
                    labels = labels.as_tensor()
                    
                outputs = model(inputs)

                if config.LOSS.USE_LOSS == 'FastMulticlassDiceBettiMatching' or config.LOSS.USE_LOSS == 'HuTopo' or config.LOSS.USE_LOSS == 'Topograph' or config.LOSS.USE_LOSS == 'MOSIN':
                    if config.LOSS.ALPHA > 0 and int(config.LOSS.ALPHA_WARMUP_EPOCHS) < epoch:
                        #p = float(iteration + (epoch - config.LOSS.ALPHA_WARMUP_EPOCHS) * num_iterations) / (config.TRAIN.MAX_EPOCHS * num_iterations)
                        #alpha = (2. / (1. + np.exp(-10 * p)) - 1) * config.LOSS.ALPHA
                        alpha = config.LOSS.ALPHA

                    loss, dic = loss_function(outputs, labels, alpha=alpha)
                else:
                    loss, dic = loss_function(outputs, labels)

                loss.backward()
                optimizer.step()
                scheduler.step()

                if step % config.TRAIN.LOG_INTERVAL == 0 and not args.disable_wandb:
                    logging = {
                        "train/lr": scheduler.get_last_lr()[0],
                        "train/alpha": alpha,
                        "train/train_loss": loss.item(),
                    }

                    if args.log_train_images:
                        class_labels = {
                            0: "Zero",
                            1: "One",
                            2: "Two",
                            3: "Three",
                            4: "Four",
                            5: "Five",
                            6: "Six",
                            7: "Seven",
                            8: "Eight",
                            9: "Nine",
                            10: "Background"
                        }
                        mask_img = wandb.Image(inputs[0].cpu(), masks={
                            "predictions": {"mask_data": torch.argmax(outputs[0].cpu(), dim=0).numpy(), "class_labels": class_labels},
                            "ground_truth": {"mask_data": torch.argmax(labels[0].cpu(), dim=0).numpy(), "class_labels": class_labels},
                        })
                        logging["train image"] = mask_img

                    if "bm" in dic:
                        logging["train/bm_loss"] = dic["bm"]
                    if "dice" in dic:
                        logging["train/dice_loss"] = dic["dice"]
                    if "cldice" in dic:
                        logging["train/cldice_loss"] = dic["cldice"]
                    if "wasserstein" in dic:
                        logging["train/wasserstein"] = dic["wasserstein"]
                    if "topograph_loss" in dic:
                        logging["train/topograph_loss"] = dic["topograph_loss"]

                    wandb.log(logging)

            # Validation loop
            print("starting validation loop")
            if (epoch + 1) % config.TRAIN.VAL_INTERVAL == 0:
                model.eval()
                with torch.no_grad():
                    val_images = None
                    val_labels = None
                    val_outputs = None

                    for val_data in tqdm(val_loader):
                        val_images, val_labels = val_data["img"].to(device), val_data["seg"].to(device)
                        # convert meta tensor back to normal tensor
                        if isinstance(val_images, monai.data.meta_tensor.MetaTensor): # type: ignore
                            val_images = val_images.as_tensor()
                            val_labels = val_labels.as_tensor()

                        val_outputs = model(val_images)

                        # Get the class index with the highest value for each pixel
                        pred_indices = torch.argmax(val_outputs, dim=1)

                        # Convert to onehot encoding
                        one_hot_pred = torch.nn.functional.one_hot(pred_indices, num_classes=val_outputs.shape[1])

                        # Move channel dimension to the second dim
                        one_hot_pred = one_hot_pred.permute(0, 3, 1, 2)

                        # compute metric for current iteration
                        dice_metric(y_pred=one_hot_pred, y=val_labels)
                        clDice_metric(y_pred=one_hot_pred, y=val_labels)
                        betti_number_metric(y_pred=one_hot_pred, y=val_labels)
                        topograph_metric(y_pred=one_hot_pred, y=val_labels)


                    # aggregate the final mean dice result
                    dice_score = dice_metric.aggregate().item()
                    clDice_score = clDice_metric.aggregate().item()
                    b0, b1, bm, norm_bm = betti_number_metric.aggregate()
                    topograph_score = topograph_metric.aggregate().item()
                    combined_metric = dice_score + 2 * (1 - norm_bm)
                    #combined_metric = 1 - norm_bm
                    
                    # Save checkpoint
                    # If it is best model, save it seperately and conduct a test run
                    # Note that the perfect dice score is a score of 1 and the perfect betti number score is 0
                    # Therefore, we want to maximize the dice score and minimize the betti number score
                    if combined_metric > best_combined_val_metric:
                        dic = {}
                        dic['model'] = model.state_dict()
                        dic['optimizer'] = optimizer.state_dict()
                        dic['scheduler'] = scheduler.state_dict()
                        best_combined_val_metric = combined_metric.item()
                        best_metric_epoch = epoch + 1
                        best_dice = dice_score
                        best_bm = bm
                        torch.save(dic, './models/'+config.DATA.DATASET+'/'+exp_name+'/best_model_dict.pth')
                        if args.integrate_test:
                            print("starting test loop")
                            evaluate_model(
                                model, 
                                test_loader, 
                                device,
                                config.DATA.INCLUDE_BACKGROUND,
                                config.LOSS.EIGHT_CONNECTIVITY,
                                logging=not args.disable_wandb,
                                mask_background=False,

                            )

                    if not args.disable_wandb and val_images is not None and val_labels is not None and pred_indices is not None:
                        class_labels = {
                            0: "Zero",
                            1: "One",
                            2: "Two",
                            3: "Three",
                            4: "Four",
                            5: "Five",
                            6: "Six",
                            7: "Seven",
                            8: "Eight",
                            9: "Nine",
                            10: "Background"
                        }
                        mask_img = wandb.Image(val_images[0].cpu(), masks={
                            "predictions": {"mask_data": pred_indices.cpu()[0].numpy(), "class_labels": class_labels},
                            "ground_truth": {"mask_data": torch.argmax(val_labels[0].cpu(), dim=0).numpy(), "class_labels": class_labels},
                        })
                        wandb.log({
                            "val/val_mean_dice": dice_score,
                            "val/val_mean_cldice": clDice_score,
                            "val/val_b0_error": b0,
                            "val/val_b1_error": b1,
                            "val/val_bm_loss": bm,
                            "val/val_topograph_loss": topograph_score,
                            "val/val_normalized_bm_loss": norm_bm,
                            "val/val_combined_metric": combined_metric,
                            "val/validation image": mask_img,
                            "val/best_combined_metric": best_combined_val_metric,
                            "epoch": epoch+1,
                        })

                    # reset the status for next validation round
                    dice_metric.reset()
                    clDice_metric.reset()
                    betti_number_metric.reset()
                    topograph_metric.reset()


        if not args.disable_wandb and not args.sweep:
            wandb.finish()


    print(f"train completed, best_metric: {best_combined_val_metric:.4f} at epoch: {best_metric_epoch}")

if __name__ == "__main__":
    args = parser.parse_args()
    if args.cuda_visible_device is not None:
        os.environ["CUDA_VISIBLE_DEVICES"] = ','.join(map(str, args.cuda_visible_device))

    # Load the config files
    with open(args.config) as f:
        print('\n*** Config file')
        print(args.config)
        config = yaml.load(f, Loader=yaml.FullLoader)
    config = dict2obj(config)
    
    main(args, config)