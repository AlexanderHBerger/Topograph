import os
from sklearn.model_selection import KFold
import yaml
import random
import numpy as np
import json
import wandb
from argparse import ArgumentParser
import torch.nn as nn
from datasets.platelet import PlateletDataset

from losses.utils import ThresholdDistribution
from metrics.cldice import ClDiceMetric
from metrics.betti_error import BettiNumberMetric
import torch.multiprocessing as mp

from metrics.topograph import TopographMetric
from utils.sam import SAM

os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"
import sys
from shutil import copyfile
from glob import glob
import torch
from torch.utils.data import DataLoader
import monai
from monai.data import list_data_collate
from monai.metrics import DiceMetric

from losses.betti_losses import *
from datasets.acdc import ACDC_ShortAxisDataset
from evaluation import evaluate_model
from utils import train_utils

from tqdm import tqdm

os.environ['KMP_DUPLICATE_LIB_OK']='True'
os.environ['WANDB_API_KEY'] = '94ab84459aa1b42734e2980087f053e645c271e7'


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
parser.add_argument("--test_best", default=False, action='store_true',
                    help="Whether to test only the model with best val score")
parser.add_argument("--no_c", default=False, action='store_true', help="Whether to not use the efficient c implementation")

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
        run = wandb.init()
        exp_id = run.name + "_" + run.id
        config = train_utils.get_config_from_sweep(config)
    else:
        # set default values for new parameters to keep compatibility with old config files
        config = train_utils.set_default_values(config)

    # Load the dataset
    if args.integrate_test:
        train_dataset, test_dataset = train_utils.multiclasses_dataset_selection(config, args.integrate_test)

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
        train_dataset = train_utils.multiclasses_dataset_selection(config, args.integrate_test)
    
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
        if config.DATA.DATASET == 'ACDC_sa':
            # Add one to each elements because patient ids start at 1
            train_idx += 1
            val_idx += 1

            # Convert id tensors to list
            train_idx = train_idx.tolist()
            val_idx = val_idx.tolist()

            train_ds = ACDC_ShortAxisDataset(
                img_dir=config.DATA.DATA_PATH, 
                patient_ids=train_idx, 
                mean=-1, 
                std=-1, 
                augmentation=False,
                rand_crop=False,
                max_samples=-1,     # max samples is already set by the dataset definition above
                resize=config.MODEL.NAME == 'SwinUNETR'
            )
            val_ds = ACDC_ShortAxisDataset(
                img_dir=config.DATA.DATA_PATH, 
                patient_ids=val_idx, 
                mean=74.29, 
                std=81.47, 
                augmentation=False, 
                rand_crop=False,
                max_samples=-1,      # max samples is already set by the dataset definition above
                resize=config.MODEL.NAME == 'SwinUNETR'
            )
            train_sampler = None
            val_sampler = None
            shuffle = True
        elif config.DATA.DATASET == 'platelet':
            # Convert id tensors to list
            train_idx = train_idx.tolist()
            val_idx = val_idx.tolist()

            train_ds = PlateletDataset(
                img_file=os.path.join(config.DATA.DATA_PATH, "train-images.tif"),
                label_file=os.path.join(config.DATA.DATA_PATH, "train-labels.tif"),
                frame_ids=train_idx,
                augmentation=True,
                patch_width=config.DATA.IMG_SIZE[0],
                patch_height=config.DATA.IMG_SIZE[1],
            )
            val_ds = PlateletDataset(
                img_file=os.path.join(config.DATA.DATA_PATH, "train-images.tif"),
                label_file=os.path.join(config.DATA.DATA_PATH, "train-labels.tif"),
                frame_ids=val_idx,
                augmentation=True,
                patch_width=config.DATA.IMG_SIZE[0],
                patch_height=config.DATA.IMG_SIZE[1],
            )

            train_sampler = None
            val_sampler = None
            shuffle = True
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
            eight_connectivity=config.LOSS.EIGHT_CONNECTIVITY
        )
        topograph_metric = TopographMetric(
            num_processes=16,
            ignore_background=not config.DATA.INCLUDE_BACKGROUND,
            sphere=False,
            eight_connectivity=config.LOSS.EIGHT_CONNECTIVITY
        )

        # Create model
        model = train_utils.select_model(config, device)
        
        # Create loss function
        loss_function, exp_name = train_utils.loss_selection(config, args)

        if args.sweep:
            exp_name = 'sweep_' + exp_id + "_" + exp_name
        # Create wandb run
        elif not args.disable_wandb and not args.sweep:
            # start a new wandb run to track this script
            wandb_run = wandb.init(
                # set the wandb project where this run will be logged
                project="multiclass-bettimatching",
                
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

        match config.TRAIN.OPTIMIZER:
            case "adam":
                optimizer = torch.optim.Adam(model.parameters(), lr=config.TRAIN.LR, weight_decay=getattr(config.LOSS, 'WEIGHT_DECAY', 0.0))
            case "adamw":
                optimizer = torch.optim.AdamW(model.parameters(), lr=config.TRAIN.LR, weight_decay=getattr(config.LOSS, 'WEIGHT_DECAY', 0.01))
            case "sam":
                optimizer = SAM(model.parameters(), torch.optim.Adam, rho=0.05, adaptive=True, lr=config.TRAIN.LR)
            case _:
                raise Exception('ERROR: Optimizer not recognized')

        scheduler = train_utils.select_lr_scheduler(config, optimizer)
        
        # Resume training
        last_epoch = 0
        if args.resume:
            dic = torch.load(args.resume)
            model.load_state_dict(dic['model'])
            optimizer.load_state_dict(dic['optimizer'])
            scheduler.load_state_dict(dic['scheduler'])
            last_epoch = int(scheduler.last_epoch)
            
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

                    def closure():
                        print("start closure")
                        loss, dic = loss_function(model(inputs), labels, alpha=alpha)
                        loss.backward()
                        return loss
                else:
                    loss, dic = loss_function(outputs, labels)

                    def closure():
                        loss, dic = loss_function(model(inputs), labels)
                        loss.backward()
                        return loss

                loss.backward()
                
                if config.TRAIN.OPTIMIZER == 'sam':
                    optimizer.step(closure)
                else:
                    optimizer.step()

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
                        if args.integrate_test and not args.test_best:
                            print("starting test loop")
                            evaluate_model(
                                model, 
                                test_loader, 
                                device,
                                config.DATA.INCLUDE_BACKGROUND,
                                config.LOSS.EIGHT_CONNECTIVITY,
                                logging=not args.disable_wandb,
                                mask_background=False,
                                reproducible=False  # This is set to False because it would reset all seeds, possibly disturbing the training process
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

                    # Compute improvement
                    # dice_improvement = (dice_score - best_dice) / best_dice # positive means improvement
                    # bm_improvement = (best_bm - bm) / best_bm # positive means improvements; this is turned around because we want to minimize the bm loss

                    # #Early stopping
                    # if (
                    #     epoch > 0.3 * config.TRAIN.MAX_EPOCHS and
                    #     epoch > config.LOSS.ALPHA_WARMUP_EPOCHS and 
                    #     dice_improvement + bm_improvement < 0 and 
                    #     epoch - best_metric_epoch > 5 * config.TRAIN.VAL_INTERVAL
                    # ):
                    #     print(f"Early stopping at epoch: {epoch+1}")
                    #     break

                    # if epoch > 0.3 * config.TRAIN.MAX_EPOCHS and epoch - best_metric_epoch > 5 * config.TRAIN.VAL_INTERVAL:
                    #     print(f"Early stopping at epoch: {epoch+1}")
                    #     break

            scheduler.step()

        print(f"fold {fold+1} completed, best_metric: {best_combined_val_metric:.4f} at epoch: {best_metric_epoch}")


        if args.integrate_test and args.test_best:
            print("starting test loop")
            model = train_utils.select_model(config, device)

            dic = torch.load('./models/'+config.DATA.DATASET+'/'+exp_name+'/best_model_dict.pth')
            model.load_state_dict(dic['model'])

            print("loaded model checkpoint from epoch", dic['scheduler']['last_epoch'])

            model.to(device)

            evaluate_model(
                model, 
                test_loader, 
                device,
                config.DATA.INCLUDE_BACKGROUND,
                config.LOSS.EIGHT_CONNECTIVITY,
                logging=not args.disable_wandb,
                mask_background=False,
                reproducible=True,
                seed=config.TRAIN.SEED
            )

        if args.sweep and 4 <= train_utils.get_num_better_runs(run.sweep_id, best_combined_val_metric, "val/best_combined_metric"):
            # delete checkpoint of the current run
            os.remove('./models/'+config.DATA.DATASET+'/'+exp_name+'/best_model_dict.pth')


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
    config = train_utils.dict2obj(config)
    
    main(args, config)