import monai
import torch
import wandb
from losses.utils import ThresholdDistribution
import json


def multiclasses_dataset_selection(config, integrate_test=True):
    import os
    from datasets.platelet import PlateletDataset
    from datasets.topcow import TopCowDataset
    from datasets.m2nist import M2NIST
    from datasets.octa import Octa500Dataset
    from datasets.acdc import ACDC_ShortAxisDataset
    # Load the dataset
    if config.DATA.DATASET == 'm2nist':
        train_dataset = M2NIST(data_path=config.DATA.DATA_PATH, augmentation=False, max_samples=config.DATA.NUM_SAMPLES)
    elif config.DATA.DATASET == 'octa500_3mm':
        config.DATA.DATA_PATH = config.DATA.IMG_PATH
        train_dataset = Octa500Dataset(
            img_dir=config.DATA.IMG_PATH, 
            artery_dir=config.DATA.ARTERY_PATH, 
            vein_dir=config.DATA.VEIN_PATH, 
            augmentation=False, 
            max_samples=config.DATA.NUM_SAMPLES,
            rotation_correction=True # The 3mm dataset is rotated 90 degrees
        )
        if integrate_test:
            test_dataset = Octa500Dataset(
                img_dir=config.DATA.TEST_PATH, 
                artery_dir=config.DATA.ARTERY_PATH, 
                vein_dir=config.DATA.VEIN_PATH, 
                augmentation=False, 
                max_samples=-1,
                rotation_correction=True # The 3mm dataset is rotated 90 degrees
            )
    elif config.DATA.DATASET == 'octa500_6mm':
        config.DATA.DATA_PATH = config.DATA.IMG_PATH
        train_dataset = Octa500Dataset(
            img_dir=config.DATA.IMG_PATH, 
            artery_dir=config.DATA.ARTERY_PATH, 
            vein_dir=config.DATA.VEIN_PATH, 
            augmentation=True, 
            max_samples=config.DATA.NUM_SAMPLES,
            rotation_correction=False # The 6mm dataset is not rotated
        )
        if integrate_test:
            test_dataset = Octa500Dataset(
                img_dir=config.DATA.TEST_PATH, 
                artery_dir=config.DATA.ARTERY_PATH, 
                vein_dir=config.DATA.VEIN_PATH, 
                augmentation=False, 
                max_samples=-1,
                rotation_correction=False # The 6mm dataset is not rotated
            )
    elif config.DATA.DATASET == 'topcow':
        config.DATA.DATA_PATH = config.DATA.IMG_PATH
        train_dataset = TopCowDataset(
            img_dir=config.DATA.IMG_PATH, 
            label_dir=config.DATA.LABEL_PATH, 
            augmentation=True, 
            max_samples=config.DATA.NUM_SAMPLES,
            width=config.DATA.IMG_SIZE[0],
            height=config.DATA.IMG_SIZE[1]
        )
        if integrate_test:
            test_dataset = TopCowDataset(
                img_dir=config.DATA.TEST_PATH, 
                label_dir=config.DATA.LABEL_PATH, 
                augmentation=False, 
                max_samples=-1,
                width=config.DATA.IMG_SIZE[0],
                height=config.DATA.IMG_SIZE[1]
            )
    elif config.DATA.DATASET == 'ACDC_sa':
        train_dataset = range(0, config.DATA.NUM_SAMPLES)
        if integrate_test:
            test_dataset = ACDC_ShortAxisDataset(
                img_dir=config.DATA.TEST_PATH,
                patient_ids=list(range(101, 151)),
                mean=74.29, 
                std=81.47, 
                rand_crop=False,
                augmentation=False, 
                max_samples=-1,
                resize=config.MODEL.NAME == 'SwinUNETR'
            )
    elif config.DATA.DATASET == 'platelet':
        train_dataset = range(config.DATA.NUM_SAMPLES)
        if integrate_test:
            test_dataset = PlateletDataset(
                img_file=os.path.join(config.DATA.DATA_PATH, "eval-images.tif"),
                label_file=os.path.join(config.DATA.DATA_PATH, "eval-labels.tif"),
                frame_ids=[],
                augmentation=False,
                patch_width=config.DATA.IMG_SIZE[0],
                patch_height=config.DATA.IMG_SIZE[1],
            )
    else:
        raise Exception('ERROR: Dataset not implemented')
    
    if integrate_test:
        return train_dataset, test_dataset
    else:
        return train_dataset


def binary_dataset_selection(config, args):
    from datasets.cremi import Cremi
    from datasets.drive import Drive
    from datasets.elegans_cells import Elegans
    from datasets.roads import Roads
    from datasets.fives import Fives
    from datasets.cell_tracking import Tracking
    from datasets.buildings import Buildings
     # Load the dataset
    if config.DATA.DATASET == 'cremi':
        train_dataset = Cremi(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=True,
            five_crop=config.DATA.FIVE_CROPS
        )
        val_dataset = Cremi(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=False,
            five_crop=config.DATA.FIVE_CROPS
        )
        if args.integrate_test:
            test_dataset = Cremi(
                data_path=config.DATA.TEST_PATH,
                crop_size=config.DATA.IMG_SIZE,
                max_samples=-1,
                normalize=False,
                rescale=config.DATA.RESCALE,
                augment=False,
                five_crop=config.DATA.FIVE_CROPS
            )
    elif config.DATA.DATASET == 'roads':
        train_dataset = Roads(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=True,
            five_crop=config.DATA.FIVE_CROPS
        )
        val_dataset = Roads(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=False,
            five_crop=config.DATA.FIVE_CROPS
        )
        if args.integrate_test:
            test_dataset = Roads(
                data_path=config.DATA.TEST_PATH,
                crop_size=config.DATA.IMG_SIZE,
                max_samples=-1,
                normalize=False,
                rescale=config.DATA.RESCALE,
                augment=False,
                five_crop=config.DATA.FIVE_CROPS
            )
    elif config.DATA.DATASET == 'buildings':
        train_dataset = Buildings(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=True,
            five_crop=config.DATA.FIVE_CROPS
        )
        val_dataset = Buildings(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=False,
            five_crop=config.DATA.FIVE_CROPS
        )
        if args.integrate_test:
            test_dataset = Buildings(
                data_path=config.DATA.TEST_PATH,
                crop_size=config.DATA.IMG_SIZE,
                max_samples=-1,
                normalize=False,
                rescale=config.DATA.RESCALE,
                augment=False,
                five_crop=config.DATA.FIVE_CROPS
            )
    elif config.DATA.DATASET == 'drive':
        train_dataset = Drive(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            rescale=config.DATA.RESCALE,
            augment=True,
            five_crop=config.DATA.FIVE_CROPS
        )
        val_dataset = Drive(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            rescale=config.DATA.RESCALE,
            augment=False,
            five_crop=config.DATA.FIVE_CROPS
        )
        if args.integrate_test:
            test_dataset = Drive(
                data_path=config.DATA.TEST_PATH,
                crop_size=config.DATA.IMG_SIZE,
                max_samples=-1,
                rescale=config.DATA.RESCALE,
                augment=False,
                five_crop=config.DATA.FIVE_CROPS
            )
    elif config.DATA.DATASET == 'elegans':
        train_dataset = Elegans(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            rescale=config.DATA.RESCALE,
            augment=True
        )
        val_dataset = Elegans(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            rescale=config.DATA.RESCALE,
            augment=False
        )
        if args.integrate_test:
            test_dataset = Elegans(
                data_path=config.DATA.TEST_PATH,
                crop_size=config.DATA.IMG_SIZE,
                max_samples=-1,
                rescale=config.DATA.RESCALE,
                augment=False
            )
    elif config.DATA.DATASET == 'fives':
        train_dataset = Fives(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=True,
            five_crop=config.DATA.FIVE_CROPS
        )
        val_dataset = Fives(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=False,
            five_crop=config.DATA.FIVE_CROPS
        )
        if args.integrate_test:
            test_dataset = Fives(
                data_path=config.DATA.TEST_PATH,
                crop_size=config.DATA.IMG_SIZE,
                max_samples=-1,
                normalize=False,
                rescale=config.DATA.RESCALE,
                augment=False,
                five_crop=config.DATA.FIVE_CROPS
            )
    elif config.DATA.DATASET == 'tracking':
        train_dataset = Tracking(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=True,
            five_crop=config.DATA.FIVE_CROPS
        )
        val_dataset = Tracking(
            data_path=config.DATA.DATA_PATH,
            crop_size=config.DATA.IMG_SIZE,
            max_samples=config.DATA.NUM_SAMPLES,
            normalize=False,
            rescale=config.DATA.RESCALE,
            augment=False,
            five_crop=config.DATA.FIVE_CROPS
        )
        if args.integrate_test:
            test_dataset = Tracking(
                data_path=config.DATA.TEST_PATH,
                crop_size=config.DATA.IMG_SIZE,
                max_samples=20,
                normalize=False,
                rescale=config.DATA.RESCALE,
                augment=False,
                five_crop=config.DATA.FIVE_CROPS
            )
    else:
        raise Exception('ERROR: Dataset not implemented')
    
    if args.integrate_test:
        return train_dataset, val_dataset, test_dataset
    else:
        return train_dataset, val_dataset
    


def loss_selection(config, args):
    from losses.hutopo import MulticlassDiceWassersteinLoss
    from losses.mosin import MulticlassDiceMOSIN
    from losses.topograph import DiceTopographLoss
    from losses.dice_losses import Multiclass_CLDice
    from losses.betti_losses import FastMulticlassDiceBettiMatchingLoss
    from losses.betti_losses import FiltrationType, DiceType
    from losses.utils import AggregationType
    # Loss function choice
    if config.LOSS.USE_LOSS == 'Dice':
        exp_name = config.LOSS.USE_LOSS
        # We're using the CLDice loss with alpha=0 which is equivalent to the Dice loss
        loss_function = Multiclass_CLDice(
            softmax=not config.LOSS.ONE_VS_REST, 
            include_background=True, # irrelevant because Dice always uses background 
            smooth=1e-5, 
            alpha=0.0,
            convert_to_one_vs_rest=config.LOSS.ONE_VS_REST,
            batch=True
        )
    elif config.LOSS.USE_LOSS == 'ClDice':
        exp_name = config.LOSS.USE_LOSS
        loss_function = Multiclass_CLDice(
            softmax=not config.LOSS.ONE_VS_REST, 
            include_background=config.DATA.INCLUDE_BACKGROUND, 
            smooth=1e-5, 
            alpha=config.LOSS.CLDICE_ALPHA, 
            iter_=5, 
            convert_to_one_vs_rest=config.LOSS.ONE_VS_REST,
            batch=True
        )
    elif config.LOSS.USE_LOSS == 'HuTopo':
        exp_name = config.LOSS.USE_LOSS
        loss_function = MulticlassDiceWassersteinLoss(
            filtration_type=FiltrationType.SUBLEVEL if config.LOSS.EIGHT_CONNECTIVITY else FiltrationType.SUPERLEVEL, 
            num_processes=16,
            dice_type=DiceType[config.LOSS.DICE_TYPE.upper()],
            convert_to_one_vs_rest=config.LOSS.ONE_VS_REST,
            cldice_alpha=config.LOSS.CLDICE_ALPHA,
            ignore_background=not config.DATA.INCLUDE_BACKGROUND,
        )
    elif config.LOSS.USE_LOSS == 'FastMulticlassDiceBettiMatching':
        exp_name = config.LOSS.USE_LOSS+'_eight_connectivity'+str(config.LOSS.EIGHT_CONNECTIVITY)+'_alpha_'+str(config.LOSS.ALPHA)
        loss_function = FastMulticlassDiceBettiMatchingLoss(
            filtration_type=FiltrationType.SUBLEVEL if config.LOSS.EIGHT_CONNECTIVITY else FiltrationType.SUPERLEVEL, 
            num_processes=16,
            dice_type=DiceType[config.LOSS.DICE_TYPE.upper()],
            convert_to_one_vs_rest=config.LOSS.ONE_VS_REST,
            cldice_alpha=config.LOSS.CLDICE_ALPHA,
            push_unmatched_to_1_0=config.LOSS.PUSH_UNMATCHED_TO_1_0,
            barcode_length_threshold=config.LOSS.BARCODE_LENGTH_THRESHOLD,
            ignore_background=not config.DATA.INCLUDE_BACKGROUND,
            topology_weights=config.LOSS.TOPOLOGY_WEIGHTS,
        )
    elif config.LOSS.USE_LOSS == 'Topograph':
        exp_name = config.LOSS.USE_LOSS+'_alpha_'+str(config.LOSS.ALPHA)
        loss_function = DiceTopographLoss(
            softmax=True,
            dice_type=DiceType[config.LOSS.DICE_TYPE.upper()],
            num_processes=16,
            cldice_alpha=config.LOSS.CLDICE_ALPHA,
            include_background=not config.DATA.INCLUDE_BACKGROUND,
            use_c=not args.no_c,
            eight_connectivity=config.LOSS.EIGHT_CONNECTIVITY,
            aggregation=AggregationType[getattr(config.LOSS, "AGGREGATION_TYPE", "mean").upper()],
            thres_distr=ThresholdDistribution[getattr(config.LOSS, "THRES_DISTR", "none").upper()],
            thres_var=getattr(config.LOSS, "THRES_VAR", 0.0),
        )
    elif config.LOSS.USE_LOSS == 'MOSIN':
        exp_name = config.LOSS.USE_LOSS
        loss_function = MulticlassDiceMOSIN(
            dice_type=DiceType[config.LOSS.DICE_TYPE.upper()],
            convert_to_one_vs_rest=config.LOSS.ONE_VS_REST,
            cldice_alpha=config.LOSS.CLDICE_ALPHA,
            ignore_background=not config.DATA.INCLUDE_BACKGROUND,
        )
    else:
        raise Exception('ERROR: Loss function not implemented')
    

    return loss_function, exp_name

def select_lr_scheduler(config, optimizer):
    main_warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
        optimizer, 
        factor=0.2, 
        total_iters=5,
        last_epoch=-1
    )

    if config.LOSS.USE_LOSS == "FastMulticlassDiceBettiMatching" or config.LOSS.USE_LOSS == "HuTopo" or config.LOSS.USE_LOSS == "Topograph" or config.LOSS.USE_LOSS == "MOSIN":
        pre_warmup_scheduler = torch.optim.lr_scheduler.ConstantLR(
            optimizer, 
            factor=0.5, 
            total_iters=3,
            last_epoch=-1
        )
        schedulers = [pre_warmup_scheduler, main_warmup_scheduler]
        milestones = [config.LOSS.ALPHA_WARMUP_EPOCHS, config.LOSS.ALPHA_WARMUP_EPOCHS + 5]
    else:
        schedulers = [main_warmup_scheduler]
        milestones = [config.TRAIN.MAX_EPOCHS // 10]

    match config.TRAIN.LR_SCHEDULE:
        case "constant":
            scheduler = torch.optim.lr_scheduler.ConstantLR(
                optimizer, 
                factor=1, 
                total_iters=0,
                last_epoch=-1
            )
        case "cosine_annealing":
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
                optimizer, 
                T_max=config.TRAIN.MAX_EPOCHS,
                eta_min=0.05 * config.TRAIN.LR,
                last_epoch=-1
            )
            schedulers.append(cosine_scheduler)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers, 
                milestones
            )
        case "cosine_restarts":
            cosine_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
                optimizer, 
                T_0=(config.TRAIN.MAX_EPOCHS // 10), 
                T_mult=2,
                eta_min=0.05 * config.TRAIN.LR,
                last_epoch=-1
            )
            schedulers.append(cosine_scheduler)
            scheduler = torch.optim.lr_scheduler.SequentialLR(
                optimizer, 
                schedulers, 
                milestones
            )
        case _:
            raise Exception('ERROR: Learning rate scheduler not recognized')
        
    return scheduler

def select_model(config, device):
    # Create model
    if config.MODEL.NAME and config.MODEL.NAME == 'SwinUNETR':
        model = monai.networks.nets.SwinUNETR(
            img_size=(160,160),
            in_channels=1,
            out_channels=config.DATA.OUT_CHANNELS,
            spatial_dims=2,
            depths=(2,2,2,2),
            num_heads=(3,6,12, 24),
        ).to(device)
    else:
        model = monai.networks.nets.UNet(
            spatial_dims=2,
            in_channels=1,
            out_channels=config.DATA.OUT_CHANNELS,
            channels=config.MODEL.CHANNELS,
            strides=[2] + [1 for _ in range(len(config.MODEL.CHANNELS) - 2)],
            num_res_units=config.MODEL.NUM_RES_UNITS,
        ).to(device)

    return model

def get_num_better_runs(sweep_id, current_best_metric, metric_name):
    import wandb
    api = wandb.Api()
    sweep = api.sweep(sweep_id)
    runs = sweep.runs

    num_better = 0

    for run in runs:
        summary = run.summary
        print(run.name, " has: ", summary)
        if metric_name in summary and summary[metric_name] > current_best_metric:
            num_better += 1
            print(run.name, " is better with: ", summary[metric_name])
    
    return num_better

def lr_warmup(epoch, warmup_epochs, base_lr, factor):
    if epoch < warmup_epochs:
        return base_lr * factor
    else:
        return base_lr

def get_config_from_sweep(config):
    config.TRAIN.LR = wandb.config.lr
    config.MODEL.CHANNELS = wandb.config.channels
    config.MODEL.NUM_RES_UNITS = wandb.config.num_res_units
    config.TRAIN.BATCH_SIZE = wandb.config.batch_size
    config.LOSS.DICE_TYPE = wandb.config.dice_type
    config.LOSS.ONE_VS_REST = getattr(wandb.config, 'one_vs_rest', False)
    config.LOSS.ALPHA = getattr(wandb.config, 'alpha', 0.0)
    config.LOSS.CLDICE_ALPHA = getattr(wandb.config, 'cldice_alpha', 0.0)
    config.LOSS.ALPHA_WARMUP_EPOCHS = getattr(wandb.config, 'alpha_warmup_epochs', 0)
    config.LOSS.PUSH_UNMATCHED_TO_1_0 = getattr(wandb.config, 'push_unmatched_to_1_0', False)
    config.LOSS.BARCODE_LENGTH_THRESHOLD = getattr(wandb.config, 'barcode_length_threshold', 0.0)
    config.LOSS.TOPOLOGY_WEIGHTS = (getattr(wandb.config, 'weight_matched', 1.0), getattr(wandb.config, 'weight_unmatched_pred', 1.0))
    config.LOSS.THRES_DISTR = getattr(wandb.config, 'thres_distr', "none")
    config.LOSS.THRES_VAR = getattr(wandb.config, 'thres_var', 0.0)
    config.LOSS.AGGREGATION_TYPE = getattr(wandb.config, 'aggregation_type', 'mean')
    config.TRAIN.OPTIMIZER = getattr(wandb.config, 'optimizer', 'adam')
    config.TRAIN.WEIGHT_DECAY = getattr(wandb.config, 'weight_decay', 0.0)
    config.TRAIN.LR_SCHEDULE = getattr(wandb.config, 'lr_schedule', 'constant')
    
    return config

def set_default_values(config):
    config.LOSS.ONE_VS_REST = getattr(config.LOSS, 'ONE_VS_REST', False)
    config.LOSS.ALPHA = getattr(config.LOSS, 'ALPHA', 0.0)
    config.LOSS.CLDICE_ALPHA = getattr(config.LOSS, 'CLDICE_ALPHA', 0.0)
    config.LOSS.ALPHA_WARMUP_EPOCHS = getattr(config.LOSS, 'ALPHA_WARMUP_EPOCHS', 0)
    config.LOSS.PUSH_UNMATCHED_TO_1_0 = getattr(config.LOSS, 'PUSH_UNMATCHED_TO_1_0', False)
    config.LOSS.BARCODE_LENGTH_THRESHOLD = getattr(config.LOSS, 'BARCODE_LENGTH_THRESHOLD', 0.0)
    config.LOSS.TOPOLOGY_WEIGHTS = getattr(config.LOSS, 'TOPOLOGY_WEIGHTS', (1.0, 1.0))
    config.LOSS.THRES_DISTR = getattr(config.LOSS, 'THRES_DISTR', "none")
    config.LOSS.THRES_VAR = getattr(config.LOSS, 'THRES_VAR', 0.0)
    config.LOSS.AGGREGATION_TYPE = getattr(config.LOSS, 'AGGREGATION_TYPE', 'mean')
    config.TRAIN.OPTIMIZER = getattr(config.TRAIN, 'OPTIMIZER', 'adam')
    config.TRAIN.WEIGHT_DECAY = getattr(config.LOSS, 'WEIGHT_DECAY', 0.0)
    config.TRAIN.LR_SCHEDULE = getattr(config.TRAIN, 'LR_SCHEDULE', 'constant')

    return config

class obj:
    def __init__(self, dict1):
        self.__dict__.update(dict1)
        
def dict2obj(dict1):
    return json.loads(json.dumps(dict1), object_hook=obj)