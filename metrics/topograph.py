from typing import Union, List
import torch
from monai.metrics.utils import do_metric_reduction, is_binary_tensor
from monai.utils.enums import MetricReduction
from monai.metrics.metric import CumulativeIterationMetric
import torch.multiprocessing as mp
import torch.nn.functional as F

from losses.utils import new_compute_diffs, new_compute_diag_diffs
from losses.topograph import single_sample_class_metric

class TopographMetric(CumulativeIterationMetric):
    def __init__(self, num_processes, ignore_background=False, sphere=False, eight_connectivity = True) -> None:
        super().__init__()
        self.reduction = MetricReduction.MEAN
        self.num_processes = num_processes
        self.ignore_background = ignore_background
        self.sphere = sphere
        self.eight_connectivity = eight_connectivity
        if self.num_processes > 1:
            self.pool = mp.Pool(num_processes)
        #self.topograph_criterium = MulticlassTopoGraphParallel(softmax=False, num_processes=num_processes, include_background=True)

    def _compute_tensor(self, y_pred: torch.Tensor, y: torch.Tensor):  # type: ignore
        """
        Args:
            y_pred: input data to compute, typical segmentation model output.
                It must be one-hot format and first dim is batch, second dim is classes, example shape: [16, 11, 32, 32]. The values
                should be binarized. The first class is considered as background and is ignored.
            y: ground truth to compute mean dice metric. It must be one-hot format and first dim is batch.
                The values should be binarized. The first class is considered as background and is ignored.

        Raises:
            ValueError: when `y` is not a binarized tensor.
            ValueError: when `y_pred` has less than three dimensions.
        """
        is_binary_tensor(y_pred, "y_pred")
        is_binary_tensor(y, "y")

        dims = y_pred.ndimension()
        if dims != 4:
            raise ValueError(f"y_pred should have 4 dimensions (batch, channel, height, width), got {dims}.")

        y_pred = y_pred.float()
        y = y.float()

        num_classes = y_pred.shape[1]

        single_calc_inputs = []
        skip_index = 0 if not self.ignore_background else 1

        paired_imgs = y_pred + 2 * y
        diag_val_1, diag_val_2 = (-4, 16) if self.eight_connectivity else (16, -4)

        paired_imgs[paired_imgs==0] = diag_val_1
        paired_imgs[paired_imgs==3] = diag_val_2

        # get critical nodes for each class
        for class_index in range(skip_index, num_classes):
            if self.sphere:
                paired_imgs = F.pad(paired_imgs, (1, 1, 1, 1), value=-4)
                bin_preds = F.pad(y_pred[:,class_index].clone().int(), (1, 1, 1, 1), value=0)
                bin_gts = F.pad(y[:,class_index].clone().int(), (1, 1, 1, 1), value=0)
            else:
                bin_preds = y_pred[:,class_index].clone().int()
                bin_gts = y[:,class_index].clone().int()

            h_diff, v_diff = new_compute_diffs(paired_imgs[:, class_index])
            diagr, diagl, special_diag_r, special_diag_l = new_compute_diag_diffs(paired_imgs[:, class_index], th=7)

            # move all to cpu
            #TODO: Fix device handling
            bin_preds = bin_preds.cpu().numpy()
            bin_gts = bin_gts.cpu().numpy()
            h_diff = h_diff.cpu().numpy()
            v_diff = v_diff.cpu().numpy()
            diagr = diagr.cpu().numpy()
            diagl = diagl.cpu().numpy()
            special_diag_r = special_diag_r.cpu().numpy()
            special_diag_l = special_diag_l.cpu().numpy()

            for i in range(y_pred.shape[0]):
                # create dict with function arguments
                single_calc_input = {
                    "argmax_pred": bin_preds[i],
                    "argmax_gt": bin_gts[i],
                    "h_diff": h_diff[i],
                    "v_diff": v_diff[i],
                    "diagr": diagr[i],
                    "diagl": diagl[i],
                    "special_diagr": special_diag_r[i],
                    "special_diagl": special_diag_l[i],
                    "sample_no": i,
                }
                single_calc_inputs.append(single_calc_input)

        if self.num_processes > 1:
            chunksize = len(single_calc_inputs) // self.num_processes if len(single_calc_inputs) > self.num_processes else 1
            errors = self.pool.imap_unordered(single_sample_class_metric, single_calc_inputs, chunksize=chunksize)
        else:
            errors = map(single_sample_class_metric, single_calc_inputs)

        topograph_score = torch.zeros((y_pred.shape[0], 1))
        
        for error_count, sample_no in errors:
            topograph_score[sample_no] += error_count

        return topograph_score

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        # check type
        if not isinstance(data, torch.Tensor):
            raise ValueError("the elements of the list to aggregate must be PyTorch Tensors.")
        # do metric reduction
        topograph_losses, _ = do_metric_reduction(data, reduction=self.reduction or reduction)
        return topograph_losses