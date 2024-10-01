from typing import Union, List
import torch
from monai.metrics.utils import do_metric_reduction, is_binary_tensor
from monai.utils.enums import MetricReduction
import numpy as np
import betti_matching
from monai.metrics.metric import CumulativeIterationMetric

from losses.betti_losses import FastBettiMatchingLoss

class BettiNumberMetric(CumulativeIterationMetric):
    def __init__(self, ignore_background=False, num_processes=1, eight_connectivity=True) -> None:
        super().__init__()
        self.reduction = MetricReduction.MEAN
        self.BM_loss = FastBettiMatchingLoss(
            num_processes=num_processes,
        )
        self.eight_connectivity = eight_connectivity
        self.ignore_background = ignore_background

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
        
        skip_index = 1 if self.ignore_background else 0 # skip the background class if it is ignored
        
        b0_errors = torch.zeros((y_pred.shape[0], y_pred.shape[1] - skip_index))
        b1_errors = torch.zeros((y_pred.shape[0], y_pred.shape[1] - skip_index))
        bm_losses = torch.zeros((y_pred.shape[0], y_pred.shape[1] - skip_index))
        normalized_bm_losses = torch.zeros((y_pred.shape[0], y_pred.shape[1] - skip_index))

        if not self.eight_connectivity:
            # inverse bg and fg of the input and target
            inv_y_pred = 1 - y_pred
            inv_y = 1 - y
        else:
            inv_y_pred = y_pred
            inv_y = y
        
        # for each image in batch and each class, compute the betti numbers
        for i in range(y_pred.shape[0]):
            for c in range(skip_index, y_pred.shape[1]):
                predictions_cpu_batch = [np.ascontiguousarray(inv_y_pred[i, c].cpu())]
                targets_cpu_batch = [np.ascontiguousarray(inv_y[i, c].cpu())]

                results = betti_matching.compute_matching(predictions_cpu_batch, targets_cpu_batch, return_target_unmatched_pairs=True)
                dim_0, dim_1 = (0, 1) if not self.eight_connectivity else (1, 0)

                # Compute betti number error
                b0_pred = results[0].num_matches_by_dim[dim_0] + results[0].num_unmatched_prediction_by_dim[dim_0] + 1
                b1_pred = results[0].num_matches_by_dim[dim_1] + results[0].num_unmatched_prediction_by_dim[dim_1]
                b0_label = results[0].num_matches_by_dim[dim_0] + results[0].num_unmatched_target_by_dim[dim_0] + 1
                b1_label = results[0].num_matches_by_dim[dim_1] + results[0].num_unmatched_target_by_dim[dim_1]

                b0_errors[i, c - skip_index] = abs(b0_pred - b0_label)
                b1_errors[i, c - skip_index] = abs(b1_pred - b1_label)

                # Compute betti matching error
                bm_losses[i, c - skip_index] = self.BM_loss._betti_matching_loss(y_pred[i, c], y[i, c], results[0])

                # Compute normalized betti matching error
                normalized_bm_losses[i, c - skip_index] = min(1, (bm_losses[i, c - skip_index]) / (b0_label + b1_label))

        # convert back to tensor and return
        return [b0_errors, b1_errors, bm_losses, normalized_bm_losses]

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        data = self.get_buffer()
        # check if data is of type list
        if not isinstance(data, List):
            raise ValueError("the data to aggregate must be a list.")
        if not isinstance(data[0], torch.Tensor) and not isinstance(data[1], torch.Tensor) and not isinstance(data[2], torch.Tensor):
            raise ValueError("the elements of the list to aggregate must be PyTorch Tensors.")
        # do metric reduction
        b0, _ = do_metric_reduction(data[0], reduction=self.reduction or reduction)
        b1, _ = do_metric_reduction(data[1], reduction=self.reduction or reduction)
        bm, _ = do_metric_reduction(data[2], reduction=self.reduction or reduction)
        norm_bm, _ = do_metric_reduction(data[3], reduction=self.reduction or reduction)
        return b0, b1, bm, norm_bm