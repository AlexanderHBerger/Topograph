from typing import Union

from skimage.morphology import skeletonize
import numpy as np
import torch

from monai.metrics.utils import do_metric_reduction, is_binary_tensor
from monai.utils.enums import MetricReduction

from monai.metrics.metric import CumulativeIterationMetric

def cl_score(v, s):
    """[this function computes the skeleton volume overlap]

    Args:
        v ([bool]): [image]
        s ([bool]): [skeleton]

    Returns:
        [float]: [computed skeleton volume intersection]
    """
    return np.sum(v*s)/np.sum(s)

class ClDiceMetric(CumulativeIterationMetric):
    def __init__(self, ignore_background=False) -> None:
        super().__init__()
        self.ignore_background = ignore_background
        self.reduction = MetricReduction.MEAN

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
        
        # Convert to numpy
        y_pred = y_pred.cpu().numpy()
        y = y.cpu().numpy()

        cldice_scores = torch.zeros((y_pred.shape[0], y_pred.shape[1] - skip_index))

        for i in range(y_pred.shape[0]):
            for c in range(skip_index, y_pred.shape[1]): # ignore background
                # Skeletonize pred and label
                skeletonized_pred = skeletonize(y_pred[i, c])
                skeletonized_label = skeletonize(y[i, c])

                # Compute cldice score
                tprec = cl_score(y_pred[i, c], skeletonized_label)
                tsens = cl_score(y[i, c], skeletonized_pred)

                # Build harmonic mean
                cldice_scores[i, c - skip_index] = 2*tprec*tsens/(tprec+tsens)
        
        # convert back to tensor and return
        return cldice_scores

    def aggregate(self, reduction: Union[MetricReduction, str, None] = None):  # type: ignore
        """
        Execute reduction logic for the output of `compute_meandice`.

        Args:
            reduction: define mode of reduction to the metrics, will only apply reduction on `not-nan` values,
                available reduction modes: {``"none"``, ``"mean"``, ``"sum"``, ``"mean_batch"``, ``"sum_batch"``,
                ``"mean_channel"``, ``"sum_channel"``}, default to `self.reduction`. if "none", will not do reduction.

        """
        data = self.get_buffer()
        if not isinstance(data, torch.Tensor):
            raise ValueError("the data to aggregate must be PyTorch Tensor.")

        # do metric reduction
        f, _ = do_metric_reduction(data, reduction=self.reduction or reduction) # type: ignore
        return f