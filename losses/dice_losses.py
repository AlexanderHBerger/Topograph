from typing import List
import typing
import torch
import torch.nn as nn
import torch.nn.functional as F

from losses.utils import convert_to_one_vs_rest, soft_skel

from torch.nn.modules.loss import _Loss
#if typing.TYPE_CHECKING:
from jaxtyping import Float


class Multiclass_CLDice(_Loss):
    """
    Multiclass_CLDice is a loss function that combines the Dice loss and the CLDice loss for multiclass segmentation tasks.

    Args:
        iter_ (int): Number of iterations for soft skeleton computation. Default is 3.
        alpha (float): Weighting factor for the CLDice loss. 0 makes loss equivalent do Dice loss. Default is 0.5.
        smooth (float): Smoothing factor to avoid division by zero. Default is 1e-5.
        sigmoid (bool): Whether to apply sigmoid activation to the input. Default is False.
        softmax (bool): Whether to apply softmax activation to the input. Default is False.
        include_background (bool): Whether to include the background class in the loss calculation. CLDice component always ignores the background. Default is False.
        convert_to_one_vs_rest (bool): Whether to convert the input to one-vs-rest format. Default is False.
        batch (bool): Whether to include the batch dimension in the reduction. Default is False.

    Attributes:
        include_background (bool): Whether the background class is included in the loss calculation. CLDice component always ignores the background.
        sigmoid (bool): Whether sigmoid activation is applied to the input.
        softmax (bool): Whether softmax activation is applied to the input.
        convert_to_one_vs_rest (bool): Whether the input is converted to one-vs-rest format.
        iter_ (int): Number of iterations for soft skeleton computation.
        smooth (float): Smoothing factor to avoid division by zero.
        alpha (float): Weighting factor for the CLDice loss.

    Methods:
        forward(input, target): Computes the CLDice loss and Dice loss for the given input and target.

    Raises:
        ValueError: If incompatible values are provided for sigmoid, softmax, and convert_to_one_vs_rest.
        ValueError: If softmax=True and the number of channels for the prediction is 1.
        ValueError: If single channel prediction is used and include_background=False.
        AssertionError: If the shape of the ground truth is different from the input shape.

    """

    def __init__(self, weights: List[Float]=[], iter_=3, alpha=0.5, smooth=1e-5, sigmoid=False, softmax=False, include_background=False, convert_to_one_vs_rest=False, batch=False):
        super(Multiclass_CLDice, self).__init__()
        if int(sigmoid) + int(softmax) + int(convert_to_one_vs_rest) > 1:
            raise ValueError("Incompatible values: more than 1 of [sigmoid=True, softmax=True, convert_to_one_vs_rest=True].")
        
        self.include_background = include_background
        self.sigmoid = sigmoid
        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.iter = iter_
        self.smooth = smooth
        self.alpha = alpha
        self.sigmoid = sigmoid
        self.batch = batch
        self.weights = torch.tensor(weights)

    def forward(self, input, target):
        """
        Computes the CLDice loss and Dice loss for the given input and target.

        Args:
            input (torch.Tensor): The predicted segmentation map.
            target (torch.Tensor): The ground truth segmentation map.

        Returns:
            tuple: A tuple containing the total loss and a dictionary of individual loss components.

        Raises:
            ValueError: If softmax=True and the number of channels for the prediction is 1.
            ValueError: If single channel prediction is used and include_background=False.
            AssertionError: If the shape of the ground truth is different from the input shape.

        """
        if (len(self.weights) > 0 and (
            (self.include_background and len(self.weights) != input.shape[1]) or 
            (not self.include_background and len(self.weights) != input.shape[1] - 1))
        ):
            raise ValueError(f"Number of class weights ({len(self.weights)}) must match the number of classes ({input.shape[1]}).")
        elif len(self.weights) > 0:
            # Move weight tensor to correct device and replicate across batch dimension
            self.weights = self.weights.to(input.device)
            if self.batch:
                weights = self.weights.unsqueeze(0)
                weights = weights.expand(input.shape[0], -1)
        
        if self.sigmoid:
            input = torch.sigmoid(input)

        n_pred_ch = input.shape[1]
        if self.softmax:
            if n_pred_ch == 1:
                raise ValueError("softmax=True, but the number of channels for the prediction is 1.")
            else:
                input = torch.softmax(input, 1)
        
        if not self.include_background and n_pred_ch == 1:
            raise ValueError("single channel prediction, `include_background=False` is not a valid combination.")
        
        if self.convert_to_one_vs_rest:
            input = convert_to_one_vs_rest(input)

        if target.shape != input.shape:
            raise AssertionError(f"ground truth has different shape ({target.shape}) from input ({input.shape})")
        
        # reducing only spatial dimensions (not batch nor channels)
        reduce_axis: List[int] = torch.arange(2, len(input.shape)).tolist()
        if self.batch:
            # reducing spatial dimensions and batch
            reduce_axis = [0] + reduce_axis
        
        dic = {}
        
        if self.alpha > 0:
            starting_class = 1 if self.include_background else 0

            # Create soft skeletons
            pred_skeletons = soft_skel(input[:, starting_class:].float(), self.iter)
            target_skeletons = soft_skel(target[:, starting_class:].float(), self.iter)

            # Compute CLDice
            tprec = (torch.sum(torch.multiply(pred_skeletons, target[:, starting_class:]), dim=reduce_axis)+self.smooth)/(torch.sum(pred_skeletons, dim=reduce_axis)+self.smooth)    
            tsens = (torch.sum(torch.multiply(target_skeletons, input[:, starting_class:]), dim=reduce_axis)+self.smooth)/(torch.sum(target_skeletons, dim=reduce_axis)+self.smooth)    
            cl_dice = torch.mean(1.- 2.0*(tprec*tsens)/(tprec+tsens))
            
            # Weighted CLDice
            if len(self.weights) > 0:
                cl_dice = torch.multiply(cl_dice, weights[starting_class:])
        else:
            cl_dice = torch.zeros(size=[1], device=input.device) # TODO: dim should match of actual cl dice calculation

        # Compute Dice
        intersection = torch.sum(target * input, dim=reduce_axis)
        ground_o = torch.sum(target, dim=reduce_axis)
        pred_o = torch.sum(input, dim=reduce_axis)
        denominator = ground_o + pred_o

        dice = 1.0 - (2.0 * intersection + self.smooth) / (denominator + self.smooth) # TODO: check if smooth should be removed in nominator

        # Weighted Dice
        if len(self.weights) > 0:
            dice = torch.multiply(dice, weights)

        # build the mean (across the batch and channel dimensions) of the loss
        dice = torch.mean(dice)
        cl_dice = torch.mean(cl_dice)

        # Total loss
        loss = (1 - self.alpha) * dice + self.alpha * cl_dice

        dic = {}
        dic['dice'] = (1 - self.alpha) * dice
        dic['cldice'] = self.alpha*cl_dice
        return loss, dic 

        if self.reduction == LossReduction.MEAN.value:
            f = torch.mean(f)  # the batch and channel average
        elif self.reduction == LossReduction.SUM.value:
            f = torch.sum(f)  # sum over the batch and channel dims
        elif self.reduction == LossReduction.NONE.value:
            # If we are not computing voxelwise loss components at least
            # make sure a none reduction maintains a broadcastable shape
            broadcast_shape = list(f.shape[0:2]) + [1] * (len(input.shape) - 2)
            f = f.view(broadcast_shape)
        else:
            raise ValueError(f'Unsupported reduction: {self.reduction}, available options are ["mean", "sum", "none"].')

        return f