from __future__ import annotations
import betti_matching
import torch
import numpy as np
import monai
from torch.nn.modules.loss import _Loss
from gudhi import wasserstein

import typing
from losses.dice_losses import Multiclass_CLDice
if typing.TYPE_CHECKING:
    from typing import Tuple, List
    from numpy.typing import NDArray
    LossOutputName = str
    from jaxtyping import Float
    from torch import Tensor

from losses.utils import DiceType, FiltrationType, convert_to_one_vs_rest

class MulticlassDiceWassersteinLoss(_Loss):
    def __init__(self,
                 filtration_type: FiltrationType=FiltrationType.SUPERLEVEL, 
                 dice_type: DiceType=DiceType.CLDICE,
                 num_processes: int=1,
                 convert_to_one_vs_rest: bool = False,
                 cldice_alpha: float = 0.5,
                 ignore_background: bool = False,
                 ) -> None:
        super().__init__()

        if dice_type == DiceType.DICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=not convert_to_one_vs_rest, 
                include_background=True, 
                smooth=1e-5, 
                alpha=0.0,
                convert_to_one_vs_rest=convert_to_one_vs_rest,
                batch=True
            )
        elif dice_type == DiceType.CLDICE:
            self.DiceLoss = Multiclass_CLDice(
                softmax=not convert_to_one_vs_rest, 
                include_background=not ignore_background, 
                smooth=1e-5, 
                alpha=cldice_alpha, 
                iter_=5, 
                convert_to_one_vs_rest=convert_to_one_vs_rest,
                batch=True
            )
        else:
            raise ValueError(f"Invalid dice type: {dice_type}")
        
        self.MulticlassWassersteinloss = MulticlassWassersteinLoss(
            filtration_type=filtration_type, 
            num_processes=num_processes,
            convert_to_one_vs_rest=convert_to_one_vs_rest,
            softmax=not convert_to_one_vs_rest,
            ignore_background=ignore_background,
        )

    def forward(self, 
                prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"], 
                target: Float[torch.Tensor, "batch channel *spatial_dimensions"],
                alpha: float = 0.5
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Compute multiclass Wasserstein losses
        if alpha > 0:
            wasserstein_loss, losses = self.MulticlassWassersteinloss(prediction, target)
            losses = {"single_matches": losses}
        else:
            wasserstein_loss = torch.zeros(1, device=prediction.device)
            losses = {}

        # Multiclass Dice loss
        dice_loss, dic = self.DiceLoss(prediction, target)
        
        losses["dice"] = dic["dice"]
        losses["cldice"] = dic["cldice"]
        losses["wasserstein"] = alpha * wasserstein_loss.item()

        return dice_loss + alpha * wasserstein_loss, losses

class MulticlassWassersteinLoss(_Loss):
    def __init__(self,
                 filtration_type: FiltrationType=FiltrationType.SUPERLEVEL, 
                 num_processes: int=1,
                 convert_to_one_vs_rest: bool = True,
                 softmax: bool = False,
                 ignore_background: bool = False,
                 ) -> None:
        super().__init__()
        if not softmax and not convert_to_one_vs_rest:
            raise ValueError("If softmax is False, convert_to_one_vs_rest must be True")
        if softmax and convert_to_one_vs_rest:
            raise ValueError("If softmax is True, convert_to_one_vs_rest must be False. Softmax is already handled by one vs rest")
        
        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.ignore_background = ignore_background

        self.WassersteinLoss = WassersteinLoss(
            filtration_type=filtration_type, 
            num_processes=num_processes,
        )

    def forward(self, 
                prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"], 
                target: Float[torch.Tensor, "batch channel *spatial_dimensions"]
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        
        if self.softmax:
            prediction = torch.softmax(prediction, dim=1)
        
        if self.convert_to_one_vs_rest:
            prediction = convert_to_one_vs_rest(prediction.clone())

        if self.ignore_background:
            prediction = prediction[:, 1:]
            target = target[:, 1:]

        # Flatten out channel dimension to treat each channel as a separate instance
        prediction = torch.flatten(prediction, start_dim=0, end_dim=1).unsqueeze(1)
        converted_target = torch.flatten(target, start_dim=0, end_dim=1).unsqueeze(1)

        # Compute Wasserstein loss
        wasserstein_loss, losses = self.WassersteinLoss(prediction, converted_target)

        return wasserstein_loss, losses


class WassersteinLoss(torch.nn.modules.loss._Loss):
    def __init__(
        self,
        filtration_type: FiltrationType=FiltrationType.SUPERLEVEL,
        num_processes=1,
    ) -> None:
        super().__init__()
        self.filtration_type = filtration_type
        self.num_processes = num_processes

    def forward(self,
                input: Float[Tensor, "batch channels *spatial_dimensions"],
                target: Float[Tensor, "batch channels *spatial_dimensions"]
                ) -> tuple[Tensor, dict[str, List[Tensor]]]:
        
        wasserstein_losses = self.compute_wasserstein_loss(input, target)

        dic = {
            'losses': wasserstein_losses,
        }
        loss: torch.Tensor = torch.mean(torch.concatenate(wasserstein_losses))
        return loss, dic
    
    def compute_wasserstein_loss(self, 
                                prediction: Float[Tensor, "batch channels *spatial_dimensions"],
                                target: Float[Tensor, "batch channels *spatial_dimensions"],
                                ) -> List[torch.Tensor]:
        if self.filtration_type == FiltrationType.SUPERLEVEL:
            # Using (1 - ...) to allow binary sorting optimization on the label, which expects values [0, 1]
            prediction = 1 - prediction
            target = 1 - target
        if self.filtration_type == FiltrationType.BOTHLEVELS:
            # Just duplicate the number of elements in the batch, once with sublevel, once with superlevel
            prediction = torch.concat([prediction, 1 - prediction])
            target = torch.concat([target, 1 - target])

        split_indices = np.arange(self.num_processes, prediction.shape[0], self.num_processes)
        predictions_list_numpy = np.split(prediction.detach().cpu().numpy().astype(np.float64), split_indices)
        targets_list_numpy = np.split(target.detach().cpu().numpy().astype(np.float64), split_indices)
        
        losses = []

        current_instance_index = 0
        for predictions_cpu_batch, targets_cpu_batch in zip(predictions_list_numpy, targets_list_numpy):
            predictions_cpu_batch, targets_cpu_batch = list(predictions_cpu_batch.squeeze(1)), list(targets_cpu_batch.squeeze(1))
            if not (all(a.data.contiguous for a in predictions_cpu_batch) and all(a.data.contiguous for a in targets_cpu_batch)):
                print("WARNING! Non-contiguous arrays encountered. Shape:", predictions_cpu_batch[0].shape)
                global ENCOUNTERED_NONCONTIGUOUS
                ENCOUNTERED_NONCONTIGUOUS=True
            predictions_cpu_batch = [np.ascontiguousarray(a) for a in predictions_cpu_batch]
            targets_cpu_batch = [np.ascontiguousarray(a) for a in targets_cpu_batch]

            barcodes_batch = betti_matching.compute_barcode(predictions_cpu_batch + targets_cpu_batch)
            barcodes_predictions, barcodes_targets = barcodes_batch[:len(barcodes_batch)//2], barcodes_batch[len(barcodes_batch)//2:]

            for barcode_prediction, barcode_target in zip(barcodes_predictions, barcodes_targets):
                losses.append(self._wasserstein_loss(prediction[current_instance_index].squeeze(0), target[current_instance_index].squeeze(0), barcode_prediction, barcode_target))
                current_instance_index += 1

        return losses

    def _wasserstein_loss(self, prediction: Float[Tensor, "*spatial_dimensions"],
                        target: Float[Tensor, "*spatial_dimensions"],
                        barcode_result_prediction: betti_matching.return_types.BarcodeResult,
                        barcode_result_target: betti_matching.return_types.BarcodeResult,
                        ) -> Float[Tensor, "one_dimension"]:

        (prediction_birth_coordinates, prediction_death_coordinates, target_birth_coordinates, target_death_coordinates) = (
            [torch.tensor(array, device=prediction.device, dtype=torch.long) if array.strides[-1] > 0 else torch.zeros(0, len(prediction.shape), device=prediction.device, dtype=torch.long)
            for array in [barcode_result_prediction.birth_coordinates, barcode_result_prediction.death_coordinates,
                            barcode_result_target.birth_coordinates, barcode_result_target.death_coordinates]])
        
        # (M, 2) tensor of persistence pairs for prediction
        prediction_pairs = torch.stack([
            prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [prediction_birth_coordinates, prediction_death_coordinates]
        ], dim=1)
        # (M, 2) tensor of persistence pairs for target
        target_pairs = torch.stack([
            target[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [target_birth_coordinates, target_death_coordinates]
        ], dim=1)

        prediction_pairs = prediction_pairs.as_tensor() if isinstance(prediction_pairs, monai.data.meta_tensor.MetaTensor) else prediction_pairs
        target_pairs = target_pairs.as_tensor() if isinstance(target_pairs, monai.data.meta_tensor.MetaTensor) else target_pairs

        losses_matched_by_dim = []
        losses_unmatched_by_dim = []

        for prediction_pairs_dim, target_pairs_dim in zip(
            torch.split(prediction_pairs, barcode_result_prediction.num_pairs_by_dim.tolist()),
            torch.split(target_pairs, barcode_result_target.num_pairs_by_dim.tolist())
        ):
            _, matching = wasserstein.wasserstein_distance(prediction_pairs_dim.detach().cpu(), target_pairs_dim.detach().cpu(),
                                                                matching=True, keep_essential_parts=False) # type: ignore
            matching = torch.tensor(matching.reshape(-1, 2), device=prediction.device, dtype=torch.long)

            matched_pairs = matching[(matching[:,0] >= 0) & (matching[:,1] >= 0)]
            loss_matched = ((prediction_pairs_dim[matched_pairs[:,0]] - target_pairs_dim[matched_pairs[:,1]])**2).sum() # type: ignore
            prediction_pairs_unmatched = prediction_pairs_dim[matching[matching[:,1] == -1][:,0]]
            target_pairs_unmatched = target_pairs_dim[matching[matching[:,0] == -1][:,1]]
            loss_unmatched = 0.5*(((prediction_pairs_unmatched[:,0] - prediction_pairs_unmatched[:,1])**2).sum()
                                + ((target_pairs_unmatched[:,0] - target_pairs_unmatched[:,1])**2).sum()) # type: ignore

            losses_matched_by_dim.append(loss_matched)
            losses_unmatched_by_dim.append(loss_unmatched)

        return (sum(losses_matched_by_dim) + sum(losses_unmatched_by_dim)).reshape(1)
