from __future__ import annotations

import enum
import torch
from torch.nn.modules.loss import _Loss
from functools import partial
import numpy as np

from monai.losses.dice import DiceLoss

import betti_matching           # C++ Implementation

import typing
from losses.dice_losses import Multiclass_CLDice
from losses.utils import FiltrationType, ActivationType, DiceType

from losses.utils import convert_to_one_vs_rest
if typing.TYPE_CHECKING:
    from jaxtyping import Float

class FastMulticlassBettiMatchingLoss(_Loss):
    def __init__(self,
                 filtration_type: FiltrationType=FiltrationType.SUPERLEVEL, 
                 num_processes: int=1,
                 convert_to_one_vs_rest: bool = True,
                 softmax: bool = False,
                 ignore_background: bool = False,
                 push_unmatched_to_1_0: bool = False,
                 barcode_length_threshold: float = 0.0,
                 topology_weights: tuple[float, float] = (1,1),
                 sphere: bool = False
                 ) -> None:
        super().__init__()
        if not softmax and not convert_to_one_vs_rest:
            raise ValueError("If softmax is False, convert_to_one_vs_rest must be True")
        if softmax and convert_to_one_vs_rest:
            raise ValueError("If softmax is True, convert_to_one_vs_rest must be False. One vs rest is already handled by softmax.")
        
        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.ignore_background = ignore_background

        self.BMLoss = FastBettiMatchingLoss(
            activation=ActivationType.NONE, 
            filtration_type=filtration_type, 
            num_processes=num_processes,
            push_unmatched_to_1_0=push_unmatched_to_1_0,
            barcode_length_threshold=barcode_length_threshold,
            topology_weights=topology_weights,
            sphere=sphere
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

        # Compute Betti matching loss
        bm_loss, losses = self.BMLoss(prediction, converted_target)

        return bm_loss, losses
    
class FastMulticlassDiceBettiMatchingLoss(_Loss):
    def __init__(self,
                 filtration_type: FiltrationType=FiltrationType.SUPERLEVEL, 
                 dice_type: DiceType=DiceType.CLDICE,
                 num_processes: int=1,
                 convert_to_one_vs_rest: bool = False,
                 cldice_alpha: float = 0.5,
                 ignore_background: bool = False,
                 push_unmatched_to_1_0: bool = False,
                 barcode_length_threshold: float = 0.0,
                 topology_weights: tuple[float, float] = (1,1),
                 sphere: bool = False
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
                include_background=True, 
                smooth=1e-5, 
                alpha=cldice_alpha, 
                iter_=5, 
                convert_to_one_vs_rest=convert_to_one_vs_rest,
                batch=True
            )
        else:
            raise ValueError(f"Invalid dice type: {dice_type}")
        
        self.MulticlassBMLoss = FastMulticlassBettiMatchingLoss(
            filtration_type=filtration_type, 
            num_processes=num_processes,
            convert_to_one_vs_rest=convert_to_one_vs_rest,
            softmax=not convert_to_one_vs_rest,
            push_unmatched_to_1_0=push_unmatched_to_1_0,
            ignore_background=ignore_background,
            barcode_length_threshold=barcode_length_threshold,
            topology_weights=topology_weights,
            sphere=sphere
        )

    def forward(self, 
                prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"], 
                target: Float[torch.Tensor, "batch channel *spatial_dimensions"],
                alpha: float = 0.5
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Compute multiclass BM losses
        if alpha > 0:
            bm_loss, losses = self.MulticlassBMLoss(prediction, target)
            losses = {"single_matches": losses}
        else:
            bm_loss = torch.zeros(1, device=prediction.device)
            losses = {}

        # Multiclass Dice loss
        dice_loss, dic = self.DiceLoss(prediction, target)
        
        losses["dice"] = dic["dice"]
        losses["cldice"] = dic["cldice"]
        losses["bm"] = alpha * bm_loss.item()

        return dice_loss + alpha * bm_loss, losses

class FastDiceBettiMatchingLoss(_Loss):
    def __init__(self,
                 alpha: float = 0.5,
                 activation: ActivationType=ActivationType.SIGMOID,
                 filtration_type: FiltrationType=FiltrationType.SUPERLEVEL, 
                 num_processes: int=1,
                 push_unmatched_to_1_0: bool = False,
                 barcode_length_threshold: float = 0.0,
                 sphere: bool = False
                 ) -> None:
        super().__init__()
        self.alpha = alpha

        if activation == ActivationType.SIGMOID:
            self.activation = torch.sigmoid
        elif activation == ActivationType.SOFTMAX:
            self.activation = partial(torch.softmax, dim=1)
        else:
            self.activation = None
        
        self.DiceLoss = DiceLoss(sigmoid=False)
        self.BMLoss = FastBettiMatchingLoss(
            activation=ActivationType.NONE, 
            filtration_type=filtration_type, 
            num_processes=num_processes,
            push_unmatched_to_1_0=push_unmatched_to_1_0,
            barcode_length_threshold=barcode_length_threshold,
            sphere=sphere
            )

    def forward(self, 
                prediction: Float[torch.Tensor, "*spatial_dimensions"], 
                target: Float[torch.Tensor, "*spatial_dimensions"]
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.activation is not None:
            prediction = self.activation(prediction)

        bm_loss, losses = self.BMLoss(prediction, target)
        dice_loss = self.DiceLoss(prediction, target)

        return dice_loss + self.alpha * bm_loss, losses
        
    
class FastBettiMatchingLoss(_Loss):
    def __init__(self,
                 activation: ActivationType=ActivationType.SIGMOID,
                 filtration_type: FiltrationType=FiltrationType.SUPERLEVEL, 
                 num_processes: int=1,
                 push_unmatched_to_1_0: bool = False,
                 barcode_length_threshold: float = 0.0,
                 topology_weights: tuple[float, float] = (1., 1.), # weights for the topology classes in the following order: [matched, unmatched_pred, unmatched_target]
                 sphere: bool = False
                ) -> None:
        super().__init__()
        self.filtration_type = filtration_type
        self.num_processes = num_processes
        self.push_unmatched_to_1_0 = push_unmatched_to_1_0
        self.barcode_length_threshold = barcode_length_threshold

        if len(topology_weights) != 2:
            raise ValueError("Topology weights must be a list of length 2, where the first element is the weight for matched pairs and the second for unmatched pairs in the prediction.")

        self.topology_weights = topology_weights
        self.sphere = sphere

        if activation == ActivationType.SIGMOID:
            self.activation = torch.sigmoid
        elif activation == ActivationType.SOFTMAX:
            self.activation = partial(torch.softmax, dim=1)
        else:
            self.activation = None

    def forward(self, 
                prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"], 
                target: Float[torch.Tensor, "batch channel *spatial_dimensions"]
                ) -> tuple[torch.Tensor, list[torch.Tensor]]:
        if self.activation is not None:
            prediction = self.activation(prediction)

        if self.sphere:
            # add padding of 1 to all sides of the spatial dimensions with correct predicted background
            prediction = torch.nn.functional.pad(prediction, (1, 1, 1, 1), mode='constant', value=0)
            target = torch.nn.functional.pad(target, (1, 1, 1, 1), mode='constant', value=0)
        
        return self._compute_batched_loss(prediction, target)
    
    def _compute_batched_loss(self, 
                              prediction: Float[torch.Tensor, "batch channels *spatial_dimensions"], 
                              target: Float[torch.Tensor, "batch channels *spatial_dimensions"]
                              ) -> tuple[torch.Tensor, list[torch.Tensor]]:
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

        num_dimensions = prediction.ndim - 2
        num_matched_by_dim = torch.zeros((num_dimensions,), device=prediction.device)
        num_unmatched_prediction_by_dim = torch.zeros((num_dimensions,), device=prediction.device)

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

            results = betti_matching.compute_matching(predictions_cpu_batch, targets_cpu_batch)

            for result_arrays in results:
                losses.append(self._betti_matching_loss(prediction[current_instance_index].squeeze(0), target[current_instance_index].squeeze(0), result_arrays))

                num_matched_by_dim += torch.tensor(result_arrays.num_matches_by_dim, device=prediction.device, dtype=torch.long)
                num_unmatched_prediction_by_dim += torch.tensor(result_arrays.num_unmatched_prediction_by_dim, device=prediction.device, dtype=torch.long)

                current_instance_index += 1

        return torch.mean(torch.concatenate(losses)), losses

    def _betti_matching_loss(self,
                             prediction: Float[torch.Tensor, "*spatial_dimensions"],
                             target: Float[torch.Tensor, "*spatial_dimensions"],
                             betti_matching_result: betti_matching.return_types.BettiMatchingResult,
                             ) -> Float[torch.Tensor, "one_dimension"]:
        
        # Combine all birth and death coordinates from prediction and target into one array
        (prediction_matches_birth_coordinates, prediction_matches_death_coordinates, target_matches_birth_coordinates,
        target_matches_death_coordinates, prediction_unmatched_birth_coordinates, prediction_unmatched_death_coordinates) = (
            [torch.tensor(array, device=prediction.device, dtype=torch.long) if array.strides[-1] > 0 else torch.zeros(0, len(prediction.shape), device=prediction.device, dtype=torch.long)
            for array in [betti_matching_result.prediction_matches_birth_coordinates, betti_matching_result.prediction_matches_death_coordinates,
                            betti_matching_result.target_matches_birth_coordinates, betti_matching_result.target_matches_death_coordinates,
                            betti_matching_result.prediction_unmatched_birth_coordinates, betti_matching_result.prediction_unmatched_death_coordinates]])

        # Get the Barcode interval of the matched pairs from the prediction using the coordinates
        # (M, 2) tensor of matched persistence pairs for prediction
        prediction_matched_pairs = torch.stack([
            prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [prediction_matches_birth_coordinates, prediction_matches_death_coordinates]
        ], dim=1)

        # Get the Barcode interval of the matched pairs from the target using the coordinates
        # (M, 2) tensor of matched persistence pairs for target
        target_matched_pairs = torch.stack([
            target[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [target_matches_birth_coordinates, target_matches_death_coordinates]
        ], dim=1)

        # Get the Barcode interval of all unmatched pairs  in the prediction using the coordinates
        # (M, 2) tensor of unmachted persistence pairs for prediction
        prediction_unmatched_pairs = torch.stack([
            prediction[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [prediction_unmatched_birth_coordinates, prediction_unmatched_death_coordinates]
        ], dim=1)

        # Get the Barcode interval of all unmatched pairs in the target using the coordinates
        # (M, 2) tensor of unmatched persistence pairs for target
        target_unmatched_pairs = torch.stack([
            target[tuple(coords[:, i] for i in range(coords.shape[1]))]
            for coords in [betti_matching_result.target_unmatched_birth_coordinates, betti_matching_result.target_unmatched_death_coordinates]
        ], dim=1)

        # filter all pairs where abs(birth - death) < threshold
        prediction_unmatched_pairs = prediction_unmatched_pairs[torch.abs(prediction_unmatched_pairs[:, 0] - prediction_unmatched_pairs[:, 1]) > self.barcode_length_threshold]

        # sum over ||(birth_pred_i, death_pred_i), (birth_target_i, death_target_i)||²
        loss_matched = 2 * ((prediction_matched_pairs - target_matched_pairs) ** 2).sum() * self.topology_weights[0]

        # sum over ||(birth_pred_i, death_pred_i), 1/2*(birth_pred_i+death_pred_i, birth_pred_i+death_pred_i)||²
        # reformulated as (birth_pred_i^2 / 4 + death_pred_i^2/4 - birth_pred_i*death_pred_i/2)
        if self.push_unmatched_to_1_0:
            loss_unmatched_pred = 2 * ((prediction_unmatched_pairs[:, 0] - 1) ** 2 + prediction_unmatched_pairs[:, 1]**2).sum() * self.topology_weights[1]
            loss_unmatched_target = 2 * ((target_unmatched_pairs[:, 0] - 1) ** 2 + target_unmatched_pairs[:, 1]**2).sum()
        else:
            loss_unmatched_pred = ((prediction_unmatched_pairs[:, 0] - prediction_unmatched_pairs[:, 1])**2).sum() * self.topology_weights[1]
            loss_unmatched_target = ((target_unmatched_pairs[:, 0] - target_unmatched_pairs[:, 1])**2).sum()

        return (loss_matched + loss_unmatched_pred + loss_unmatched_target).reshape(1)