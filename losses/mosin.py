from __future__ import annotations

import torch
import torch.nn.functional as F
import torchvision
from torch.nn.modules.loss import _Loss

import typing
from losses.dice_losses import Multiclass_CLDice
from losses.utils import DiceType, convert_to_one_vs_rest
if typing.TYPE_CHECKING:
    from typing import Tuple, List
    from numpy.typing import NDArray
    LossOutputName = str
    from jaxtyping import Float
    from torch import Tensor

    
class MulticlassDiceMOSIN(_Loss):
    def __init__(self,
                 dice_type: DiceType=DiceType.CLDICE,
                 convert_to_one_vs_rest: bool = False,
                 cldice_alpha: float = 0.5,
                 ignore_background: bool = False,) -> None:
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
        
        self.MulticlassMOSINLoss = MulticlassMOSIN(
            convert_to_one_vs_rest=convert_to_one_vs_rest,
            softmax=not convert_to_one_vs_rest,
            ignore_background=ignore_background,
        )

    def forward(self, 
                prediction: Float[torch.Tensor, "batch channel *spatial_dimensions"], 
                target: Float[torch.Tensor, "batch channel *spatial_dimensions"],
                alpha: float = 0.5
                ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        # Compute multiclass MOSIN losses
        if alpha > 0:
            mosin_loss, losses = self.MulticlassMOSINLoss(prediction, target)
            losses = {"single_matches": losses}
        else:
            mosin_loss = torch.zeros(1, device=prediction.device)
            losses = {}

        # Multiclass Dice loss
        dice_loss, dic = self.DiceLoss(prediction, target)
        
        losses["dice"] = dic["dice"]
        losses["cldice"] = dic["cldice"]
        losses["mosin"] = alpha * mosin_loss.item()

        return dice_loss + alpha * mosin_loss, losses

class MulticlassMOSIN(_Loss):
    def __init__(self,
                 convert_to_one_vs_rest: bool = True,
                 softmax: bool = False,
                 ignore_background: bool = False,) -> None:
        super().__init__()
        if not softmax and not convert_to_one_vs_rest:
            raise ValueError("If softmax is False, convert_to_one_vs_rest must be True")
        if softmax and convert_to_one_vs_rest:
            raise ValueError("If softmax is True, convert_to_one_vs_rest must be False. One vs rest is already handled by softmax.")
        
        self.softmax = softmax
        self.convert_to_one_vs_rest = convert_to_one_vs_rest
        self.ignore_background = ignore_background

        self.MOSINLoss = MOSIN()

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

        # Compute MOSIN loss
        mosin_loss, losses = self.MOSINLoss(prediction, converted_target)

        return mosin_loss, losses

class MOSIN(_Loss):
    def __init__(self):
        super(MOSIN, self).__init__()
        self.vgg = torchvision.models.vgg19(torchvision.models.VGG19_Weights.IMAGENET1K_V1).features.to('cuda')
        self.activation = {}

        self.vgg[2].register_forward_hook(self.get_activation('conv1_2'))
        self.vgg[7].register_forward_hook(self.get_activation('conv2_2'))
        self.vgg[16].register_forward_hook(self.get_activation('conv3_4'))

        self.vgg.eval()
    
    def get_activation(self, name):
        def hook(model, input, output):
            self.activation[name] = output
        return hook
    
    def forward(self, prediction, target, alpha=0.0):
        #ce_loss = F.cross_entropy(prediction, target)

        prediction = prediction.expand(-1, 3, -1, -1)
        target = target.argmax(dim=1, keepdim=True).to(torch.float32).expand(-1, 3, -1, -1)
        self.activation = {}
        self.vgg(prediction)
        prediction_conv1_2 = self.activation['conv1_2']
        prediction_conv2_2 = self.activation['conv2_2']
        prediction_conv3_4 = self.activation['conv3_4']

        self.activation = {}
        self.vgg(target)
        target_conv1_2 = self.activation['conv1_2']
        target_conv2_2 = self.activation['conv2_2']
        target_conv3_4 = self.activation['conv3_4']

        l_topo = F.mse_loss(prediction_conv1_2, target_conv1_2) + F.mse_loss(prediction_conv2_2, target_conv2_2) + F.mse_loss(prediction_conv3_4, target_conv3_4)

        return l_topo, {}

def mosin_forward(model, image, target, loss, K):
    mosin_loss = torch.tensor(0.0, device='cuda')

    y = torch.zeros_like(image)
    dic = {}

    for k in torch.arange(K, device='cuda'):
        if k > 0:
            y = F.softmax(y, dim=1)[:, 1, :, :].unsqueeze(dim=1)
        concat_image = torch.cat([image, y], dim=1)
        y = model(concat_image)
        if not target is None:
            loss_k, dic = loss(y, target)
            mosin_loss += (k+1) * loss_k
    
    mosin_loss /= (0.5 * K * (K + 1))

    dic['output'] = y

    return mosin_loss, dic
