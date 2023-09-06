import parser

from segment_anything import sam_model_registry

from lora_segment_anything import LoRA_Sam

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss


## TODO: give credit. Move to utils.
class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        assert (
            inputs.size() == target.size()
        ), "predict {} & target {} shape do not match".format(
            inputs.size(), target.size()
        )
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


def calc_loss(
    outputs, low_res_label_batch, ce_loss, dice_loss, dice_weight: float = 0.8
):
    low_res_logits = outputs["low_res_logits"]
    loss_ce = ce_loss(low_res_logits, low_res_label_batch[:].long())
    loss_dice = dice_loss(low_res_logits, low_res_label_batch, softmax=True)
    loss = (1 - dice_weight) * loss_ce + dice_weight * loss_dice
    return loss, loss_ce, loss_dice


def main() -> None:
    sam = sam_model_registry["vit_b"](checkpoint="./checkpoints/sam_vit_b_01ec64.pth")
    loraModel = LoRA_Sam(sam)

    trainloader = DataLoader(dataset, batch_size=1, shuffle=True)
    loraModel.train()

    ce_loss = CrossEntropyLoss()
    dice_loss = DiceLoss(3)
    # Focal loss?

    optimizer = optim.AdamW(filter(lambda p: p.requires_grad, loraModel.parameters()))
    # SGD?

    max_epochs = 1
    stop_epochs = 2

    for _, sampled_batch in enumerate(trainloader):
        image_batch = sampled_batch["image"]
        low_res_label_batch = sampled_batch["low_res_label"]

        outputs = loraModel(image_batch)
        low_res_logits = outputs["low_res_logits"]

        loss, loss_ce, loss_dice = calc_loss(
            outputs, low_res_label_batch, ce_loss, dice_loss, 0.8
        )

        optimizer.zero_grad()
        optimizer.step()


if __name__ == "main":
    args = parser.parse_args()
    main(args)
