import torch
import torch.nn as nn
from iou import intersection_over_union


class YoloLoss(nn.Module):
    def __init__(self, S=7, B=2, C=20):
        super(YoloLoss, self).__init__()
        self.mse = nn.MSELoss(reduction="sum")
        self.S = S
        self.B = B
        self.C = C
        self.lambda_noobj = 0.5
        self.lambda_coord = 5

    def forward(self, predictions, target):
        # predictions: (s, s, b*5+c)
        predictions = predictions.reshape(-1, self.S, self.S, self.C + self.B * 5)

        iou_box1 = intersection_over_union(predictions[..., 21:25], target[..., 21:25])
        iou_box2 = intersection_over_union(predictions[..., 26:30], target[..., 21:25])
        ious = torch.cat([iou_box1.unsqueeze(0), iou_box2.unsqueeze(0)], dim=0)
        imou_max, best_box = torch.max(ious, dim=0)
        box_exist = target[..., 20].unsqueeze(3)  # (s, s, b, 1) this variable is like the identity function needed
        # for the loss function

        ''' Box Coordinates Loss '''

        box_predictions = box_exist * (best_box * predictions[..., 26:30] + (1 - best_box) * predictions[..., 21:25])
        box_target = box_exist * target[..., 21:25]
        # Now as per paper the height and width are square rooted
        box_predictions[..., 2:4] = torch.sign(box_predictions[..., 2:4]) * torch.sqrt(
            torch.abs(box_predictions[..., 2:4] + 1e-6)
        )
        box_target[..., 2:4] = torch.sqrt(box_target[..., 2:4])
        # Now we calculate the loss for box coordinates
        box_loss = self.mse(
            torch.flatten(box_predictions, end_dim=-2),
            torch.flatten(box_target, end_dim=-2)
        )

        ''' Object Loss '''
        # Now we calculate the loss for object loss if it exists
        pred_box = (best_box * predictions[..., 25:26] + (1 - best_box) * predictions[..., 20:21])
        object_loss = self.mse(
            torch.flatten(box_exist * pred_box),
            torch.flatten(box_exist * target[..., 20:21])
        )

        ''' No Object Loss '''
        # Now we calculate the loss for no object loss if it does not exist for first box
        no_object_loss = self.mse(
            torch.flatten((1 - box_exist) * predictions[..., 20:21], start_dim=1),
            torch.flatten((1 - box_exist) * target[..., 20:21], start_dim=1)
        )
        # Now we calculate the loss for no object loss if it does not exist for second box
        no_object_loss += self.mse(
            torch.flatten((1 - box_exist) * predictions[..., 25:26], start_dim=1),
            torch.flatten((1 - box_exist) * target[..., 20:21], start_dim=1)
        )

        ''' Class Loss '''
        # Now we calculate the loss for class probability loss (in mse rather than cross entropy as mentioned in the
        # paper)
        class_loss = self.mse(
            torch.flatten(box_exist * predictions[..., :20], end_dim=-2),
            torch.flatten(box_exist * target[..., :20], end_dim=-2)
        )

        ''' Final Loss '''
        total_loss = (
            self.lambda_coord * box_loss + # First two losses in the paper
            object_loss +
            self.lambda_noobj * no_object_loss +
            class_loss
        )
        return total_loss