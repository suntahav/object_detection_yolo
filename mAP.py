import torch
from collections import Counter

from iou import intersection_over_union


def mean_average_precision(
        pred_boxes, true_boxes, iou_threshold=0.5, box_format="midpoint", num_classes=20
):
    """
    Calculates mean average precision 

    Parameters:
        pred_boxes (list): list of lists containing all bboxes with each bboxes
        specified as [train_idx, class_prediction, prob_score, x1, y1, x2, y2]
        true_boxes (list): Similar as pred_boxes except all the correct ones 
        iou_threshold (float): threshold where predicted bboxes is correct
        box_format (str): "midpoint" or "corners" used to specify bboxes
        num_classes (int): number of classes

    Returns:
        float: mAP value across all classes given a specific IoU threshold 
    """

    average_precision = []
    epsilon = 1e-6

    for c in range(num_classes):
        detections = []
        ground_truth = []

        for detection in pred_boxes:
            if detection[1] == c:
                detections.append(detection)

        for truth in true_boxes:
            if truth[1] == c:
                ground_truth.append(truth)

        amount_bboxes = Counter([gt[0] for gt in ground_truth])

        for key, value in amount_bboxes.items():
            amount_bboxes[key] = torch.zeros(value)

        detections.sort(key=lambda x: x[2], reverse=True)
        TP = torch.zeros(len(detections))
        FP = torch.zeros(len(detections))

        total_true_bboxes = len(ground_truth)

        # Now if no ground truth means we can skip the class
        if total_true_bboxes == 0:
            continue

        for detection_idx, detection in enumerate(detections):
            # taking only ground truth that have same train idx as current one
            ground_truth_img = [
                gt for gt in ground_truth if gt[0]==detection[0]
            ]

            num_gts = len(ground_truth_img)
            best_iou = 0

            for idx, gt in enumerate(ground_truth_img):
                iou = intersection_over_union(torch.tensor(detection[3:]),
                                              torch.tensor(gt[3:]),
                                              box_format=box_format)

                if best_iou < iou:
                    best_iou = iou
                    best_gt_idx = idx

            if best_iou > iou_threshold:
                if amount_bboxes[detection[0]][best_gt_idx] == 0:
                    # If this index in the torch zero tensor is 0 set to 1 and mark as true positive
                    TP[detection_idx] = 1
                    amount_bboxes[detection[0]][best_gt_idx] = 1
                else:
                    FP[detection_idx] = 1

            else:
                # FP if iou less
                FP[detection_idx] = 1
        TP_cumsum = torch.cumsum(TP, dim=0)
        FP_cumsum = torch.cumsum(FP, dim=0)

        recalls = TP_cumsum / (total_true_bboxes + epsilon)
        precisions = TP_cumsum/(TP_cumsum + FP_cumsum + epsilon)
        # The graph has y value start at 1 for trapz and 0 for x axis
        recalls = torch.cat((torch.tensor([0]), recalls))
        precisions = torch.cat((torch.tensor([1]), precisions))

        average_precision.append(torch.trapz(precisions, recalls))
    # Return mAP for a particular iou threshold
    return sum(average_precision) / len(average_precision)
