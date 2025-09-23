import time
import torch
from utils.train_utils import MetricLogger
from utils.coco_utils import get_coco_api_from_dataset, CocoEvaluator
from config.test_config import test_cfg

from utils.Evaluator import Evaluator
from utils.BoundingBoxes import BoundingBoxes
from utils.BoundingBox import BoundingBox
from utils.ev_utils import *
from copy import deepcopy
import seaborn as sn
import numpy as np
import matplotlib.pyplot as plt
import warnings
from pathlib import Path

@torch.no_grad()
def evaluate(model, data_loader, device, mAP_list=None, extra_val=True, single_cls=False):
    n_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = MetricLogger(delimiter="  ")
    header = "Test: "
    conf = 0.0001


    # coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = ["bbox"]
    # coco_evaluator = CocoEvaluator(coco, iou_types)

    if extra_val:
        # Initialize external evaluation
        eval_boxes = BoundingBoxes()
        evaluator = Evaluator()
        confusion_matrix1 = ConfusionMatrix(4, conf=0.1, iou_thres=0.5)
        confusion_matrix2 = ConfusionMatrix(4, conf=0.2, iou_thres=0.5)
        confusion_matrix3 = ConfusionMatrix(4, conf=0.3, iou_thres=0.5)
        confusion_matrix4 = ConfusionMatrix(4, conf=0.4, iou_thres=0.5)
        confusion_matrix5 = ConfusionMatrix(4, conf=0.5, iou_thres=0.5)
        confusion_matrix6 = ConfusionMatrix(4, conf=0.6, iou_thres=0.5)
        confusion_matrix7 = ConfusionMatrix(4, conf=0.7, iou_thres=0.5)

    for image, targets_in in metric_logger.log_every(data_loader, 100, header):
        targets = deepcopy(targets_in)
        del targets_in

        image = list(img.to(device) for img in image)

        if device != torch.device("cpu"):
            torch.cuda.synchronize(device)

        model_time = time.time()
        outputs = model(image)

        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        
        if extra_val:
            labels = []
            for t in targets:
                for i, b in enumerate(t["boxes"]):
                    if t["labels"][i] < 5:
                        
                        bb = BoundingBox(t["image_id"], classId= 1 if single_cls else t["labels"][i], 
                                                x=b[0], y=b[1], w=b[2], h=b[3], typeCoordinates=CoordinatesType.Absolute,
                                                bbType=BBType.GroundTruth, format=BBFormat.XYX2Y2)
                        eval_boxes.addBoundingBox(bb)
                        labels.append([t["labels"][i]-1, b[0], b[1], b[2], b[3]])
            labels = torch.zeros((0,5)) if len(labels) == 0 else torch.Tensor(labels)
                        
            detections = []
            for j, o in enumerate(outputs):
                for i, b in enumerate(o["boxes"]):
                    if o["labels"][i] < 5 and o["scores"][i] > conf:
                        bb = BoundingBox(targets[j]["image_id"], classId=1 if single_cls else o["labels"][i], classConfidence=o["scores"][i], 
                                                x=b[0], y=b[1], w=b[2], h=b[3], typeCoordinates=CoordinatesType.Absolute,
                                                bbType=BBType.Detected, format=BBFormat.XYX2Y2)
                        eval_boxes.addBoundingBox(bb)
                    if o["labels"][i] < 5:
                        detections.append([b[0], b[1], b[2], b[3], o["scores"][i], o["labels"][i]-1])

                        # gts = eval_boxes.getBoundingBoxesByImageName(targets[j]["image_id"]).getBoundingBoxesByType(BBType.GroundTruth)
                        # ious = Evaluator._getAllIOUs(bb, gts)

                        # print(ious)
                        # if len(ious) > 0:
                        #     hit = ious[0]
                        #     print(hit)
                        #     bb_hit = hit[2]
                        #     iou = hit[0]
                        #     print(iou, bb_hit)
                        #     gt_id = bb_hit.getClassId()
                        #     print(gt_id)
                        #     if iou < 0.5:
                        #         gt_id = 5
                        #     confusion_matrix[o["labels"][i]-1, gt_id-1] += 1
                        #     print(confusion_matrix)
            detections = torch.zeros((0,6)) if len(detections) == 0 else torch.Tensor(detections)
            confusion_matrix1.process_batch(detections, labels)
            confusion_matrix2.process_batch(detections, labels)
            confusion_matrix3.process_batch(detections, labels)
            confusion_matrix4.process_batch(detections, labels)
            confusion_matrix5.process_batch(detections, labels)
            confusion_matrix6.process_batch(detections, labels)
            confusion_matrix7.process_batch(detections, labels)
        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}

        evaluator_time = time.time()
        # coco_evaluator.update(res)
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)

    if extra_val:
        for i in range(1,10):
            print()
            print("IoU: ", str(i/10))
            results = evaluator.PlotPrecisionRecallCurve(eval_boxes, # Object containing all bounding boxes (ground truths and detections)
                                                    IOUThreshold=i/10, # IOU threshold
                                                    showAP=True, # Show Average Precision in the title of the plot
                                                    showInterpolatedPrecision=False, # Don't plot the interpolated precision curve
                                                    savePath='runs/eval_PR.png',
                                                    showGraphic=False)
            # Print results per class
            for mc in results:
                c = mc['class']
                precision = mc['precision'][-1] if len(mc['precision']) > 0 else "no data"
                recall = mc['recall'][-1] if len(mc['recall']) > 0 else "no data"
                average_precision = mc['AP']
                POS = mc['total positives']
                TP = mc['total TP']
                FP = mc['total FP']
                print(c)
                print("precision: ", precision)
                print("recall: ", recall)
                print("average_precision: ", average_precision)
                print("POS: ", POS)
                print("TP: ", TP)
                print("FP: ", FP)
                print()

            total_pos = sum([r["total positives"] for r in results])
            print(total_pos)
            mAP = sum([r["AP"]*(r["total positives"]/total_pos) for r in results])
            print("mAP: ", mAP)

        confusion_matrix1.plot(save_dir="runs")
        confusion_matrix2.plot(save_dir="runs")
        confusion_matrix3.plot(save_dir="runs")
        confusion_matrix4.plot(save_dir="runs")
        confusion_matrix5.plot(save_dir="runs")
        confusion_matrix6.plot(save_dir="runs")
        confusion_matrix7.plot(save_dir="runs")

    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    # coco_evaluator.synchronize_between_processes()

    # accumulate predictions from all images
    # coco_evaluator.accumulate()
    # coco_evaluator.summarize()
    torch.set_num_threads(n_threads)

    # print_txt = coco_evaluator.coco_eval[iou_types[0]].stats
    # coco_mAP = print_txt[0]
    # voc_mAP = print_txt[1]
    # if isinstance(mAP_list, list):
    #     mAP_list.append(voc_mAP)
    mAP = sum([r["AP"] for r in results])/5
    return None, mAP

    return coco_evaluator, voc_mAP

class ConfusionMatrix:
    # Updated version of https://github.com/kaanakan/object_detection_confusion_matrix
    def __init__(self, nc, conf=0.25, iou_thres=0.5):
        self.matrix = np.zeros((nc + 1, nc + 1))
        self.nc = nc  # number of classes
        self.conf = conf
        self.iou_thres = iou_thres

    def process_batch(self, detections, labels):
        """
        Return intersection-over-union (Jaccard index) of boxes.
        Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
        Arguments:
            detections (Array[N, 6]), x1, y1, x2, y2, conf, class
            labels (Array[M, 5]), class, x1, y1, x2, y2
        Returns:
            None, updates confusion matrix accordingly
        """
        detections = detections[detections[:, 4] > self.conf]
        gt_classes = labels[:, 0].int()
        detection_classes = detections[:, 5].int()
        iou = box_iou(labels[:, 1:], detections[:, :4])

        x = torch.where(iou > self.iou_thres)
        if x[0].shape[0]:
            matches = torch.cat((torch.stack(x, 1), iou[x[0], x[1]][:, None]), 1).cpu().numpy()
            if x[0].shape[0] > 1:
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 1], return_index=True)[1]]
                matches = matches[matches[:, 2].argsort()[::-1]]
                matches = matches[np.unique(matches[:, 0], return_index=True)[1]]
        else:
            matches = np.zeros((0, 3))

        n = matches.shape[0] > 0
        m0, m1, _ = matches.transpose().astype(np.int16)
        for i, gc in enumerate(gt_classes):
            j = m0 == i
            if n and sum(j) == 1:
                self.matrix[detection_classes[m1[j]], gc] += 1  # correct
            else:
                self.matrix[self.nc, gc] += 1  # background FP

        if n:
            for i, dc in enumerate(detection_classes):
                if not any(m1 == i):
                    self.matrix[dc, self.nc] += 1  # background FN

    def matrix(self):
        return self.matrix

    def tp_fp(self):
        tp = self.matrix.diagonal()  # true positives
        fp = self.matrix.sum(1) - tp  # false positives
        # fn = self.matrix.sum(0) - tp  # false negatives (missed detections)
        return tp[:-1], fp[:-1]  # remove background class

    def plot(self, normalize=True, save_dir='', names=()):
        try:
            
            for i in range(2):
                #array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if normalize else 1)  # normalize columns
                array = self.matrix / ((self.matrix.sum(0).reshape(1, -1) + 1E-6) if i==0 else 1)  # normalize columns
                array[array < 0.005] = np.nan  # don't annotate (would appear as 0.00)

                fig = plt.figure(figsize=(12, 9), tight_layout=True)
                sn.set(font_scale=2.0 if self.nc < 50 else 0.8)  # for label size
                labels = (0 < len(names) < 99) and len(names) == self.nc-1  # apply names to ticklabels
                with warnings.catch_warnings():
                    warnings.simplefilter('ignore')  # suppress empty matrix RuntimeWarning: All-NaN slice encountered
                    sn.heatmap(array, annot=self.nc < 30, annot_kws={"size": 24}, cmap='Blues', fmt='.2f', square=True,
                            xticklabels=names + ['background FP'] if labels else "auto",
                            yticklabels=names + ['background FN'] if labels else "auto").set_facecolor((1, 1, 1))
                fig.axes[0].set_xlabel('True')
                fig.axes[0].set_ylabel('Predicted')
                fig.savefig(Path(save_dir) / (str(self.conf) + 'confusion_matrix_normalized.png') if i==0 else Path(save_dir) / (str(self.conf) + 'confusion_matrix.png'), dpi=250)
                plt.close()
        except Exception as e:
            print(f'WARNING: ConfusionMatrix plot failure: {e}')

    def print(self):
        for i in range(self.nc + 1):
            print(' '.join(map(str, self.matrix[i])))


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)