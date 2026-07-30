import torch
import torchmetrics
import numpy as np

def get_cls_metrics_binary_pt(y_true, y_pred):
    y_true = y_true.to(torch.long)

    # 1. Confusion matrix (for TP, TN, FP, FN)
    # binary_confusion_matrix returns shape [2, 2]: [[TN, FP], [FN, TP]]
    conf_matrix = torchmetrics.functional.classification.binary_confusion_matrix(
        y_pred, y_true, threshold=0.5  # threshold 0.5: convert probabilities to 0/1
    )
    tn, fp, fn, tp = conf_matrix.flatten().tolist()  # flatten matrix to list
    
    mcc = torchmetrics.functional.classification.binary_matthews_corrcoef(y_pred, y_true, threshold=0.5)
    auroc = torchmetrics.functional.classification.binary_auroc(y_pred, y_true, thresholds=None)
    prauc = torchmetrics.functional.classification.binary_average_precision(y_pred, y_true)  # PRAUC（Precision-Recall AUC）
    accuracy = torchmetrics.functional.classification.binary_accuracy(y_pred, y_true, threshold=0.5)
    f1 = torchmetrics.functional.classification.binary_f1_score(y_pred, y_true, threshold=0.5)

    return auroc, prauc, mcc, accuracy, f1, tp, tn, fp, fn


def get_cls_metrics_multilabel_pt(y_true, y_pred, num_cls):
    y_true = y_true.to(torch.long)

    mcc = torchmetrics.functional.classification.multilabel_matthews_corrcoef(
        y_pred, y_true, num_labels=num_cls, threshold=0.5
    )
    auroc = torchmetrics.functional.classification.multilabel_auroc(
        y_pred, y_true, num_labels=num_cls, average="macro", thresholds=None
    )
    accuracy = torchmetrics.functional.classification.multilabel_accuracy(
        y_pred,
        y_true,
        num_labels=num_cls,
        average="macro",
        threshold=0.5,
    )
    f1 = torchmetrics.functional.classification.multilabel_f1_score(
        y_pred, y_true, num_labels=num_cls, average="macro", threshold=0.5
    )

    return auroc, mcc, accuracy, f1


def get_cls_metrics_multiclass_pt(y_true, y_pred, num_cls):
    y_true = y_true.to(torch.long)

    if 0 not in set(y_true.tolist()):
        y_true = y_true - 1

    mcc = torchmetrics.functional.classification.multiclass_matthews_corrcoef(y_pred, y_true, num_classes=num_cls)
    auroc = torchmetrics.functional.classification.multiclass_auroc(
        y_pred, y_true, num_classes=num_cls, average="macro", thresholds=None
    )
    accuracy = torchmetrics.functional.classification.multiclass_accuracy(
        y_pred,
        y_true,
        num_classes=num_cls,
        average="macro",
    )
    f1 = torchmetrics.functional.classification.multiclass_f1_score(
        y_pred, y_true, num_classes=num_cls, average="macro"
    )
    ap = torchmetrics.functional.classification.multiclass_average_precision(
        y_pred,
        y_true,
        num_classes=num_cls,
        average="macro",
        thresholds=None
    )

    return auroc, mcc, accuracy, f1, ap
