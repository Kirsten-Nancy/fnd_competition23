from sklearn import metrics
import torch

def compute_measures(logit, y_gt):
    predicts = logit

    accuracy = metrics.accuracy_score(y_true=y_gt, y_pred=predicts)

    # macro
    macro_precision = metrics.precision_score(y_true=y_gt, y_pred=predicts, average='macro', zero_division=0)
    macro_recall = metrics.recall_score(y_true=y_gt, y_pred=predicts, average='macro', zero_division=0)
    macro_f1 = metrics.f1_score(y_true=y_gt, y_pred=predicts, average='macro', zero_division=0)


    measures = {"accuracy":accuracy,
                "macro_precision": macro_precision, "macro_recall": macro_recall, "macro_f1": macro_f1
                }
    return measures


def print_measures(epoch, loss, metrics, category):
    if category == 'train':
        print("-Epoch: {} Train Loss: {:.4f}  Accuracy: {:4f} \n" \
                " Macro:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n" 
            .format(epoch, loss, metrics['accuracy'],
                    metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'])) 
    else:
        print("-Test Loss: {:.4f}  Accuracy: {:4f} \n" \
                " Macro:  Precision: {:4f}  Recall: {:4f}  F1: {:4f}  \n" 
            .format(loss, metrics['accuracy'],
                    metrics['macro_precision'], metrics['macro_recall'], metrics['macro_f1'])) 