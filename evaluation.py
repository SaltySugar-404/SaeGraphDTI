import math
import numpy as np
import matplotlib.pyplot as plt
from sklearn import metrics
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc


def get_confusion_matrix_results(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    TP = float(np.sum((labels == 1) & (predicts == 1)))
    TN = float(np.sum((labels == 0) & (predicts == 0)))
    FN = float(np.sum((labels == 1) & (predicts == 0)))
    FP = float(np.sum((labels == 0) & (predicts == 1)))
    accuracy = (TP + TN) / (TP + TN + FP + FN) if TP + TN + FP + FN else 0
    precision = TP / (TP + FP) if TP + FP else 0
    recall = TP / (TP + FN) if TP + FN else 0
    f1_score = 2 * precision * recall / (precision + recall) if precision * recall else 0
    return accuracy, precision, recall, f1_score


def get_curve_results(labels, predicts):
    labels = np.array(labels)
    predicts = np.array(predicts)
    auc_score = roc_auc_score(labels, predicts)
    sort_indices = np.argsort(predicts)[::-1]
    labels = labels[sort_indices]
    predicts = predicts[sort_indices]
    precision_, recall_, _ = precision_recall_curve(labels, predicts)
    aupr_score = auc(recall_, precision_)
    return auc_score, aupr_score


def draw_ROC(labels: np.array, predicts: np.array):
    fpr, tpr, thresholds = metrics.roc_curve(labels, predicts)
    auc = metrics.auc(fpr, tpr)
    plt.figure(dpi=200)
    fontsize = 16
    plt.plot(fpr, tpr, color='darkorange', lw=2, label='AUC = %0.3f' % auc)
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.1])
    plt.ylim([0.0, 1.1])
    plt.xlabel('False Positive Rate', fontsize=fontsize)
    plt.ylabel('True Positive Rate', fontsize=fontsize)
    plt.title('ROC', fontsize=fontsize)
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.show()


def draw_PR(labels: np.array, predicts: np.array):
    sort_indices = np.argsort(predicts)[::-1]
    labels = labels[sort_indices]
    predicts = predicts[sort_indices]

    precision, recall, _ = metrics.precision_recall_curve(labels, predicts)
    auprc = metrics.auc(recall, precision)
    fontsize = 16
    plt.figure(dpi=200)
    plt.plot(recall, precision, color='darkorange', lw=2, label='AUC = %0.3f' % auprc)
    plt.xlim([0.0, 1.05])
    plt.ylim([0.0, 1.05])
    plt.xlabel('Recall', fontsize=fontsize)
    plt.ylabel('Precision', fontsize=fontsize)
    plt.title('PR', fontsize=fontsize)
    plt.legend(loc='lower right', fontsize=fontsize)
    plt.show()


def draw_confusion_matrix(labels: np.array, predicts: np.array):
    cm = metrics.confusion_matrix(labels, predicts)
    plt.figure(dpi=200)
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    fontsize = 16
    plt.title('Confusion Matrix', fontsize=fontsize)
    plt.colorbar()
    tick_marks = np.arange(len(np.unique(labels)))
    plt.xticks(tick_marks, np.unique(labels))
    plt.yticks(tick_marks, np.unique(labels))
    plt.xlabel('Predicted Label', fontsize=fontsize)
    plt.ylabel('True Label', fontsize=fontsize)

    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, format(cm[i, j], 'd'),
                     horizontalalignment='center',
                     color='white' if cm[i, j] > thresh else 'black')
    plt.show()


def draw_bar_chart(title: str, categories: list, values: list, colors: list):
    plt.bar(categories, values, colors=colors)
    plt.title(title)
    plt.xlabel(categories)
    plt.ylabel(values)
    plt.ylim(0, 1)
    plt.show()


def draw_comparison_bar_chart(fontsize: int, title: str, categories: list, values: list, labels: list, colors: list, picture_path: str):
    plt.figure(dpi=200, figsize=(8, 7))
    bar_width = 0.35
    x = np.arange(len(categories))
    rects1 = plt.bar(x - bar_width / 2, values[0], bar_width, label=labels[0], color=colors[0])
    rects2 = plt.bar(x + bar_width / 2, values[1], bar_width, label=labels[1], color=colors[1])
    for rect in rects1:
        rect.set_linewidth(1)
        rect.set_edgecolor('black')
    for rect in rects2:
        rect.set_linewidth(1)
        rect.set_edgecolor('black')
    plt.title(title, fontsize=fontsize)
    plt.xticks(x, categories, fontsize=fontsize)
    plt.yticks(np.arange(0, 1.1, 0.1), fontsize=fontsize)
    plt.legend(loc='upper center', bbox_to_anchor=(0.7, -0.05), shadow=False, ncol=2, fontsize=fontsize)
    plt.savefig(picture_path)
    plt.show()


def get_mean_and_std(numbers):
    mean = sum(numbers) / len(numbers)
    squared_diff = [(x - mean) ** 2 for x in numbers]
    variance = sum(squared_diff) / len(numbers)
    std_dev = math.sqrt(variance)
    return round(mean, 3), round(std_dev, 3)
