import os, sys
import argparse
import numpy as np
import pickle
from sklearn.metrics import accuracy_score, f1_score, roc_curve, recall_score, roc_auc_score, precision_score
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, RocCurveDisplay
import matplotlib.pyplot as plt

ROOT_DIRECTORY = './../'
ROOT_DIRECTORY = os.path.abspath( os.path.join(os.getcwd(), ROOT_DIRECTORY) )
sys.path.append(ROOT_DIRECTORY)
os.chdir(ROOT_DIRECTORY)

from utils.utils import print_log, get_timestring

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', default='results/multiagent_bert_test/model_test_6/results')
args = parser.parse_args()


time_str = get_timestring()
log = open(os.path.join(args.file_path, 'metrics.txt'), 'a+')
print_log("time str: {}".format(time_str), log)


p_files = [f for f in os.listdir(args.file_path) if f.endswith("predictions.pkl")]
gt_files = [f for f in os.listdir(args.file_path) if f.endswith("labels.pkl")]


p_data = {}
gt_data = {}


for p, gt in zip(p_files, gt_files):
    with open(os.path.join(args.file_path, p), 'rb') as f:
        p_d = pickle.load(f)
    with open(os.path.join(args.file_path, gt), 'rb') as f:
        gt_d = pickle.load(f)


    p_data[p.split('_')[-2]] = p_d
    gt_data[gt.split('_')[-2]] = gt_d


fig, axis = plt.subplots(1)
for k in p_data.keys():
    print_log(f" Metrics for the {k} attack".center(70, "="), log)

    f1 = f1_score(gt_data[k], p_data[k])
    fpr, tpr, thresholds = roc_curve(gt_data[k], p_data[k])
    recall = recall_score(gt_data[k], p_data[k])
    auc = roc_auc_score(gt_data[k], p_data[k])
    precision = precision_score(gt_data[k], p_data[k])
    cm = confusion_matrix(gt_data[k], p_data[k],normalize='true')

    print_log(f" F1 : {f1}", log)
    print_log(f" Precision : {precision}", log)
    print_log(f" Recall : {recall}", log)
    print_log(f" AUC : {auc}", log)
    print_log(f" FPR : {fpr}", log)
    print_log(f" TPR : {tpr}", log)

    plt.figure()
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    plt.rcParams.update({'font.size': 16})
    disp.plot()
    plt.title(f"{k.capitalize()}")
    plt.show()
    plt.savefig(f'{args.file_path}/{k}_confusion.png',
          fontsize = 20)

    plt.figure()
    disp =RocCurveDisplay.from_predictions(gt_data[k], p_data[k])
    disp.plot()
    plt.title(f"{k.capitalize()}")
    plt.show()
    plt.savefig(f'{args.file_path}/{k}_roc.png',
          fontsize = 20)

    axis.plot(fpr,tpr,label=f'{k.capitalize()} (AUC={auc:.2f})')

axis.set_xlabel("True Positive Rate")
axis.set_ylabel("False Positive Rate")
axis.legend()
fig.savefig(f'{args.file_path}/ROC_all.png')

