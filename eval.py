import os
import pandas as pd
import numpy as np

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score

import argparse

def eval_metrics(actual, pred):
    accuracy = accuracy_score(actual, pred)
    precision = precision_score(actual, pred)
    recall = recall_score(actual, pred)
    f1 = f1_score(actual, pred)
    auc = roc_auc_score(actual, pred)
    return accuracy, precision, recall, f1, auc

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--ground_truth', type=str, help='Path to ground truth file')
    parser.add_argument('--predictions', type=str, help='Path to predictions file')
    args = parser.parse_args()

    # load the test data
    test_data_path = args.ground_truth
    test_data = pd.read_csv(test_data_path)

    # load the predictions
    predictions_path = args.predictions
    predictions = pd.read_csv(predictions_path)

    # evaluate the model
    y_true = test_data['sentiment']
    y_pred = predictions['sentiment']
    metrics = eval_metrics(y_true, y_pred)
    print(metrics)

    # log the metrics
    with open('metrics.txt', 'w') as f:
        f.write("accuracy: " + str(metrics[0]) + "\n")
        f.write("precision: " + str(metrics[1]) + "\n")
        f.write("recall: " + str(metrics[2]) + "\n")
        f.write("f1: " + str(metrics[3]) + "\n")
        f.write("auc: " + str(metrics[4]) + "\n")