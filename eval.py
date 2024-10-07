import os
import pandas as pd
import csv

def performance(TP, FP, FN):
    if (TP + FP) == 0:
        precision = float("NaN")
    else:
        precision = TP / float((TP + FP))

    if (TP + FN) == 0:
        recall = float("NaN")
    else:
        recall = TP / float((TP + FN))

    if (recall != float("NaN")) and (precision != float("NaN")):
        f1_score = (2.0 * precision * recall) / (precision + recall)
    else:
        f1_score = float("NaN")

    return precision, recall, f1_score

def get_dataset_metrics(true_labels, pred_labels):
    metrics_dict = dict()

    for true_label, pred_label in zip(true_labels, pred_labels):
        if true_label not in metrics_dict:
            metrics_dict[true_label] = {"TP": 0, "FP": 0, "FN": 0, "Support": 0}

        if true_label != "OTHER":
            metrics_dict[true_label]["Support"] += 1
            
            if true_label == pred_label:
                metrics_dict[true_label]["TP"] += 1
            elif pred_label == "OTHER":
                metrics_dict[true_label]["FN"] += 1
        else:
            if pred_label != "OTHER":
                metrics_dict[pred_label]["FP"] += 1

    df = pd.DataFrame()

    for field in metrics_dict:
        precision, recall, f1_score = performance(
            metrics_dict[field]["TP"], 
            metrics_dict[field]["FP"], 
            metrics_dict[field]["FN"]
        )
        support = metrics_dict[field]["Support"]
        
        if field != "OTHER":
            temp_df = pd.DataFrame([[precision, recall, f1_score, support]], 
                                    columns=["Precision", "Recall", "F1-Score", "Support"], 
                                    index=[field])
            df = pd.concat([df, temp_df])  # Use concat instead of append for efficiency

    return df

def get_doc_labels(doc_true, doc_pred):
    true_labels = [row[-1] for row in csv.reader(open(doc_true, "r"))]
    pred_labels = [row[-1] for row in csv.reader(open(doc_pred, "r"))]
    return true_labels, pred_labels

def get_dataset_labels(true_path, pred_path, save=False):
    y_true, y_pred = [], []

    for true_file in os.listdir(true_path):
        if true_file.endswith(".tsv"):  # Check if it's a TSV file
            pred_file = true_file  # Match predicted file with the true file
            true_file_path = os.path.join(true_path, true_file)
            pred_file_path = os.path.join(pred_path, pred_file)

            if os.path.exists(pred_file_path):
                true_labels, pred_labels = get_doc_labels(true_file_path, pred_file_path)
                y_true.extend(true_labels)
                y_pred.extend(pred_labels)

    df = get_dataset_metrics(y_true, y_pred)
    print(df)

    if save:
        df.to_csv("eval_metrics.tsv")

if __name__ == "__main__":
    doc_true = f"{os.getcwd()}/train/boxes_transcripts_labels"
    doc_pred = f"{os.getcwd()}/train/boxes_transcripts_labels"
    get_dataset_labels(doc_true, doc_pred, save=False)
