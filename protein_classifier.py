"""
Script to create features and cross validate classification algorithms 
to classify protein amino acid sequences' interctions with DNA and RNA

Features are generated using iFeature: https://github.com/Superzchen/iFeature
Classifiction algorithms are from sklearn: https://scikit-learn.org/stable/

CLI directions:
    1. Ensure that iFeature is installed in your working directory.
        To install iFeature, run:
        git clone https://github.com/Superzchen/iFeature.git
    2. Run:
            pip install requirements.txt
        to install the required python libraries
    3. Run this command:
        python protein_classifier.py -f <FILE-NAME> -e <ENCODING-SCHEME> -c <CLASSIFIER> -o <OUTPUT-FILE>

        FILE-NAME -> path to .txt file containing protein sequences and labels
        ENCODING-SCHEME -> the iFeature encoding to use
        CLASSIFIER -> the sklearn classifier to train and evaluate
            options currently available: kNN, SVM, LinearSVM, MLP, RandomForest, NB
        OUTPUT-FILE (optional) -> file path to output metric evaluations. 
            If not specified, output will only be writen to stdout

*** the CLI cannot currently change classifier parameters ***
*** parameters can be changed for individual classifiers by changing them in the "decode_classifier()" function ***

Encoding Scheme Description:
source: https://academic.oup.com/bioinformatics/article/34/14/2499/4924718

'AAC' -> Amino Acid Composition
'DPC' -> Dipeptide composition
'DDE' -> Dipeptide deviation from expected mean
'GAAC' -> Grouped amino acid composition
'CKSAAGP' -> Composition of k-spaced amino acid group pairs
'GDPC' -> Grouped dipeptide composition
'GTPC' -> Grouped tripeptide composition (GTPC) 
'CTDC' -> Composition
'CTDT' -> Transition
'CTDD' -> Distribution
'CTriad' -> Conjoint triad
"""

import argparse
import math
import os
from datetime import datetime
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold
from sklearn import svm
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import confusion_matrix


def write_fasta(txt_path = "./sequences_training.txt", output_path = "./proteins.fasta"):
    """
    writes .txt input sequences to pseudo-fasta format. iFeature requires input in fasta format
    """
    if not os.path.exists(txt_path):
        print("Input file not found")
        exit()
    df = pd.read_csv(txt_path, names=["sequence", "class"])
    seqs = df["sequence"]
    with open(output_path, "w") as f:
        for i, seq in enumerate(seqs):
            name = str(i)
            f.write(">" + name + "\n")
            f.write(seq + "\n")


def read_input(train_fp="./encoding.tsv", label_fp="./sequences_training.txt"):
    """
    read fasta file and input labels into format suitable to train sklearn classifier (numpy arrays)
    """
    if not os.path.exists(train_fp) or not os.path.exists(label_fp):
        print("Training file not found")
        exit()
    train_df = pd.read_csv(train_fp, sep="\t")
    train_df = train_df.set_index("#").to_numpy()
    
    labels = np.array(pd.read_csv(label_fp, names=["sequence", "class"])["class"])
    return train_df, labels

def validate(classifier, x, y):
    """
    Run 5 fold cross validation
    """

    # get list of unique labels to train
    labels = np.unique(y)

    all_metrics = {}
    # iterate through each label
    for label in labels:
        print(f"\tTraining for label: {label}")
        # generate binary labels (if input is label, 1, else 0)
        binary_labels = y == label
        binary_labels = binary_labels.astype(int)

        skf = StratifiedKFold(n_splits=5)
        fold_metrics = {"sensitivity": [], "specificity": [], "accuracy": [], "mcc": []}
        for train_index, test_index in skf.split(x, binary_labels):
            x_train, x_test = x[train_index], x[test_index]
            y_train, y_test = binary_labels[train_index], binary_labels[test_index]
            clf = classifier
            clf.fit(x_train, y_train)
            preds = clf.predict(x_test)
            sensitivity, specificity, accuracy, mcc = eval(y_test, preds)

            fold_metrics["sensitivity"].append(sensitivity) 
            fold_metrics["specificity"].append(specificity)
            fold_metrics["accuracy"].append(accuracy)
            fold_metrics["mcc"].append(mcc)

        for key, value in fold_metrics.items():
            fold_metrics[key] = {"avg": sum(value) / len(value), "std": np.std(value)}
        all_metrics[label] = fold_metrics
    return all_metrics


def eval(y_true, y_pred):
    """
    generate metrics from predicted and actual labels
    """
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    sensitivity = 100 * (tp / (tp + fn))
    specificity = 100 * (tn / (tn + fp))
    accuracy = 100 * ((tp + tn) / (tp + tn + fp + fn))
    mcc = math.sqrt(abs((tp + fp) * (tp + fn) * (tp + fp) * (tn + fn)))
    if not mcc == 0:
        numerator = (tp * tn) - (fp * fn)
        denominator = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
        denominator = abs(denominator)
        denominator = math.sqrt(denominator)
        mcc = numerator / denominator
    return sensitivity, specificity, accuracy, mcc


def decode_classifier(classifier_name):
    # util function to create sklearn classifier from cl input string
    if classifier_name == 'kNN':
        return KNeighborsClassifier(2)
    if classifier_name == "SVM":
        return svm.SVC(kernel="poly", degree=3)
    if classifier_name == "LinearSVM":
        return svm.LinearSVC()
    if classifier_name == "MLP":
        return MLPClassifier()
    if classifier_name == "RandomForest":
        return RandomForestClassifier()
    if classifier_name == "NB":
        return GaussianNB()


def print_metrics(metrics, classifier, encoding, fp=None):
    # util function to print metrics
    print(f"Result for training {classifier} classifier with {encoding} encoding:\n")
    if fp:
        now = datetime.now()
        dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
        with open(fp, "a") as f:
            f.write(f"\n[{dt_string}] Result for training {classifier} classifier with {encoding} encoding:\n")
    for label, results in metrics.items():
        print(f"\nResults for label {label}:")
        print("\tSensitivity: %.2f +/- %.2f" % (results["sensitivity"]["avg"], results["sensitivity"]["std"]))
        print("\tSpecificity: %.2f +/- %.2f" % (results["specificity"]["avg"], results["specificity"]["std"]))
        print("\tAccuracy: %.2f +/- %.2f" % (results["accuracy"]["avg"], results["accuracy"]["std"]))
        print("\tMCC: %.2f +/- %.2f" % (results["mcc"]["avg"], results["mcc"]["std"]))
        if fp:
            with open(fp, "a") as f:
                f.write(f"\nResults for label {label}:\n")
                f.write("\tSensitivity: %.2f +/- %.2f\n" % (results["sensitivity"]["avg"], results["sensitivity"]["std"]))
                f.write("\tSpecificity: %.2f +/- %.2f\n" % (results["specificity"]["avg"], results["specificity"]["std"]))
                f.write("\tAccuracy: %.2f +/- %.2f\n" % (results["accuracy"]["avg"], results["accuracy"]["std"]))
                f.write("\tMCC: %.2f +/- %.2f\n" % (results["mcc"]["avg"], results["mcc"]["std"]))
        

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-f', '--file', required=True, help="path to training file")
    parser.add_argument(
        '-e', '--encoding', 
        required=True, 
        help="iFeature encdoing scheme", 
        choices=['AAC', 'DPC', 'DDE', 'GAAC', 'CKSAAGP', 'GDPC', 'GTPC', 'CTDC', 'CTDT', 'CTDD', 'CTriad']
    )
    parser.add_argument(
        '-c', '--classifier', 
        required=True, 
        help="classification algorithm to train",
        choices=["kNN", "SVM", "LinearSVM", "MLP", "RandomForest", "NB"]
    )
    parser.add_argument(
        '-o', '--output', 
        help="file to output results. If not specified, results will not be writen to file"
    )
    args = parser.parse_args()

    file = args.file
    encoding = args.encoding
    classifier = args.classifier
    output_file = None
    if args.output:
        output_file = args.output
    
    if not os.path.exists(file):
        print("Input file not found")
        exit()

    if not os.path.exists("./iFeature"):
        print("\niFeature package not found. Please run:")
        print("\tgit clone https://github.com/Superzchen/iFeature.git\n")
        exit()

    print("\nWriting fasta file to ./proteins.fasta...\n")
    write_fasta(file)

    print("Writing feature encodings to ./encoding.tsv...")
    os.system(f"python ./iFeature/iFeature.py --file ./proteins.fasta --type {encoding}")

    print("\nReading encoded training data...\n")
    x, y = read_input(label_fp=file)

    clf = decode_classifier(classifier)
    print("Running 5 Fold Cross Validation...")
    metrics = validate(clf, x, y)
    print_metrics(metrics, classifier, encoding, output_file)


if __name__ == "__main__":
    main()
