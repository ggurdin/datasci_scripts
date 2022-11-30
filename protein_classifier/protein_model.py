import numpy as np
import math
from sklearn.utils import resample
import pandas as pd
from sklearn.metrics import matthews_corrcoef, multilabel_confusion_matrix


class ProteinModel:

    def __init__(self, classifier, dataset):
        self.clf = classifier
        self.dataset = dataset

    
    def __str__(self):
        return "this is a model"

    def upsample(self, x):
        pass

    def validate(self, dna_only=False, rna_only=False):
        predicted, real = [], []
        folds = self.dataset.split(dna_only, rna_only)
        for fold in folds:
            train, test = fold
            train_x, train_y = train

            # x_positive, y_positive = resample(
            #     train_x[train_y== 1],
            #     train_y[train_y == 1],
            #     replace=True,
            #     n_samples=train_x[train_y == 0].shape[0],
            #     random_state=123
            # )
            # x_negative = train_x[train_y == 0]
            # y_negative = train_y[train_y == 0]
            
            # train_x = np.concatenate([x_positive, x_negative])
            # train_y = np.concatenate([y_positive, y_negative])

            self.clf.fit(train_x, train_y)

            test_x, test_y = test
            real.extend(test_y)
            predicted.extend(self.clf.predict(test_x))

        real = np.array(real)
        predicted = np.array(predicted)

        if dna_only or rna_only:
            labels = [0, 1]
        else:
            labels = ['nonDRNA', 'DRNA', 'RNA', 'DNA']
        
        self.eval(predicted, real, labels)


    def eval(self, predicted, real, labels):
        real = np.array(real)
        predicted = np.array(predicted)
        cm = {}

        for pred_label in labels:
            pred_indices = np.argwhere(predicted == pred_label)
            reals = real[pred_indices]

            for actual_label in labels:
                if pred_label not in cm:
                    cm[pred_label] = {}
                cm[pred_label][actual_label] = np.sum((reals == actual_label).astype(int))

        labels = ["DNA", "RNA", "DRNA", "nonDRNA"]
        cms = multilabel_confusion_matrix(real, predicted, labels=labels)

        mccs, senss, specs, accs = [], [], [], []
        for i, cm in enumerate(cms):
            print(labels[i])
            print(cm)
            tn = cm[0][0]
            fn = cm[1][0]
            fp = cm[0][1]
            tp = cm[1][1]
            if ((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)) == 0:
                mcc_ = 0
            else:
                mcc_ = ((tp * tn) - (fp * fn)) / math.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
            mccs.append(mcc_)

            sens = 100 * (tp / (tp + fn))
            spec = 100 * (tn / (tn + fp))
            acc = 100 * ((tp + tn) / (tp + tn + fp + fn))

            senss.append(sens)
            specs.append(sens)
            accs.append(acc)

            print("Sens: ", sens)
            print("Spec: ", spec)
            print("Acc: ", acc)
            print("MCC: ", mcc_)
            print("\n")

        avg_mcc = sum(mccs) / len(mccs)
        avg_acc = sum(accs) / len(accs)
        print("Average MCC: ", avg_mcc)
        print("Average accuracy: ", avg_acc)


    def print_metrics(self, cm, sens, spec, acc, mcc, labels):
        print("\tActual: " + "\t".join([str(label) for label in labels]))
        print("Predicted:")
        for pred_label in labels:
            row_str = "\t" + str(pred_label) + "\t"
            for actual_label in labels:
                row_str += str(cm[pred_label][actual_label]) + "\t"
            print(row_str)  
        print("\n") 
        if 0 in labels:
            print(f"Sens: {sens}")
            print(f"Spec: {spec}")
            print(f"Acc: {acc}")
            print(f"MCC: {mcc}") 
        elif "DNA" in labels:
            for label in labels:
                print(f"{label} metrics:")
                print(f"\tSens: {sens[label]}")
                print(f"\tSpec: {spec[label]}")
                print(f"\tAcc: {acc[label]}")
                print(f"\tMCC: {mcc[label]}")
                print("\n")
