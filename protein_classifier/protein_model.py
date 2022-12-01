import numpy as np
import math
from sklearn.metrics import matthews_corrcoef, accuracy_score, multilabel_confusion_matrix, confusion_matrix


class ProteinModel:
    '''
    Models are trained on ProteinDatasets 

    Attributes:
        clf (sklearn classifier): model to train
        dataset (ProteinDataset): dataset to train, predict, or validate with
    '''
    def __init__(self, classifier, dataset):
        '''
        Initializes model

        Args:
            clf (sklearn classifer): classifer to train or validate
            dataset (ProteinDataset): dataset to train, predict, or validate with
        '''
        self.clf = classifier
        self.dataset = dataset

    def set_dataset(self, dataset):
        self.dataset = dataset

    def train(self):
        '''trains classifier'''
        x = self.dataset.x
        y = self.dataset.y
        self.clf.fit(x, y)

    def predict(self, dataset):
        '''predicts using trained classifier'''
        self.set_dataset(dataset)
        x = self.dataset.x
        preds = self.clf.predict(x)
        return preds

    def validate(self):
        '''5-fold cross validation'''
        predicted, real = [], []
        folds = self.dataset.split()
        for fold in folds:
            train, test = fold
            train_x, train_y = train
            self.clf.fit(train_x, train_y)

            test_x, test_y = test
            real.extend(test_y)
            predicted.extend(self.clf.predict(test_x))

        real = np.array(real)
        predicted = np.array(predicted)

        labels = ['nonDRNA', 'DRNA', 'RNA', 'DNA']
        self.eval(predicted, real, labels)


    def eval(self, predicted, real, labels):
        '''
        evaluates prediction performance during cross validation

        Args:
            predicted (np.array): list of predicted labels
            real (np.array): list of real labels
            labels (list): list of labels, used to determine order of confusion matrix
        '''
        real = np.array(real)
        predicted = np.array(predicted)
        print(labels)
        print(confusion_matrix(real, predicted, labels=labels))
        print("Accuracy: ", accuracy_score(real, predicted))
        print("MCC: ", matthews_corrcoef(real, predicted))

        labels = ["DNA", "RNA", "DRNA", "nonDRNA"]
        cms = multilabel_confusion_matrix(real, predicted, labels=labels)

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

            sens = 100 * (tp / (tp + fn))
            spec = 100 * (tn / (tn + fp))
            acc = 100 * ((tp + tn) / (tp + tn + fp + fn))

            print("Sens: ", sens)
            print("Spec: ", spec)
            print("Acc: ", acc)
            print("MCC: ", mcc_)
            print("\n")


    # def print_metrics(self, cm, sens, spec, acc, mcc, labels):
    #     print("\tActual: " + "\t".join([str(label) for label in labels]))
    #     print("Predicted:")
    #     for pred_label in labels:
    #         row_str = "\t" + str(pred_label) + "\t"
    #         for actual_label in labels:
    #             row_str += str(cm[pred_label][actual_label]) + "\t"
    #         print(row_str)  
    #     print("\n") 
    #     if 0 in labels:
    #         print(f"Sens: {sens}")
    #         print(f"Spec: {spec}")
    #         print(f"Acc: {acc}")
    #         print(f"MCC: {mcc}") 
    #     elif "DNA" in labels:
    #         for label in labels:
    #             print(f"{label} metrics:")
    #             print(f"\tSens: {sens[label]}")
    #             print(f"\tSpec: {spec[label]}")
    #             print(f"\tAcc: {acc[label]}")
    #             print(f"\tMCC: {mcc[label]}")
    #             print("\n")
