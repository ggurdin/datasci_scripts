import numpy as np


class ProteinModel:

    def __init__(self, classifier, dataset):
        self.clf = classifier
        self.dataset = dataset

    
    def __str__(self):
        return "this is a model"

    def validate(self):
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
        
        cm = self.eval(predicted, real)
        self.print_metrics(cm)


    def eval(self, predicted, real):
        labels = ['nonDRNA', 'DRNA', 'RNA', 'DNA']
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
        return cm


    def print_metrics(self, cm):
        labels = ['nonDRNA', 'DRNA', 'RNA', 'DNA']
        print("\tActual: " + "\t".join(labels))
        print("Predicted:")
        for pred_label in labels:
            row_str = "\t" + pred_label + "\t"
            for actual_label in labels:
                row_str += str(cm[pred_label][actual_label]) + "\t"
            print(row_str)    