import numpy as np
import math


class ProteinModel:

    def __init__(self, classifier, dataset):
        self.clf = classifier
        self.dataset = dataset

    
    def __str__(self):
        return "this is a model"

    def validate(self, dna_only=False, rna_only=False):
        predicted, real = [], []
        folds = self.dataset.split(dna_only, rna_only)
        for fold in folds:
            train, test = fold
            train_x, train_y = train
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
        
        cm, sens, spec, acc, mcc = self.eval(predicted, real, labels)
        self.print_metrics(cm, sens, spec, acc, mcc, labels)


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

        if 0 in labels:
            tp = cm[1][1]
            tn = cm[0][0]
            fp = cm[1][0]
            fn = cm[0][1]

            sens = 100 * (tp / (tp + fn))
            spec = 100 * (tn / (tn + fp))
            acc = 100 * ((tp + tn) / (tp + tn + fp + fn))
            numerator = (tp * fn) - (fp * fn)
            denom = (tp + fp) * (tp + fn) * (tn + fp) * (tn + fn)
            denom = math.sqrt(denom)
            mcc = numerator / denom

        elif "DNA" in labels:
            tp_dna = cm["DNA"]["DNA"]
            tp_rna = cm["RNA"]["RNA"]
            tp_drna = cm["DRNA"]['DRNA']
            tp_nondrna = cm["nonDRNA"]["nonDRNA"]

            tn_dna = cm["nonDRNA"]["nonDRNA"] + cm["nonDRNA"]["RNA"] + cm["nonDRNA"]["DRNA"]
            tn_dna += cm["DRNA"]["DRNA"] + cm["DRNA"]["nonDRNA"] + cm["DRNA"]["RNA"]
            tn_dna += cm["RNA"]["RNA"] + cm["RNA"]["nonDRNA"] + cm["RNA"]["DRNA"]

            tn_rna = cm["nonDRNA"]["nonDRNA"] + cm["nonDRNA"]["DNA"] + cm["nonDRNA"]["DRNA"]
            tn_rna += cm["DRNA"]["DRNA"] + cm["DRNA"]["nonDRNA"] + cm["DRNA"]["DNA"]
            tn_rna += cm["DNA"]["DNA"] + cm["DNA"]["nonDRNA"] + cm["DNA"]["DRNA"]

            tn_drna = cm["nonDRNA"]["nonDRNA"] + cm["nonDRNA"]["DNA"] + cm["nonDRNA"]["RNA"]
            tn_drna += cm["RNA"]["RNA"] + cm["RNA"]["nonDRNA"] + cm["RNA"]["DNA"]
            tn_drna += cm["DNA"]["DNA"] + cm["DNA"]["nonDRNA"] + cm["DNA"]["RNA"]

            tn_nondrna = cm["DRNA"]["DRNA"] + cm["DRNA"]["DNA"] + cm["DRNA"]["RNA"]
            tn_nondrna += cm["RNA"]["RNA"] + cm["RNA"]["DRNA"] + cm["RNA"]["DNA"]
            tn_nondrna += cm["DNA"]["DNA"] + cm["DNA"]["DRNA"] + cm["DNA"]["RNA"]

            fp_dna = cm["DNA"]["nonDRNA"] + cm["DNA"]["RNA"] + cm["DNA"]["DRNA"]
            fp_rna = cm["RNA"]["nonDRNA"] + cm["RNA"]["DNA"] + cm["RNA"]["DRNA"]
            fp_drna = cm["DRNA"]["nonDRNA"] + cm["DRNA"]["RNA"] + cm["DRNA"]["DNA"]
            fp_nondrna = cm["nonDRNA"]["DRNA"] + cm["nonDRNA"]["RNA"] + cm["nonDRNA"]["DNA"]

            fn_dna = cm["nonDRNA"]["DNA"] + cm["DRNA"]["DNA"] + cm["RNA"]["DNA"]
            fn_rna = cm["nonDRNA"]["RNA"] + cm["DRNA"]["RNA"] + cm["DNA"]["RNA"]
            fn_drna = cm["nonDRNA"]["DRNA"] + cm["DNA"]["DRNA"] + cm["RNA"]["DRNA"]
            fn_nondrna = cm["DRNA"]["nonDRNA"] + cm["RNA"]["nonDRNA"] + cm["DNA"]["nonDRNA"]

            sens_dna = 100 * (tp_dna / (tp_dna + fn_dna))
            spec_dna = 100 * (tn_dna / (tn_dna + fp_dna))
            acc_dna = 100 * ((tp_dna + tn_dna) / (tp_dna + tn_dna + fp_dna + fn_dna))
            mcc_dna = ((tp_dna * fn_dna) - (fp_dna * fn_dna)) / (math.sqrt((tp_dna + fp_dna) * (tp_dna + fn_dna) * (tn_dna + fp_dna) * (tn_dna + fn_dna)))

            sens_rna = 100 * (tp_rna / (tp_rna + fn_rna))
            spec_rna = 100 * (tn_rna / (tn_rna + fp_rna))
            acc_rna = 100 * ((tp_rna + tn_rna) / (tp_rna + tn_rna + fp_rna + fn_rna))
            mcc_rna = ((tp_rna * fn_rna) - (fp_rna * fn_rna)) / (math.sqrt((tp_rna + fp_rna) * (tp_rna + fn_rna) * (tn_rna + fp_rna) * (tn_rna + fn_rna)))

            sens_drna = 100 * (tp_drna / (tp_drna + fn_drna))
            spec_drna = 100 * (tn_drna / (tn_drna + fp_drna))
            acc_drna = 100 * ((tp_drna + tn_drna) / (tp_drna + tn_drna + fp_drna + fn_drna))
            mcc_drna = ((tp_drna * fn_drna) - (fp_drna * fn_drna)) / (math.sqrt((tp_drna + fp_drna) * (tp_drna + fn_drna) * (tn_drna + fp_drna) * (tn_drna + fn_drna)))

            sens_nondrna = 100 * (tp_nondrna / (tp_nondrna + fn_nondrna))
            spec_nondrna = 100 * (tn_nondrna / (tn_nondrna + fp_nondrna))
            acc_nondrna = 100 * ((tp_nondrna + tn_nondrna) / (tp_nondrna + tn_nondrna + fp_nondrna + fn_nondrna))
            mcc_nondrna = ((tp_nondrna * fn_nondrna) - (fp_nondrna * fn_nondrna)) / (math.sqrt((tp_nondrna + fp_nondrna) * (tp_nondrna + fn_nondrna) * (tn_nondrna + fp_nondrna) * (tn_nondrna + fn_nondrna)))

            sens = {"DNA": sens_dna, "RNA": sens_rna, "DRNA": sens_drna, "nonDRNA": sens_nondrna}
            spec = {"DNA": spec_dna, "RNA": spec_rna, "DRNA": spec_drna, "nonDRNA": spec_nondrna}
            acc = {"DNA": acc_dna, "RNA": acc_rna, "DRNA": acc_drna, "nonDRNA": acc_nondrna}
            mcc = {"DNA": mcc_dna, "RNA": mcc_rna, "DRNA": mcc_drna, "nonDRNA": mcc_nondrna}

        return cm, sens, spec, acc, mcc


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
