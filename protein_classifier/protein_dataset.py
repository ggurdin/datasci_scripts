import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


class ProteinDataset:

    def __init__(
        self, 
        encoding_scheme,
        labelled_input_file=None,
        unlabelled_input_file=None,
        encoded_input_file=None,
        ifeature_path="../iFeature/iFeature.py",
        encoded_output_file="../data/encodings.tsv",
    ):
        self.encoding = encoding_scheme

        if not encoded_input_file:
            input_file = None
            if unlabelled_input_file:
                input_file = unlabelled_input_file
                self.file_exists(input_file)
                df = pd.read_csv(input_file, names=["sequence"])
            elif labelled_input_file:
                input_file = labelled_input_file
                self.file_exists(input_file)
                df = pd.read_csv(input_file, names=["sequence", "label"])
            self.write_fasta(df["sequence"])
            self.write_encodings(encoded_output_file, ifeature_path)
            os.remove("./proteins.fasta")
            x, y = self.read_input(encoded_fp=encoded_output_file, label_fp=input_file)

        else:
            self.file_exists(encoded_input_file)
            x, y = self.read_input(encoded_fp=encoded_input_file, label_fp=labelled_input_file)

        self.x = x
        self.y = y


    def __str__(self):
        ret = "Protein Dataset"
        ret += f"\nEncoding Scheme: {self.encoding}"
        ret += f"\nLength: {len(self.x)}"
        return ret


    def file_exists(self, fp):
        if not os.path.exists(fp):
            print(f"File path not found: {fp}")
            exit()

    
    def write_fasta(self, seqs):
        output_path="./proteins.fasta"
        with open(output_path, "w") as f:
            for i, seq in enumerate(seqs):
                name = str(i)
                f.write(">" + name + "\n")
                f.write(seq + "\n")

    
    def write_encodings(self, output_file, ifeature_path="../iFeature/iFeature.py"):
        if not os.path.exists("../iFeature"):
            print("\niFeature package not found. Please run:")
            print("\tgit clone https://github.com/Superzchen/iFeature.git\n")
            exit()
        os.system(f"python {ifeature_path} --file ./proteins.fasta --type {self.encoding} --out {output_file}")


    def read_input(self, encoded_fp=None, label_fp=None):
        train_df = pd.read_csv(encoded_fp, sep="\t")
        train_df = train_df.set_index("#").to_numpy()
        
        labels = None
        if label_fp:
            labels = np.array(pd.read_csv(label_fp, names=["sequence", "class"])["class"])
        return train_df, labels

    def split(self):
        skf = StratifiedKFold(n_splits=5)
        split_data = []
        for train_idx, test_idx in skf.split(self.x, self.y):
            train_patterns = self.x[train_idx]
            train_labels = self.y[train_idx]
            test_patterns = self.x[test_idx]
            test_labels = self.y[test_idx]
            split_data.append(((train_patterns, train_labels), (test_patterns, test_labels)))
        return split_data