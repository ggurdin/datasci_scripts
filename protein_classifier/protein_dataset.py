import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


class ProteinDataset:

    def __init__(
        self, 
        encoding_scheme=None,
        labelled_input_file=None,
        unlabelled_input_file=None,
        encoded_input_files=None,
        ifeature_path="../iFeature/iFeature.py",
        encoded_output_file="../data/encodings.csv",
        encode=False
    ):
        self.encoding = encoding_scheme
        self.features = None

        if not encoded_input_files and not labelled_input_file:
            print("Warning: creating an empty dataset")
            x = None
            self.y = None

        elif not encoded_input_files:
            input_file = None
            if unlabelled_input_file:
                input_file = unlabelled_input_file
                self.file_exists(input_file)
                df = pd.read_csv(input_file, names=["sequence"])
            elif labelled_input_file:
                input_file = labelled_input_file
                self.file_exists(input_file)
                df = pd.read_csv(input_file, names=["sequence", "label"])
            if encode:
                self.write_fasta(df["sequence"])
                self.write_encodings(encoded_output_file, ifeature_path)
                os.remove("./proteins.fasta")
                x, _ = self.read_input(encoded_fp=encoded_output_file)
                self.features = list(x.columns)
                self.set_labels(labelled_input_file)
            else:
                self.features = ["sequence"]
                x = df["sequence"]
                if labelled_input_file:
                    self.set_labels(labelled_input_file)

        else:
            dfs = []
            for f in encoded_input_files:
                self.file_exists(f)
                x, _ = self.read_input(encoded_fp=f)
                dfs.append(x)
            x = pd.concat(dfs, axis=1)
            self.features = list(x.columns)
            self.set_labels(labelled_input_file)

        self.x = x


    def set_labels(self, fp):
        df = pd.read_csv(fp, names=["sequence", "label"])
        labels = df["label"].to_numpy()
        self.y = labels
        self.original_labels = labels


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
        df = pd.read_csv(output_file, sep="\t")
        df = df.drop("#", axis=1)
        os.remove(output_file)
        df.to_csv(output_file, index=False)


    def read_input(self, encoded_fp=None, label_fp=None):
        train_df = pd.read_csv(encoded_fp)
        labels = None
        if label_fp:
            labels = np.array(pd.read_csv(label_fp, names=["sequence", "class"])["class"])
        return train_df, labels


    def split(self, dna_only=False, rna_only=False):
        if dna_only:
            self.dna_or_not()
        elif rna_only:
            self.rna_or_not()
        if isinstance(self.x, pd.core.frame.DataFrame):
            self.make_numpy()
        skf = StratifiedKFold(n_splits=5)
        split_data = []
        for train_idx, test_idx in skf.split(self.x, self.y):
            train_patterns = self.x[train_idx]
            train_labels = self.y[train_idx]
            test_patterns = self.x[test_idx]
            test_labels = self.y[test_idx]
            split_data.append(((train_patterns, train_labels), (test_patterns, test_labels)))
        return split_data


    def dna_or_not(self):
        labels = self.original_labels
        labels = (labels == "DNA") | (labels == "DRNA")
        labels = labels.astype(int)
        self.y = labels


    def rna_or_not(self):
        labels = self.original_labels
        labels = (labels == "RNA") | (labels == "DRNA")
        labels = labels.astype(int)
        self.y = labels


    def make_numpy(self):
        self.x = self.x.to_numpy()


    def make_df(self):
        self.x = pd.DataFrame(self.x, columns=self.features)