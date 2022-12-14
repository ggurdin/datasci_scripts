import pandas as pd
import os
import numpy as np
from sklearn.model_selection import StratifiedKFold


class ProteinDataset:
    '''
    Datasets are mainly used to store data, but can
    be used to create features via iFeature
    (should move this ability to featurizer.py in the future, should be more decoupled)

    Attributes:
        x (np.array): features vectors, or raw sequences if not encoded
        y (np.array): labels
        encoding (str): encoding scheme, if using iFeature
        features (list): list of features that the dataset contains
        original_data (np.array): stores original vectors. Used during feature selection to reset
        original_labels (np.array): stores original labels. Used when encoding binary labels to reset

    Examples:

        # create a dataset from labelled data

        dataset = ProteinDataset(
            labelled_input_file="../data/sequences_training.txt"
        )

        # to create features with iFeature, create a dataset and 
        # provide the path to iFeature, an embedding scheme
        # and set the encode flag to true. Optionally, you can provide a path to output encodings

        dataset = ProteinDataset(
            encoding_scheme="AAC",
            labelled_input_file="../data/sequences_training.txt",
            ifeature_path="../iFeature/iFeature.py",
            encoded_output_file="../data/aac.csv",
            encode=True
        )

        # to create a dataset from csvs of encoded features (generated either by iFeature
        # or by featurizer.py), create a dataset with a list of paths to 
        # the encoded files

        dataset = ProteinDataset(
            labelled_input_file="../data/sequences_training.txt",
            encoded_input_files=["../data/disorder.csv", "../data/entropy.csv", "../data/aac.csv"]
        )
    '''
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
        '''
        Initializes ProteinDataset
        If not inputs are provided, creates an empty dataset
        If encoded_input_files are provided, stores feature vectors as a combination of those files
            Will also set labels if labels are provided
        Otherwise, will store raw sequences (and labels if provided)
            Or, if encode flag is set to true and an iFeature path and encoding_scheme are provided,
            will encode data via iFeature

        Args:
            encoding_scheme (str): iFeature encoding scheme
            labelled_input_file (str): path to labelled data
            unlabelled_input_file (str): path to unlabelled data
            encoded_input_files (list): paths to encoded data
            ifeature_path (str): path to ifeature package, used for encoding
            encoded_output_file (str): path to output ifeature encodings
            encode (bool): flag for whether or not to encode with ifeature
        '''
        self.encoding = encoding_scheme
        self.features = None
        self.y = []

        if not encoded_input_files and not labelled_input_file and not unlabelled_input_file:
            print("Warning: creating an empty dataset")
            x = None

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
                if labelled_input_file:
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
            if labelled_input_file:
                self.set_labels(labelled_input_file)

        self.x = x


    def set_labels(self, fp):
        '''
        Sets labels from labelled input file

        Args:
            fp (str): path to labelled input file
        '''
        df = pd.read_csv(fp, names=["sequence", "label"])
        labels = df["label"].to_numpy()
        self.y = labels
        self.original_labels = labels


    def file_exists(self, fp):
        '''
        Checks if file exists

        Args:
            fp (str): path to check
        '''
        if not os.path.exists(fp):
            print(f"File path not found: {fp}")
            exit()

    
    def write_fasta(self, seqs):
        '''
        iFeature requires fasta format. Writes sequences to pseudo-fasta format

        Args:
            seqs (np.array): protein sequences to write
        '''
        output_path="./proteins.fasta"
        with open(output_path, "w") as f:
            for i, seq in enumerate(seqs):
                name = str(i)
                f.write(">" + name + "\n")
                f.write(seq + "\n")

    
    def write_encodings(self, output_file, ifeature_path="../iFeature/iFeature.py"):
        '''
        encodes via iFeature and writes as csv

        Args:
            output_file (str): path to output encodings
            ifeature_path (str): path to iFeature package
        '''
        if not os.path.exists(ifeature_path):
            print("\niFeature package not found. Please run:")
            print("\tgit clone https://github.com/Superzchen/iFeature.git\n")
            exit()
        os.system(f"python {ifeature_path} --file ./proteins.fasta --type {self.encoding} --out {output_file}")
        df = pd.read_csv(output_file, sep="\t")
        df = df.drop("#", axis=1)
        os.remove(output_file)
        df.to_csv(output_file, index=False)


    def read_input(self, encoded_fp=None, label_fp=None):
        '''
        reads in encoded data and labels

        Args:
            encoded_fp (str): path to file with features
            label_fp (str): path to file with labels
        '''
        train_df = pd.read_csv(encoded_fp)
        labels = None
        if label_fp:
            labels = np.array(pd.read_csv(label_fp, names=["sequence", "class"])["class"])
        return train_df, labels


    def split(self, dna_only=False, rna_only=False):
        '''
        splits data for 5-fold cross validation

        Args:
            dna_only (bool): whether to encode data as binary DNA vs. not DNA
            rna_only (bool): whether to encode data as binary RNA vs. not RNA
        '''
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
        '''Transforms self.y into binary labels'''
        labels = self.original_labels
        labels = (labels == "DNA") | (labels == "DRNA")
        labels = labels.astype(int)
        self.y = labels


    def rna_or_not(self):
        '''Transforms self.y into binary labels'''
        labels = self.original_labels
        labels = (labels == "RNA") | (labels == "DRNA")
        labels = labels.astype(int)
        self.y = labels


    def make_numpy(self):
        '''Transforms self.x from dataframe to np.array'''
        self.x = self.x.to_numpy()


    def make_df(self):
        '''Transforms self.x from np.array to dataframe'''
        self.x = pd.DataFrame(self.x, columns=self.features)


    def select_features(self, features):
        '''
        filters down self.x to list of features

        Args:
            features (np.array): list of features to select
        '''
        features = np.array([x for x in features if x == x and x != 'nan'])
        self.original_data = self.x
        self.x = self.x[features]

    def reset_features(self):
        '''resets features after feature selection'''
        self.x = self.original_data