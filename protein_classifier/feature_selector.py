import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pointbiserialr


class FeatureSelector:
    '''
    Used to selected features for training model

    Attributes:
        x (np.array): protein feature vectors
        y (np.array): protein labels
    '''
    def __init__(self, dataset=None):
        '''
        Initializes FeatureSelector object

        Args:
            dataset (ProteinDataset): dataset from which to select features
        '''
        self.x = dataset.x
        if len(dataset.y) > 0:
            self.y = dataset.y

    def feature_elimination(self, output="../metrics/rf_scores.csv"):
        '''
        Feature selection via wrapper method with Random Forest
        Trains sklearn RandomForest on all features and outputs 
        their feature_importances to a file

        Args:
            output (str): file to output features and scores
        '''
        clf = RandomForestClassifier(random_state=123)
        clf.fit(self.x, self.y)
        scores = clf.feature_importances_
        features = clf.feature_names_in_

        score_tuples = [(name, score) for name, score in zip(features, scores)]
        score_tuples = sorted(score_tuples, key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(score_tuples, columns=["feature", "score"])
        df.to_csv(output, index=False)

    def correlation(self, output="../metrics/pbc_dna_scores.csv", rna=False, dna=False):
        '''
        Feature selection via point biserial correlation. Splits dataset
        into 2 binary sets (DNA vs. not DNA and RNA vs. not RNA) then calculates
        and outputs the PBC between the features and binary labels

        Args:
            output (str): file to output features and scores
            rna (bool): generate RNA vs. not RNA scores
            dna (bool): generate DNA vs. not RNA scores
        '''
        if (rna and dna) or (not dna and not rna):
            print("Please select either DNA or RNA, and not both")
            exit()
        feature_names = list(self.x.columns)
        features = self.x.to_numpy().transpose()
        if rna:
            labels = (self.y == "RNA") | (self.y == "DRNA")
        elif dna:
            labels = (self.y == "DNA") | (self.y == "DRNA")
        labels = labels.astype(int)
        
        score_tuples = []
        for i, f in enumerate(features):
            pbc = abs(pointbiserialr(f, labels).correlation)
            score_tuples.append((feature_names[i], pbc))
        score_tuples = sorted(score_tuples, key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(score_tuples, columns=["feature", "score"])
        df = df.dropna(axis="rows")
        df.to_csv(output, index=False)


    def select_from_scores(self, fp="../metrics/rf_scores.csv", top_n=None, threshold=None):
        '''
        Given a file with feature importance scores, selects a list of features
        Either top-N features of features whose scores reach a threshold

        Args:
            fp (str): path to feature score file
            top_n (int): number of features to select
            threshold (float): threshold for feature selection

        Returns:
            features (np.array): a list of selected feature names
        '''
        scores = pd.read_csv(fp)
        if top_n:
            top_n_features = scores[:top_n]
            features = top_n_features["feature"].to_numpy()
        elif threshold:
            threshold_features = scores.loc[scores["score"] >= threshold]
            features = threshold_features["feature"].to_numpy()
        return features


    def combine_features(self, features):
        combined = set()
        for feature in features:
            feature = set(feature)
            combined = combined.union(feature)
        return np.array(list(combined))

