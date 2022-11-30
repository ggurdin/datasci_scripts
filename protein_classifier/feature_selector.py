import pandas as pd
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier
from scipy.stats import pointbiserialr


class FeatureSelector:

    def __init__(self, dataset=None):
        self.x = dataset.x
        self.y = dataset.y

    def feature_elimination(self, output="../metrics/rf_scores.csv"):
        clf = RandomForestClassifier(random_state=123)
        clf.fit(self.x, self.y)
        scores = clf.feature_importances_
        features = clf.feature_names_in_

        score_tuples = [(name, score) for name, score in zip(features, scores)]
        score_tuples = sorted(score_tuples, key=lambda x: x[1], reverse=True)
        df = pd.DataFrame(score_tuples, columns=["feature", "score"])
        df.to_csv(output, index=False)

    def correlation(self, output="../metrics/pbc_dna_scores.csv", rna=False, dna=True):
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
        scores = pd.read_csv(fp)
        if top_n:
            top_n_features = scores[:top_n]
            features = top_n_features["feature"].to_numpy()
        elif threshold:
            threshold_features = scores.loc[scores["score"] >= threshold]
            features = threshold_features["feature"].to_numpy()
        return features

