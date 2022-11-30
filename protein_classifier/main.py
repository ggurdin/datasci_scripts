from protein_dataset import ProteinDataset
from protein_model import ProteinModel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from featurizer import Featurizer
from scipy.stats import pointbiserialr
from feature_selector import FeatureSelector
from sklearn.linear_model import LogisticRegression
from sklearn import preprocessing
from sklearn.model_selection import GridSearchCV
import numpy as np



def main():
    dataset = ProteinDataset(
        labelled_input_file="../data/sequences_training.txt",
        encoded_input_files=[
            "../data/disorder.csv", 
            "../data/entropy.csv", 
            "../data/aac.csv", 
            "../data/dpc.csv",
            "../data/aaindex.csv",
            "../data/bio_data.csv",
            "../data/ctriad.csv"
        ]
    )

    sel = FeatureSelector(dataset)
    rna_features = set(sel.select_from_scores(fp="../metrics/pbc_rna_scores.csv", threshold=0.2))
    dna_features = set(sel.select_from_scores(fp="../metrics/pbc_dna_scores.csv", threshold=0.2))
    features = np.array(list(dna_features.union(rna_features)))

    dataset.select_features(features)
    clf = RandomForestClassifier()
    # clf = KNeighborsClassifier(3)
    model = ProteinModel(clf, dataset)
    model.validate()


if __name__ == "__main__":
    main()
