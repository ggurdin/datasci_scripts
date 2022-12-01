import numpy as np
from sklearn.ensemble import RandomForestClassifier
from protein_dataset import ProteinDataset
from protein_model import ProteinModel
from feature_selector import FeatureSelector
from featurizer import Featurizer

def main():
    # Usage Guide:

    # Installation:
    '''
    After cloning from github, run:
        pip install -e .
    This should install all the required packages
    '''

    # Create Features
    '''
    If you want to generate features, there are two ways:

    -> With iFeature
        Make sure you have iFeature installed
        Then, create a ProteinDataset:
    '''
    dataset = ProteinDataset(
        labelled_input_file="../../data_train/sequences_training.txt",  # replace with path to labelled data
        encoding_scheme="AAC", # replace with your desired encoding scheme
        ifeature_path="../../iFeature/iFeature.py",
        encoded_output_file="../../demo/aac.csv", # path to output encodings
        encode=True  # flag tells the dataset whether or not to perform encoding
    )

    '''
    -> With a Featurizer
        First, create a dataset like so
    '''
    dataset = ProteinDataset(
        labelled_input_file="../../data_train/sequences_training.txt"  # replace with path to labelled data
    )
    ''' Then create a featurizer for that dataset '''
    ft = Featurizer(dataset)
    ''' 
    The featurizer has lots of function, which can be found in featurizer.py
    Example usage:
    '''
    ft.biopython_features(output="../../demo/biofeatures.csv")


    '''
    Once you have generated your desired features, the next step in feature selection
    There are two main methods available:

    1) Random Forest Wrapper
        -> train a RF on all features, and rank them based on feature_importance
    2) Point Biserial Correlation Scoring
        -> Calculated the PBC between each feature and the labels
    Once you've calculated scores, you can selected features by a threshold score 
    or select top-N features

    Example:
    '''
    ''' first, create a dataset from encoded features: '''
    dataset = ProteinDataset(
        labelled_input_file="../../data_train/sequences_training.txt",
        encoded_input_files=["../../demo/aac.csv", "../../demo/biofeatures.csv"]
    )
    '''
    this will read in and combine features from the list of files
    next, create a feature selector
    '''
    sel = FeatureSelector(dataset)

    ''' to use the wrapper method: '''
    sel.feature_elimination(output="../../demo/wrapper_scores.csv")

    ''' to use PBC, you have to specify whether to generate features for DNA or RNA '''
    sel.correlation(output="../../demo/pbc_dna_scores.csv", dna=True)
    sel.correlation(output="../../demo/pbc_rna_scores.csv", rna=True)

    '''
    Once you've generate scores, you can also use the feature selector to 
    pick out the features that match a criteria
    '''
    ''' To pick to top 100 features, run: '''
    features = sel.select_from_scores(fp="../../demo/wrapper_scores.csv", top_n=100)

    ''' To pick scores over 0.05, run: '''
    features = sel.select_from_scores(fp="../../demo/wrapper_scores.csv", threshold=0.05)

    ''' To combine selected features, do this: '''
    rna_features = sel.select_from_scores(fp="../../demo/pbc_rna_scores.csv", threshold=0.2)
    dna_features = sel.select_from_scores(fp="../../demo/pbc_dna_scores.csv", threshold=0.2)
    features = sel.combine_features([rna_features, dna_features])

    ''' And to set a dataset to only include selected features, run: '''
    dataset.select_features(features)

    '''
    Once you have your dataset setup how you want, you can create a model
    The model can be used for training, predicting, and validation

    First, create an sklearn classifier:
    '''
    clf = RandomForestClassifier()

    '''Then create a model'''
    model = ProteinModel(clf, dataset)

    '''Validate:'''
    model.validate()

    '''Train and predict'''
    model.train()

    '''Create testing dataset'''
    test_dataset = ProteinDataset(
        encoded_input_files=["../../data_test/encodings/aac.csv", "../../data_test/encodings/bio_data.csv"]
    )

    sel = FeatureSelector(test_dataset)
    rna_features = sel.select_from_scores(fp="../../demo/pbc_rna_scores.csv", threshold=0.2)
    dna_features = sel.select_from_scores(fp="../../demo/pbc_dna_scores.csv", threshold=0.2)
    features = sel.combine_features([rna_features, dna_features])
    test_dataset.select_features(features)

    predictions = model.predict(test_dataset)
    print(predictions)


if __name__ == "__main__":
    main()



# dataset = ProteinDataset(
#     labelled_input_file="../data/sequences_training.txt",
#     encoded_input_files=[
#         "../data/disorder.csv", 
#         "../data/entropy.csv", 
#         "../data/aac.csv", 
#         "../data/dpc.csv",
#         "../data/aaindex.csv",
#         "../data/bio_data.csv",
#         "../data/ctriad.csv"
#     ]
# )

# sel = FeatureSelector(dataset)
# rna_features = set(sel.select_from_scores(fp="../metrics/pbc_rna_scores.csv", threshold=0.2))
# dna_features = set(sel.select_from_scores(fp="../metrics/pbc_dna_scores.csv", threshold=0.2))
# features = np.array(list(dna_features.union(rna_features)))

# dataset.select_features(features)
# clf = RandomForestClassifier()
# model = ProteinModel(clf, dataset)
# model.validate()
