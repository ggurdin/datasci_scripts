from protein_dataset import ProteinDataset
from protein_model import ProteinModel
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from featurizer import Featurizer
from scipy.stats import pointbiserialr



def main():
    # create a dataset from labelled data

    dataset = ProteinDataset(
        labelled_input_file="../data/sequences_training.txt"
    )

    # Datasets are mainly used to store data, but can
    # be used to create features via iFeature
    # (should move this ability to featurizer.py in the future, should be more decoupled)

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
        encoded_input_files=["../data/disorder.csv", "../data/entropy.csv", "../data/aac.csv"],
    )

    # datasets store 3 important properites:
    # x -> the feature vectors
    # y -> the labels
    # features -> the names of the features

    # to encode datasets with featurizer.py, create a featurizer
    # then call the correspoding function, and specify an output file

    ft = Featurizer(dataset)
    ft.amino_acid_comp(output="../data/aac.csv")

    # some of the functions if featurizer.py are resource intensive
    # I ended up running those on Google Collab
    # they are: percent_disorder(), compute_aaindex(), and tripeptide_comp()

    # once you've generated your desired features, you can create a dataset
    # from multiple csv files as described above

    # from that dataset, you can generate a model
    # first create whichever classifier you want to use:
    clf = KNeighborsClassifier(3)

    # then create the model
    model = ProteinModel(clf, dataset)

    # then validate the model
    model.validate()

    # if you want to validate the model by training on DNA / not DNA
    # or RNA / not RNA, you can specify that in you call to validate:

    model.validate(dna_only=True)
    model.validate(rna_only=True)


if __name__ == "__main__":
    main()
