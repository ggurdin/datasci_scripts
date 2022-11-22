from protein_dataset import ProteinDataset
from protein_model import ProteinModel
from sklearn.svm import SVC


def main():
    dataset = ProteinDataset(
        "AAC",
        labelled_input_file="../data/sequences_training.txt",
        encoded_input_file="../data/aac.tsv"
    )
    clf = SVC(kernel="poly")
    model = ProteinModel(clf, dataset)
    model.validate()


if __name__ == "__main__":
    main()
