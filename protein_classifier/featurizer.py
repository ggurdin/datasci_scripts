from collections import Counter
import numpy as np
import pandas as pd
import re
import metapredict as meta
import collections
from scipy.stats import entropy
from Bio.SeqUtils.ProtParam import ProteinAnalysis


class Featurizer:

    def __init__(self, dataset):
        self.sequences = dataset.x
        self.encodings = {}
        self.aas = [
            'A', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 
            'K', 'L', 'M', 'N', 'P', 'Q', 'R', 'S', 
            'T', 'U', 'V', 'W', 'X', 'Y', 'Z'
        ]

    def biopython_features(self, output="../data/bio_data.csv"):
        all_features = []
        for seq in self.sequences:
            features = {}
            prot = ProteinAnalysis(seq)
            try:
                features["instability_index"] = prot.instability_index()
                features["gravy"] = prot.gravy()
                ss_fractions = prot.secondary_structure_fraction()
                features["helix"] = ss_fractions[0]
                features["turn"] = ss_fractions[1]
                features["sheet"] = ss_fractions[2]
                features["charge"] = prot.charge_at_pH(7)
            except KeyError:
                features["instability_index"] = 0
                features["gravy"] = 0
                features["helix"] = 0
                features["turn"] = 0
                features["sheet"] = 0
                features["charge"] = 0
            all_features.append(features)
        df = pd.DataFrame.from_dict(all_features)
        df.to_csv(output, index=False)

    def percent_disorder(self, output="../data/disorder.csv"):
        disorders = []
        for i, seq in enumerate(self.sequences):
            if i % 100 == 0:
                print(i)
            try:
                disorders.append(meta.percent_disorder(seq))
            except ValueError:
                disorders.append(0)
        df = pd.DataFrame(disorders, columns="percent_disoder")
        df.to_csv(output, index=False)


    # from https://onestopdataanalysis.com/shannon-entropy/
    def estimate_shannon_entropy(self, sequence):
        bases = collections.Counter([tmp_base for tmp_base in sequence])
        dist = [x/sum(bases.values()) for x in bases.values()]
        entropy_value = entropy(dist, base=2)
        return entropy_value

    def shannon_entropy(self, output="../data/entropy.csv"):
        entropies = []
        for seq in self.sequences:
            entropies.append(self.estimate_shannon_entropy(seq))
        df = pd.DataFrame(entropies, columns=["shannon_entropy"])
        df.to_csv("../data/entropy.csv", index=False)
            

    def compute_aaindex(self, parser, output="../data/aaindex.csv"):
        aaindex = parser.aaindex
        vectors = []

        for i, seq in enumerate(self.sequences):
            if i % 100 == 0:
                print(i)
            seq_len = len(seq)
            vector = {}
            for aaindex_code, aaindex_info in aaindex.items():
                aaindex_vals = aaindex_info["vals"]
                aaindex_sum = 0
                for aa in seq:
                    if aa in aaindex_vals:
                        aaindex_sum += aaindex_vals[aa]
                aaindex_avg = aaindex_sum / seq_len
                vector[aaindex_code] = aaindex_avg
            vectors.append(vector)
        df = pd.DataFrame.from_dict(vectors)
        df.to_csv(output, index=False)


    def substring_count(self, sequence, peptide):
        return len(re.findall(rf"{peptide}", sequence)) / len(sequence)

    def amino_acid_comp(self, output="../data/aac.csv"):
        aacs = []
        for seq in self.sequences:
            seq_length = len(seq)
            counts = dict(Counter(seq))
            aacs.append({k: (v / seq_length) for k, v in counts.items()})
        aacs = pd.DataFrame.from_dict(aacs)[self.aas]
        aacs = aacs.fillna(0).to_numpy()
        df = pd.DataFrame(aacs, columns=self.aas)
        df.to_csv(output, index=False)

    def dipeptide_comp(self, output="../data/dpc.csv"):
        dps = []
        for aa1 in self.aas:
            for aa2 in self.aas:
                dps.append(aa1 + aa2)
        dps.sort()
        self.dps = dps

        dpc_encodings = []
        seqs = pd.Series(self.sequences)
        for dp in dps:
            dp = seqs.apply(self.substring_count, args=[dp]).to_numpy()
            dpc_encodings.append(dp)
        dpc_encodings = np.array(dpc_encodings)
        dpc_encodings = dpc_encodings.transpose()
        df = pd.DataFrame(dpc_encodings, columns=dps)
        df.to_csv(output, index=False)

    def tripeptide_comp(self, output="../data/tpc.csv"):
        tps = []
        for aa1 in self.aas:
            for aa2 in self.aas:
                for aa3 in self.aas:
                    tps.append(aa1 + aa2 + aa3)
        tps.sort()
        self.tps = tps

        tpc_encodings = []
        seqs = pd.Series(self.sequences)
        num_tps = len(tps)
        for i, tp in enumerate(tps):
            tp = seqs.apply(self.substring_count, args=[tp]).to_numpy()
            tpc_encodings.append(tp)
            if i % 100 == 0 and i != 0:
                print(f"{i} / {num_tps}")
        tpc_encodings = np.array(tpc_encodings)
        tpc_encodings = tpc_encodings.transpose()
        df = pd.DataFrame(tpc_encodings, columns=tps)
        df.to_csv(output, index=False)
