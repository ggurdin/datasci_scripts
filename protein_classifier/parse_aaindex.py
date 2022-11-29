import re


class Parser:

    def __init__(self, path):
        self.path = path
        self.aaindex = self.parse()

    def parse(self):
        with open(self.path, "r") as fp:
            aaindex_txt = fp.read()
        aaindex_lst = [[y for y in x.split("\n") if y != ""] for x in aaindex_txt.split("//")][:-1]

        aaindex_measures = {}
        for measure in aaindex_lst:
            measure_code = measure[0][2:]
            measure_name = measure[1][2:]

            for i, measure_item in enumerate(measure):
                if measure_item.startswith("I "):
                    amino_acids = re.split(r"\s+|/", measure_item)[1:]
                    amino_acids_one = [x for i, x in enumerate(amino_acids) if i % 2 == 0]
                    amino_acids_two = [x for i, x in enumerate(amino_acids) if i % 2 == 1]
                    vals_one = [float(x) for x in re.split(r"\s+", measure[i + 1].strip().replace("NA", "0"))]
                    vals_two = [float(x) for x in re.split(r"\s+", measure[i + 2].strip().replace("NA", "0"))]
                    map_one = {aa: val for aa, val in zip(amino_acids_one, vals_one)}
                    map_two = {aa: val for aa, val in zip(amino_acids_two, vals_two)}
                    measure_vals = {**map_one, **map_two}
                    aaindex_measures[measure_code] = {"name": measure_name, "vals": measure_vals}
                    break
        return aaindex_measures
                    
                