import numpy as np
import torch
from torch.utils.data import Dataset
def get_fold_data(i, datasets, k=5):
    
    fold_size = len(datasets) // k  

    val_start = i * fold_size
    if i != k - 1 and i != 0:
        val_end = (i + 1) * fold_size
        valid = datasets[val_start:val_end]
        train = datasets[0:val_start] + datasets[val_end:]
    elif i == 0:
        val_end = fold_size
        valid = datasets[val_start:val_end]
        train = datasets[val_end:]
    else:
        valid = datasets[val_start:]
        train = datasets[0:val_start]

    return train, valid

def shuffle_dataset(dataset, seed):
    np.random.seed(seed)
    np.random.shuffle(dataset)
    return dataset

CHARISOSMISET = {"#": 29, "%": 30, ")": 31, "(": 1, "+": 32, "-": 33, "/": 34, ".": 2,
                 "1": 35, "0": 3, "3": 36, "2": 4, "5": 37, "4": 5, "7": 38, "6": 6,
                 "9": 39, "8": 7, "=": 40, "A": 41, "@": 8, "C": 42, "B": 9, "E": 43,
                 "D": 10, "G": 44, "F": 11, "I": 45, "H": 12, "K": 46, "M": 47, "L": 13,
                 "O": 48, "N": 14, "P": 15, "S": 49, "R": 16, "U": 50, "T": 17, "W": 51,
                 "V": 18, "Y": 52, "[": 53, "Z": 19, "]": 54, "\\": 20, "a": 55, "c": 56,
                 "b": 21, "e": 57, "d": 22, "g": 58, "f": 23, "i": 59, "h": 24, "m": 60,
                 "l": 25, "o": 61, "n": 26, "s": 62, "r": 27, "u": 63, "t": 28, "y": 64}

CHARISOSMILEN = 64

CHARPROTSET = {"A": 1, "C": 2, "B": 3, "E": 4, "D": 5, "G": 6,
               "F": 7, "I": 8, "H": 9, "K": 10, "M": 11, "L": 12,
               "O": 13, "N": 14, "Q": 15, "P": 16, "S": 17, "R": 18,
               "U": 19, "T": 20, "W": 21, "V": 22, "Y": 23, "X": 24, "Z": 25}

CHARPROTLEN = 25


def form_digit(string_list, max_long, the_dict):

    vector_list = []
    for string in string_list:
        vector = np.zeros(max_long, np.int64())
        for i, ch in enumerate(string[:max_long]):
            vector[i] = the_dict[ch]
        vector_list.append(vector)

    return vector_list

class CustomDataSet(Dataset):
    def __init__(self, pairs):
        self.pairs = pairs

    def __getitem__(self, item):
        return self.pairs[item]

    def __len__(self):
        return len(self.pairs)

class CustomCollateFn:
    """先把所有的蛋白质字符和药物字符都转换成数字向量后构成Protein_list和Drug_list"""
    def __init__(self, Drug_list, Protein_list, drug_simdict=None, protein_simdict=None,generate_batchsim=False):
        """
        contrastive_sim_dict:表示用于对比学习的相似性字典
        """
        self.Drug_list = Drug_list
        self.Protein_list = Protein_list
        self.drug_simdict =  drug_simdict
        self.protein_simdict = protein_simdict
        self.generate_batchsim = generate_batchsim
    def __call__(self, batch_data):

        batch_data = np.array(batch_data)
        drug_new = torch.tensor(self.Drug_list[batch_data[:,0]])
        protein_new = torch.tensor(self.Protein_list[batch_data[:,1]])
        labels_new = torch.tensor(batch_data[:,2])
        if self.generate_batchsim:
            batch_drugsim_dict = {}
            batch_proteinsim_dict = {}
            for key in self.protein_simdict.keys():
                batch_proteinsim_dict[key] = torch.tensor(
                    self.protein_simdict[key][batch_data[:, 1]][:, batch_data[:, 1]])
            for key in self.drug_simdict.keys():
                batch_drugsim_dict[key] = torch.tensor(self.drug_simdict[key][batch_data[:, 0]][:, batch_data[:, 0]])
            return drug_new, protein_new, labels_new, batch_drugsim_dict, batch_proteinsim_dict

        else:
            return drug_new, protein_new, labels_new




