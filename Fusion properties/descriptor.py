
import numpy as np
import torch
from rdkit import Chem
import csv
from torch_geometric.data import Data, Dataset


def one_of_k_encoding(x, allowable_set):
    if x not in allowable_set:
        raise Exception("input {0} not in allowable set{1}:".format(
            x, allowable_set))
    return [x == s for s in allowable_set]


def one_of_k_encoding_unk(value, options, unk_token=None):
    """Return a one-hot encoding for an element with a fallback for unknown values."""
    if value not in options:
        value = unk_token
    return [int(value == v) for v in options]


def one_of_k_atompair_encoding(x, allowable_set):
    for atompair in allowable_set:
        if x in atompair:
            x = atompair
            break
        else:
            if atompair == allowable_set[-1]:
                x = allowable_set[-1]
            else:
                continue
    return [x == s for s in allowable_set]


def atom_features(atom, explicit_H=False, use_chirality=True):
    results = one_of_k_encoding_unk(atom.GetSymbol(),['B','C','N','O','F','Si','P','S','Cl','As','Se','Br','Te','I','At','other'])\
              + one_of_k_encoding(atom.GetDegree(),[0, 1, 2, 3, 4, 5, 6]) + \
              [atom.GetFormalCharge(), atom.GetNumRadicalElectrons()] + \
              one_of_k_encoding_unk(atom.GetHybridization(), [
                  Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2,
                  Chem.rdchem.HybridizationType.SP3, Chem.rdchem.HybridizationType.SP3D,
                  Chem.rdchem.HybridizationType.SP3D2, 'other']) + [atom.GetIsAromatic()]
    if not explicit_H:
        results = results + one_of_k_encoding_unk(atom.GetTotalNumHs(),[0, 1, 2, 3, 4])
    if use_chirality:
        try:
            results = results + one_of_k_encoding_unk(atom.GetProp('_CIPCode'),['R', 'S']) + \
                      [atom.HasProp('_ChiralityPossible')]
        except:
            results = results + [False, False] + [atom.HasProp('_ChiralityPossible')]

    return np.array(results)


def bond_features(bond, use_chirality=True, atompair=False):
    bt = bond.GetBondType()
    bond_feats = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
        bond.GetIsConjugated(),
        bond.IsInRing()
    ]
    if use_chirality:
        bond_feats = bond_feats + one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
    if atompair:
        atom_pair_str = bond.GetBeginAtom().GetSymbol() + bond.GetEndAtom().GetSymbol()
        bond_feats = bond_feats + one_of_k_atompair_encoding(
            atom_pair_str, [['CC'], ['CN', 'NC'], ['ON', 'NO'], ['CO', 'OC'], ['CS', 'SC'],
                            ['SO', 'OS'], ['NN'], ['SN', 'NS'], ['CCl', 'ClC'], ['CF', 'FC'],
                            ['CBr', 'BrC'], ['others']]
        )
    return np.array(bond_feats).astype(float)


def etype_features(bond, use_chirality=True):
    bt = bond.GetBondType()
    bond_feats_1 = [
        bt == Chem.rdchem.BondType.SINGLE, bt == Chem.rdchem.BondType.DOUBLE,
        bt == Chem.rdchem.BondType.TRIPLE, bt == Chem.rdchem.BondType.AROMATIC,
    ]
    for i, m in enumerate(bond_feats_1):
        if m == True:
            a = i

    bond_feats_2 = bond.GetIsConjugated()
    if bond_feats_2 == True:
        b = 1
    else:
        b = 0

    bond_feats_3 = bond.IsInRing
    if bond_feats_3 == True:
        c = 1
    else:
        c = 0

    index = a * 1 + b * 4 + c * 8
    if use_chirality:
        bond_feats_4 = one_of_k_encoding_unk(
            str(bond.GetStereo()),
            ["STEREONONE", "STEREOANY", "STEREOZ", "STEREOE"])
        for i, m in enumerate(bond_feats_4):
            if m == True:
                d = i
        index = index + d * 16
    return index


def graph_mol2vec(mol, y=None):
    # mol = Chem.MolFromSmiles(smiles)
    nodes = []
    edges = []
    edge_attrs = []

    for atom in mol.GetAtoms():
        node_feat = atom_features(atom)
        nodes.append(node_feat)

    for bond in mol.GetBonds():
        bond_feat = bond_features(bond)
        etype_feat = etype_features(bond)
        edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
        edge_attrs.append(etype_feat + bond_feat)

    x = np.array(nodes, dtype=np.float32)
    edge_index = np.array(edges, dtype=np.int64).T
    edge_attr = np.array(edge_attrs, dtype=np.float32)

    x = torch.from_numpy(x)
    edge_index = torch.from_numpy(edge_index)
    edge_attr = torch.from_numpy(edge_attr)

    data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor(y))
    data.y = torch.unsqueeze(data.y, dim=0)
    data.y = torch.unsqueeze(data.y, dim=0)
    return data


def read_smiles(data_path, target):
    smiles_data, labels = [], []
    with open(data_path) as csv_file:
        csv_reader = csv.DictReader(csv_file, delimiter=',')

        for i, row in enumerate(csv_reader, start=1):
            smiles = row['smiles']
            label = row[target]
            mol = Chem.MolFromSmiles(smiles)
            if mol is not None and label != '':
                smiles_data.append(smiles)
                labels.append(float(label))

    print(len(smiles_data))
    return smiles_data, labels


class MolTestDataset(Dataset):
    def __init__(self, data_path, target):
        super().__init__()
        self.smiles_data, self.labels = read_smiles(data_path, target)

    def __getitem__(self, index):
        mol = Chem.MolFromSmiles(self.smiles_data[index])

        nodes = []
        edges = []
        edge_attrs = []

        for atom in mol.GetAtoms():
            node_feat = atom_features(atom)
            nodes.append(node_feat)
            
        for bond in mol.GetBonds():
            bond_feat = bond_features(bond)
            etype_feat = etype_features(bond)
            edges.append([bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()])
            edge_attrs.append(etype_feat + bond_feat)
        
        x = torch.tensor(np.vstack(nodes), dtype=torch.float)
        edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        edge_attr = torch.tensor(np.vstack(edge_attrs), dtype=torch.float)
        y = torch.tensor(self.labels[index], dtype=torch.float).view(1,-1)
        smiles = self.smiles_data[index]  
        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=y, smiles=smiles)  
        return data


    def __len__(self):
        return len(self.smiles_data)


# smiles='OC(O)=C1CCC(=C(O)O)[Se][Se]1'
# mol = Chem.MolFromSmiles(smiles)
# print(graph_mol2vec(mol,y=0))


