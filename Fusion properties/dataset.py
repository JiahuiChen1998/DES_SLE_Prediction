from __future__ import print_function, division
import functools
import  numpy  as  np
import torch
from torch.utils.data import Dataset
from rdkit import Chem
from torch_geometric.data import Data
from descriptor import atom_features,bond_features,etype_features
import numpy as np
import torch
from rdkit import Chem


class SMILES_dataset(Dataset):
    'Characterizes a dataset for PyTorch'

    def __init__(self, df, tokenizer, target):
        self.target = target
        self.smiles = df['SMILES']
        self.tokens = np.array(
            [tokenizer.encode(i, max_length=128, truncation=True, padding='max_length') for i in self.smiles])
        self.label = df[target]
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.label)

    @functools.lru_cache(maxsize=None)
    def __getitem__(self, index):
        X = torch.from_numpy(np.asarray(self.tokens[index]).astype(np.float32))
        y = torch.from_numpy(np.asarray(self.label[index])).view(-1, 1)

        return (X,y), y.float()
        


class Graph_dataset(torch.utils.data.Dataset):
    """
    torch dataset
    """
    def __init__(self, df,target):
        super(Graph_dataset, self).__init__()
        self.smiles = df['SMILES']
        self.label = df[target]
        self.index = df['index']
        self.length = len(df)
        self.target=target
        self.df=df

        print("----info----")
        print("data_length", self.length)
        print("------------")

    def __len__(self):
        return self.length

    def __getitem__(self, idx):

        row = self.df.iloc[idx]
        smiles = row['SMILES']

        index=self.index

        combine_graph = self.smiles2graph(smiles)

        label = torch.tensor(row[self.target], dtype=torch.float).view(1, -1)

        data = Data(x=combine_graph.x, edge_index=combine_graph.edge_index, edge_attr=combine_graph.edge_attr, y=label,index=index)

        return data,label


    def smiles2graph(self,smiles):
        try:
            mol = Chem.MolFromSmiles(smiles)
            if mol is None:
                raise ValueError(f'Failed to generate mol object from SMILES: {smiles}')
        except Exception as e:
            raise ValueError(f'Invalid SMILES: {str(e)}')
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


        data = Data(x=x, edge_index=edge_index, edge_attr=edge_attr)
        return data

    def collate_fn(self,batch):
        graphs = [data[0] for data in batch]


        batch = torch.cat([data.batch for data in graphs], dim=0)

        return batch
