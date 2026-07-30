import torch
import torch_geometric.transforms as T
from .chemprop_featurisation import (
    atom_features,
    atom_features_int,
    bond_features,
    bond_features_int,
    get_atom_constants,
)
from rdkit import Chem


def add_chemprop_features(data, one_hot, max_atomic_number):
    atom_constants = get_atom_constants(max_atomic_number)
    mol = Chem.MolFromSmiles(data.smiles)
    if mol is None:
        return None
    mol = Chem.AddHs(mol)

    ei = torch.nonzero(torch.from_numpy(Chem.rdmolops.GetAdjacencyMatrix(mol))).T
    if one_hot:
        atom_feat = torch.tensor(
            [atom_features(atom, atom_constants) for atom in mol.GetAtoms()],
        )

        bond_feat = torch.tensor(
            [bond_features(mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item())) for i in range(ei.shape[1])],
        )
    else:
        atom_feat = torch.tensor(
            [atom_features_int(atom, atom_constants) for atom in mol.GetAtoms()],
        )

        bond_feat = torch.tensor(
            [bond_features_int(mol.GetBondBetweenAtoms(ei[0][i].item(), ei[1][i].item())) for i in range(ei.shape[1])],
        )

    # ei, bond_feat = to_undirected(ei, edge_attr=bond_feat)

    data.x = atom_feat
    data.edge_index = ei
    data.edge_attr = bond_feat

    return data



class ChempropFeatures(T.BaseTransform):
    def __init__(self, one_hot, max_atomic_number):
        self.one_hot = one_hot
        self.max_atomic_number = max_atomic_number

    def forward(self, data):
        data = add_chemprop_features(data, self.one_hot, self.max_atomic_number)

        return data


class AddNumNodes(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            data.num_nodes = data.x.shape[0]
        return data


class AddMaxEdge(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            if data.edge_index.numel() > 0:
                data.max_edge = torch.tensor(data.edge_index.shape[-1]).unsqueeze(0)
            else:
                return None

        return data


class AddMaxNode(T.BaseTransform):
    def forward(self, data):
        if data is not None:
            data.max_node = torch.tensor(data.num_nodes).unsqueeze(0)

        return data
    

class AddMaxEdgeGlobal(T.BaseTransform):
    def __init__(self, max_edge: int):
        self.max_edge = max_edge

    def forward(self, data):
        data.max_edge_global = self.max_edge

        return data


class AddMaxNodeGlobal(T.BaseTransform):
    def __init__(self, max_node: int):
        self.max_node = max_node

    def forward(self, data):
        data.max_node_global = self.max_node

        return data
