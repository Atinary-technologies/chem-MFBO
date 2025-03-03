"""Utils for running benchmarks."""
from copy import copy
from typing import Dict
import numpy as np
import pandas as pd
import torch
from botorch.models import AffineFidelityCostModel
from rdkit import Chem
from rdkit.Chem import MACCSkeys
from rdkit.Chem.Descriptors import CalcMolDescriptors
from sklearn.decomposition import PCA
from torch import Tensor


class FixedCostFids(AffineFidelityCostModel):
    """Class to model specific cost of the problem."""

    def __init__(self, fidelity_weights: Dict[int, float], fixed_cost=0, min_cost=0.5):
        super().__init__(fidelity_weights, fixed_cost)
        self.min_cost = min_cost

    def forward(self, X: Tensor):

        fids = X[..., -1]
        
        max_fid = max(fids).item() 

        final_cost = torch.where(fids != max_fid, self.min_cost, fids)

        final_cost = torch.where(fids == max_fid, 1.0, fids)

        #final_cost = torch.where(fids != 1.0, self.min_cost, fids)

        return final_cost


def encode_fps(smiles: list[str]) -> np.array:
    """Encode SMILES as fps"""

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]

    fps = np.array([MACCSkeys.GenMACCSKeys(mol) for mol in mols])

    pca = PCA(n_components=20)

    fps = pca.fit_transform(fps)

    fps = (fps - np.min(fps, axis=0)) / (np.max(fps, axis=0) - np.min(fps, axis=0))

    return fps


def encode_descriptors(smiles: list[str]) -> np.array:
    """Encode SMILES as descriptors"""

    mols = [Chem.MolFromSmiles(smile) for smile in smiles]

    desc = [CalcMolDescriptors(mol) for mol in mols]

    desc_df = pd.DataFrame(desc)

    desc_arr = desc_df.to_numpy()

    pca = PCA(n_components=10)

    fps = pca.fit_transform(desc_arr)

    fps = (fps - np.min(fps, axis=0)) / (np.max(fps, axis=0) - np.min(fps, axis=0))

    return fps


def diverse_set(X, seed_cof, train_size):
    """Initialization method that takes a candidate randomly and then samples the most
    diverse ones.
    """
    # initialize with one random point; pick others in a max diverse fashion
    nb_COFs = X.shape[0]
    ids_train = copy(seed_cof)
    # select remaining training points
    for j in range(train_size - 1):
        # for each point in data set, compute its min dist to training set
        dist_to_train_set = np.linalg.norm(X - X[ids_train, None, :], axis=2)
        assert np.shape(dist_to_train_set) == (len(ids_train), nb_COFs)
        min_dist_to_a_training_pt = np.min(dist_to_train_set, axis=0)
        assert np.size(min_dist_to_a_training_pt) == nb_COFs

        # acquire point with max(min distance to train set) i.e. Furthest from train set
        ids_train.append(np.argmax(min_dist_to_a_training_pt))
    assert np.size(np.unique(ids_train)) == train_size  # must be unique
    return np.array(ids_train)