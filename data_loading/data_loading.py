from .morgan_feature_extractor_classification import compute_fingerprints_batch  # Import Morgan fingerprint computation function
import numpy as np
import pandas as pd
import os
import torch
import torch_geometric
from torch_geometric.utils import degree
from torch_geometric.data import Data, InMemoryDataset
from torch.utils.data import random_split
from sklearn.preprocessing import StandardScaler
from tqdm.auto import tqdm

from ogb.utils import smiles2graph
from data_loading.transforms import *

from rdkit import RDLogger
from sklearn.model_selection import train_test_split
from preprocessing.extract_embeddings import (
    DEFAULT_HPARAMS,
    DEFAULT_MOLFORMER_ENV,
    ensure_embeddings,
)
from data_loading.smiles_canonical import canonicalize_smiles_for_data

RDLogger.DisableLog("rdApp.*")


def _csv_to_pyg_data(df, y_values, desc="Converting to Data list"):
    """CSV → PyG Data list; SMILES are canonicalized once here (train & infer)."""
    dataset_as_data_list = []
    smiles_list = df["Smiles"].values
    smiles_id_list = df["FeatureIndex"].values
    invalid_count = 0

    for i in tqdm(range(len(smiles_list)), desc=desc):
        canon_s = canonicalize_smiles_for_data(smiles_list[i])
        if canon_s is None:
            invalid_count += 1
            continue
        dataset_as_data_list.append(
            Data(
                smiles_id=smiles_id_list[i],
                smiles=canon_s,
                y=y_values[i],
            )
        )

    if invalid_count:
        print(f"SMILES canonicalize invalid/skipped samples: {invalid_count}")
    return dataset_as_data_list


def _get_emb_device(kwargs):
    """MolFormer embedding always uses cuda:0 by default (notebook behavior).

    Training --gpu-devices controls GNN only; do not route embedding to the
    training GPU or results diverge from frozen_embeddings_classification_lly.ipynb.
    """
    if kwargs.get("emb_device"):
        return kwargs["emb_device"]
    if kwargs.get("molformer_device"):
        return kwargs["molformer_device"]
    if torch.cuda.is_available():
        return "cuda:0"
    return "cpu"


def _maybe_ensure_embeddings(dataset_dir, datasets=None, **kwargs):
    molformer_ckpt_path = kwargs.get("molformer_ckpt_path")
    if not molformer_ckpt_path:
        return

    ensure_embeddings(
        data_dir=dataset_dir,
        ckpt_path=molformer_ckpt_path,
        hparams_path=kwargs.get("molformer_hparams_path") or DEFAULT_HPARAMS,
        molformer_env=kwargs.get("molformer_env") or DEFAULT_MOLFORMER_ENV,
        device=_get_emb_device(kwargs),
        batch_size=kwargs.get("emb_batch_size", 32),
        datasets=datasets,
        skip_if_exists=kwargs.get("emb_skip_if_exists", True),
    )


class CustomPyGDataset(InMemoryDataset):
    def __init__(self, data_list=None):
        super(CustomPyGDataset, self).__init__(".")
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)

    def _download(self):
        pass

    def _process(self):
        pass


class CustomPyGDatasetNodeMasks(InMemoryDataset):
    def __init__(self, data_list, train_mask, val_mask, test_mask):
        super(CustomPyGDatasetNodeMasks, self).__init__(".")
        if data_list is not None:
            self.data, self.slices = self.collate(data_list)
            self.train_mask = train_mask
            self.val_mask = val_mask
            self.test_mask = test_mask

    def _download(self):
        pass

    def _process(self):
        pass


def apply_scaler(data, scaler, convert_to_numpy=True, num_tasks=1):
    if convert_to_numpy:
        data.y = torch.tensor(scaler.transform(data.y.reshape(1, num_tasks).numpy()))
    else:
        data.y = torch.tensor(scaler.transform(data.y.reshape(1, num_tasks)))
    return data

# Binary classification label processing (replaces regression scaling)
def process_binary_labels(data_list):
    """
    Convert binary classification labels to PyTorch tensors.
    Args:
        data_list: List of data samples (each with a "y" attribute, e.g. PyG Data objects)
    Returns:
        Processed data_list with tensor y for each sample
    """
    # Check if input list is empty
    if not data_list:
        return None
    
    processed = []
    for data in data_list:
        # 1. Ensure labels are numeric (strip possible strings, etc.)
        # Assume y is a scalar (e.g. 0 or 1); if list/array, take first element
        y = data.y if not isinstance(data.y, (list, np.ndarray)) else data.y[0]
        
        # 2. Convert to PyTorch tensor
        # Note: BCEWithLogitsLoss expects float labels for binary classification
        y_tensor = torch.tensor(y, dtype=torch.float32)
        # y_tensor = torch.tensor(y, dtype=torch.float32, device=data.x.device if hasattr(data, 'x') else None)
        
        # 3. Ensure correct label shape (optional batch dimension)
        # If scalar, unsqueeze to shape (1,) per model requirements
        if y_tensor.ndim == 0:
            y_tensor = y_tensor.unsqueeze(0)  # scalar -> (1,)
        
        data.y = y_tensor  # replace original y with processed tensor
        processed.append(data)
    print("processed[0].y:", processed[0].y)
    return processed

# Function to convert Subset to PyG Dataset
def get_max_node_edge_global(dataset):
    max_node_global = 0
    max_edge_global = 0

    for data in tqdm(dataset):
        # Skip None values
        if data is None:
            # print("None:", data)
            continue
        if data.max_edge is None:
            # print("None:", data)
            continue
            
        # Update global max if max_edge attribute exists
        if hasattr(data, 'max_edge') and data.max_edge is not None:
            if data.max_edge > max_edge_global:
                max_edge_global = data.max_edge
        
        # Update global max if max_node attribute exists
        if hasattr(data, 'max_node') and data.max_node is not None:
            if data.max_node > max_node_global:
                max_node_global = data.max_node

    return max_edge_global, max_node_global


# --------------------------
# Helper: attach fingerprints to Data objects by SMILES
# --------------------------
def _add_fingerprint_to_data(data_list, fp_params):
    """
    Compute and attach fingerprints to a Data list (matched by SMILES).
    Args:
        data_list (list): List of Data objects with smiles attribute
        fp_params (dict): Fingerprint computation parameters
    Returns:
        list: Data objects with fingerprint attribute (invalid SMILES removed)
    """
    # 1. Extract SMILES from Data list
    smiles_list = [data.smiles for data in data_list]
    print(f"Samples to process (extract {fp_params['fingerprint_type']} fingerprints): {len(smiles_list)}")

    # 2. Compute fingerprints via Morgan_cal
    valid_smiles, valid_fps, invalid_smiles = compute_fingerprints_batch(
        smiles_list=smiles_list,** fp_params
    )
    print(f"{fp_params['fingerprint_type']} valid samples: {len(valid_smiles)}, {fp_params['fingerprint_type']} invalid samples: {len(invalid_smiles)}")
    if len(invalid_smiles) > 0:
        print(f"{fp_params['fingerprint_type']} invalid SMILES examples: {invalid_smiles[:5]}")  # first 5 invalid SMILES (debug)

    # 3. Build SMILES -> fingerprint map for fast lookup
    smile_to_fp = dict(zip(valid_smiles, valid_fps))

    # 4. Attach fingerprints to Data objects (drop invalid samples)
    data_with_fp = []
    for data in data_list:
        if data.smiles in smile_to_fp:
            # Attach fingerprint as torch.Tensor for the model
            data.fingerprint = torch.tensor(smile_to_fp[data.smiles], dtype=torch.float32)
            data_with_fp.append(data)

    # Verify matching
    assert len(data_with_fp) == len(valid_smiles), "Fingerprint/Data count mismatch"
    return data_with_fp


def add_embedding_from_pt(train_data, emb_pt_path):
    """
    Load FeatureIndex-embedding map from .pt and match onto train_data by FeatureIndex.
    
    Args:
        train_data: list[Data] - training Data list (needs feature_index / smiles_id)
        emb_pt_path: str - path to .pt with fi_to_embedding, etc.
    
    Returns:
        list[Data] - train_data with smiles_embedding (samples without match filtered out)
    """
    # -------------------------- 1. Load FeatureIndex-embedding map from .pt --------------------------
    print(f"\n=== Loading FeatureIndex-embedding map from {emb_pt_path} ===")
    try:
        # Load full payload (fi_to_embedding, etc.)
        emb_data = torch.load(emb_pt_path)
        # Core map: FeatureIndex -> embedding vector
        fi_to_embedding = emb_data["fi_to_embedding"]
        # Optional: FeatureIndex -> SMILES for debugging unmatched samples
        fi_to_smile = emb_data["fi_to_smile"]
    except KeyError as e:
        raise ValueError(f".pt file missing required mapping key: {str(e)} (expected post-FeatureIndex format)") from e
    except Exception as e:
        raise ValueError(f"Failed to load .pt file: {str(e)}") from e

    print(f"Loaded {len(fi_to_embedding)} FeatureIndex-embedding mappings")

    # -------------------------- 2. Match by FeatureIndex and attach embeddings --------------------------
    train_data_with_emb = []
    no_match_fi = []  # FeatureIndex with no embedding match (debug)

    for data in train_data:
        # Match embedding by smiles_id (more reliable than SMILES alone)
        if data.smiles_id in fi_to_embedding:
            # Attach embedding to Data object
            data.smiles_embedding = fi_to_embedding[data.smiles_id]
            train_data_with_emb.append(data)
        else:
            no_match_fi.append(data.smiles_id)

    # -------------------------- 3. Log match results (debug) --------------------------
    print(f"train_data original sample count: {len(train_data)}")
    print(f"Samples with embedding attached: {len(train_data_with_emb)}")
    
    if len(no_match_fi) > 0:
        # Log unmatched FeatureIndex and SMILES for debugging
        no_match_details = [
            (fi, data.smiles[:30] + "...")  # truncate SMILES to 30 chars
            for fi, data in zip(no_match_fi, train_data)
            if data.smiles_id == fi
        ]
        print(f"Samples without embedding match: {len(no_match_fi)}, examples: {no_match_details[:5]}")

    return train_data_with_emb
            
            
def load_del_chemprop(dataset, dataset_dir, one_hot, target_name, **kwargs):
    # --------------------------
    # Fingerprint params from kwargs (defaults adjustable)
    fp_params = {
        "fingerprint_type": kwargs.get("fingerprint_type", "morgan2048"),  # default Morgan fingerprint
        "radius": kwargs.get("morgan_radius", 2),                     # Morgan radius
        "nBits": kwargs.get("morgan_nBits", 2048),                   # Morgan bit count
        "useChirality": kwargs.get("useChirality", False),           # include chirality
        "num_processes": kwargs.get("num_processes", 4),             # parallel workers
        "chunk_size": kwargs.get("fp_chunk_size", 1000)              # chunk size
    }
    print(f"\nFingerprint params: {fp_params}")
    # --------------------------

    if "infer" in kwargs.keys():
        infer_file_name = kwargs["infer_file_name"]
        preprocessed_path_test = os.path.join(dataset_dir, f"{infer_file_name}_{fp_params['fingerprint_type']}_emb.pt")

        if os.path.isfile(preprocessed_path_test):   # check file exists
            print("Loading pre-processed infer data...")
            test = torch.load(preprocessed_path_test)
            print("Loaded pre-processed splits!")
        else:
            print("Pre-processed data unavailable, computing...")
            test = pd.read_csv(os.path.join(dataset_dir, f"{infer_file_name}.csv"))
            print(f"Inference dataset size: {len(test)}")

            print("Loading data splits...")
            test = _csv_to_pyg_data(test, np.ones(len(test), dtype=np.int64))

            print("\nDataset items look like: ", test[0])

            # --------------------------
            # Compute Morgan fingerprints and attach to Data objects
            # print("\n=== Computing fingerprints for train set ===")
            # train_data = _add_fingerprint_to_data(train_data, fp_params)
            # print("\n=== Computing fingerprints for val set ===")
            # val_data = _add_fingerprint_to_data(val_data, fp_params)
            print("\n=== Computing fingerprints for test set ===")
            test = _add_fingerprint_to_data(test, fp_params)
            # print("train_data[0]:", train_data[0])
            # --------------------------
            
            print("\n=== Loading embeddings for test set ===")
            _maybe_ensure_embeddings(
                dataset_dir,
                datasets=[(f"{infer_file_name}.csv", f"{infer_file_name}_emb.pt")],
                **kwargs,
            )
            preprocessed_path_test_emb = os.path.join(dataset_dir, f"{infer_file_name}_emb.pt")
            test = add_embedding_from_pt(test, emb_pt_path=preprocessed_path_test_emb)
            print("\n=== Embedding attachment verification ===")
            print("test[0] (embedding)：", test[0])  #train_data[0] (embedding)： Data(y=1, smiles='CC(=O)NC1CCN(C(=O)C(C)Cc2cccc(-c3noc(/C=C/c4ccc(C#N)cc4)n3)c2)CC1', feature_index='HGODEL0034-190-54-26', fingerprint=[2048], smiles_embedding=[768])
            print("test[0] embedding shape:", test[0].smiles_embedding.shape)  # torch.Size([768])

            
            transforms = [ChempropFeatures(one_hot=one_hot, max_atomic_number=53), AddNumNodes(), AddMaxEdge(), AddMaxNode()]  # max_atomic_number=53 (iodine)
            transforms = T.Compose(transforms)

            print("Computing ChemProp features for data splits...")
            test = [transforms(data) for data in tqdm(test) if data is not None]

            print("Caching pre-processed files...")
            torch.save(test, preprocessed_path_test)
            print("Caching done")

        print("Determining global node/edge counts...")
        max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)
        max_edge_global = max(max_edge_global_test)
        max_node_global = max(max_node_global_test)

        print(f"max_node_global = {max_node_global}")
        print(f"max_edge_global = {max_edge_global}")
        print(f"dataset size = {len(test)}")

        global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

        print("Applying global node/edge count transforms...")
        print("FDA count before filtering:", len(test))
        test = [data for data in test if data is not None]  # filter None entries
        print("FDA count after filtering:", len(test))
        test = [global_transforms(data) for data in tqdm(test)]
        print(f"Datasets has {len(test)} test elements")

        print("test[0]:", test[0]) # train[0]: Data(y=[1, 1], smiles='CCc1ncsc1NC(=O)c1ccc(CN[C@H](CC(=O)NC)Cc2ccc3ccccc3c2)c(F)c1', x=[65, 79], edge_index=[2, 136], edge_attr=[136, 13], num_nodes=65, max_edge=[1], max_node=[1], max_edge_global=[1], max_node_global=[1])

        test = process_binary_labels(test)    # process test labels

        print("test[0]:", test[0])  # train[0]: Data(y=[1, 1], smiles='CCc1ncsc1NC(=O)c1ccc(CN[C@H](CC(=O)NC)Cc2ccc3ccccc3c2)c(F)c1', x=[65, 79], edge_index=[2, 136], edge_attr=[136, 13], max_edge=[1], max_node=[1], max_edge_global=[1], max_node_global=[1], num_nodes=65)

        print("Finished loading data!")

        num_classes = 1
        task_type = "binary_classification"

        return test, num_classes, task_type, None
    
    else:
        preprocessed_path_train = os.path.join(dataset_dir, f"DEL_train_{target_name}_{fp_params['fingerprint_type']}_emb.pt")
        preprocessed_path_val = os.path.join(dataset_dir, f"DEL_val_{target_name}_{fp_params['fingerprint_type']}_emb.pt")
        preprocessed_path_test = os.path.join(dataset_dir, f"DEL_test_{target_name}_{fp_params['fingerprint_type']}_emb.pt")

                
        if os.path.isfile(preprocessed_path_train):   # check file exists
            print("Loading pre-processed train, val, test...")
            train = torch.load(preprocessed_path_train)
            val = torch.load(preprocessed_path_val)
            test = torch.load(preprocessed_path_test)
            print("Loaded pre-processed splits!")
        else:
            # 1. Read CSV and build base Data objects (smiles, y; SMILES canonicalized)
            print("Pre-processed data unavailable, computing...")
            # case1: split train into train and val
            # train_all = pd.read_csv(os.path.join(dataset_dir, "train_set.csv"))
            # if dataset == "DEL":
            #     train, val = train_test_split(
            #         train_all, test_size=0.2, random_state=42,
            #         stratify=train_all[target_name]  # stratify for classification
            #     )
            # else:
            #     train, val = train_test_split(
            #         train_all, test_size=0.2, random_state=42
            #     )
                
            # train = train.reset_index(drop=True)  # optional reset index
            # val = val.reset_index(drop=True)
            # train_set.to_csv(os.path.join(dataset_dir, "train_split.csv"), index=False)  # optional save splits
            # test_set.to_csv(os.path.join(dataset_dir, "test_split.csv"), index=False)

            # case2
            train = pd.read_csv(os.path.join(dataset_dir, "train_set.csv"))
            val = pd.read_csv(os.path.join(dataset_dir, "val_set.csv"))
            test = pd.read_csv(os.path.join(dataset_dir, "test_set.csv"))
            print(f"Train size: {len(train)}, val size: {len(val)}, test size: {len(test)}")

            # 2. Convert to base Data list (smiles and y only)
            print("Loading data splits...")
            train_data = _csv_to_pyg_data(train, train[target_name].values)
            # print("train:", train)
            val_data = _csv_to_pyg_data(val, val[target_name].values)
            test_data = _csv_to_pyg_data(test, test[target_name].values)
            print("\nDataset items look like: ", train_data[0])  #Dataset items look like:  Data(y=1, smiles='CCOc1ncc(C(CN(C(=O)c2ccoc2)C2CC3(CC(OC)C3)C2)C(=O)NC)cc1Cl')

            # --------------------------
            # 3. Compute Morgan fingerprints and attach to Data objects
            print("\n=== Computing fingerprints for train set ===")
            train_data = _add_fingerprint_to_data(train_data, fp_params)
            print("\n=== Computing fingerprints for val set ===")
            val_data = _add_fingerprint_to_data(val_data, fp_params)
            print("\n=== Computing fingerprints for test set ===")
            test_data = _add_fingerprint_to_data(test_data, fp_params)
            print("train_data[0] (Morgan):", train_data[0])  #train_data[0]: Data(y=1, smiles='CCOc1ncc(C(CN(C(=O)c2ccoc2)C2CC3(CC(OC)C3)C2)C(=O)NC)cc1Cl', fingerprint=[2048])
            print("train_data[1600] (Morgan):", train_data[1600])  #train_data[1600]: Data(y=0, smiles='Cc1nc(CN(C)c2nc(NCc3cc(C4CC4)no3)nc(N3CCC3C3CC3)n2)c(C)[nH]1', fingerprint=[2048])
            # --------------------------
            
            # # 4. Import wrapped embedding helper
            # from .embedding import add_embedding_to_data
            # print("\n=== Computing SMILES embeddings for train set ===")
            # train_data = add_embedding_to_data(
            #     train_data,  # Data list with smiles (invalid fingerprint samples already removed)
            #     batch_size=32  # tune batch size by memory
            # )
            # print("train_data[0] (embedding):", train_data[0])
            # print("train_data[1600] embedding shape:", train_data[1600].smiles_embedding.shape)  # torch.Size([768])

            print("\n=== Loading embeddings for train/val/test sets ===")
            _maybe_ensure_embeddings(dataset_dir, **kwargs)
            preprocessed_path_train_emb = os.path.join(dataset_dir, "train_emb.pt")
            train_data = add_embedding_from_pt(train_data, emb_pt_path=preprocessed_path_train_emb)
            preprocessed_path_val_emb = os.path.join(dataset_dir, "val_emb.pt")
            val_data = add_embedding_from_pt(val_data, emb_pt_path=preprocessed_path_val_emb)
            preprocessed_path_test_emb = os.path.join(dataset_dir, "test_emb.pt")
            test_data = add_embedding_from_pt(test_data, emb_pt_path=preprocessed_path_test_emb)
            print("\n=== Embedding attachment verification ===")
            print("train_data[0] (embedding)：", train_data[0])  #train_data[0] (embedding)： Data(y=1, smiles='CC(=O)NC1CCN(C(=O)C(C)Cc2cccc(-c3noc(/C=C/c4ccc(C#N)cc4)n3)c2)CC1', feature_index='HGODEL0034-190-54-26', fingerprint=[2048], smiles_embedding=[768])
            print("train_data[0] embedding shape:", train_data[0].smiles_embedding.shape)  # torch.Size([768])
            print("train_data[1600] embedding shape:", train_data[1600].smiles_embedding.shape)  # torch.Size([768])

            
            
            # 5. Graph feature transforms (x, edge_index, edge_attr)
            transforms = [ChempropFeatures(one_hot=one_hot, max_atomic_number=53), AddNumNodes(), AddMaxEdge(), AddMaxNode()]  # max_atomic_number=53 (iodine)
            if "pe_types" in kwargs.keys() and len(kwargs["pe_types"]) > 0:
                t_posenc = AddPosEnc(kwargs["pe_types"])
                transforms.append(t_posenc)
            transforms = T.Compose(transforms)

            print("Computing ChemProp features for data splits...")
            train = [transforms(data) for data in tqdm(train_data) if data is not None]
            # print("train[0]:", train[0])
            # print("train[1]:", train[1])
            # print("train[5]:", train[5])
            # print("train[0].x:", train[0].x)
            # print("train[0].max_node:", train[0].max_node)
            # print("train[0].max_edge:", train[0].max_edge)
            val = [transforms(data) for data in tqdm(val_data) if data is not None]
            test = [transforms(data) for data in tqdm(test_data) if data is not None]

            print("Caching pre-processed files...")
            torch.save(train, preprocessed_path_train)
            torch.save(val, preprocessed_path_val)
            torch.save(test, preprocessed_path_test)
            print("Caching done")

        print("Determining global node/edge counts...")
        max_edge_global_train, max_node_global_train = get_max_node_edge_global(train)
        # print("max_edge_global_train:", max_edge_global_train)
        # print("max_node_global_train:", max_node_global_train)
        max_edge_global_val, max_node_global_val = get_max_node_edge_global(val)
        # print("max_edge_global_val:", max_edge_global_val)
        # print("max_node_global_val:", max_node_global_val)
        max_edge_global_test, max_node_global_test = get_max_node_edge_global(test)
        # print("max_edge_global_test:", max_edge_global_test)
        # print("max_node_global_test:", max_node_global_test)

        max_edge_global = max(max_edge_global_train, max_edge_global_val, max_edge_global_test)
        max_node_global = max(max_node_global_train, max_node_global_val, max_node_global_test)

        print(f"max_node_global = {max_node_global}")
        print(f"max_edge_global = {max_edge_global}")
        print(f"dataset size = {len(train) + len(val) + len(test)}")

        global_transforms = T.Compose([AddMaxEdgeGlobal(max_edge_global), AddMaxNodeGlobal(max_node_global)])

        print("Applying global node/edge count transforms...")
        train = [global_transforms(data) for data in tqdm(train)]
        val = [global_transforms(data) for data in tqdm(val)]
        test = [global_transforms(data) for data in tqdm(test)]
        # print("train[0]:", train[0])
        # print("train[1]:", train[1])
        # print("train[5]:", train[5])
        # print("train[0].x:", train[0].x)
        # print("train[0].max_node:", train[0].max_node)
        # print("train[0].max_edge:", train[0].max_edge)
            
        print(f"Datasets has {len(train)} train elements")
        print(f"Datasets has {len(val)} validation elements")
        print(f"Datasets has {len(test)} test elements")
        print("train[0] (graph):", train[0]) # train[0]: Data(y=1, smiles='CCOc1ncc(C(CN(C(=O)c2ccoc2)C2CC3(CC(OC)C3)C2)C(=O)NC)cc1Cl', fingerprint=[2048], x=[63, 79], edge_index=[2, 132], edge_attr=[132, 13], num_nodes=63, max_edge=[1], max_node=[1], max_edge_global=[1], max_node_global=[1])
        print("train[1600] (graph):", train[1600]) # train[1600]: Data(y=0, smiles='Cc1nc(CN(C)c2nc(NCc3cc(C4CC4)no3)nc(N3CCC3C3CC3)n2)c(C)[nH]1', fingerprint=[2048], x=[64, 79], edge_index=[2, 138], edge_attr=[138, 13], num_nodes=64, max_edge=[1], max_node=[1], max_edge_global=[1], max_node_global=[1])
        
        train = process_binary_labels(train)
        val = process_binary_labels(val)
        test = process_binary_labels(test)
        
        # print("Applying label scaler for data splits...")
        # train = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(train)]
        # val = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(val)]
        # test = [apply_scaler(data, scaler=y_scaler, convert_to_numpy=False) for data in tqdm(test)]
        # train = CustomPyGDataset(train)
        # val = CustomPyGDataset(val)
        # test = CustomPyGDataset(test)
        print("train[0] after label processing:", train[0])  # train[0] after label processing: Data(y=[1], smiles='CCOc1ncc(C(CN(C(=O)c2ccoc2)C2CC3(CC(OC)C3)C2)C(=O)NC)cc1Cl', fingerprint=[2048], x=[63, 79], edge_index=[2, 132], edge_attr=[132, 13], num_nodes=63, max_edge=[1], max_node=[1], max_edge_global=[1], max_node_global=[1])
        print("train[1600] after label processing:", train[1600])  # train[1600] after label processing: Data(y=[1], smiles='Cc1nc(CN(C)c2nc(NCc3cc(C4CC4)no3)nc(N3CCC3C3CC3)n2)c(C)[nH]1', fingerprint=[2048], x=[64, 79], edge_index=[2, 138], edge_attr=[138, 13], num_nodes=64, max_edge=[1], max_node=[1], max_edge_global=[1], max_node_global=[1])
        print("train[1600].y after label processing:", train[1600].y)  # train[1600].y after label processing: tensor([0.])

        print("Finished loading data!")
        num_classes = 1
        task_type = "binary_classification"

        return train, val, test, num_classes, task_type, None


def get_dataset_train_val_test(dataset, dataset_dir, **kwargs):
    if dataset == "DEL":
        return load_del_chemprop(dataset, dataset_dir, **kwargs)
    raise ValueError(f"Unsupported dataset: {dataset}")

    
