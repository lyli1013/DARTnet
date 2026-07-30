# Morgan_cal.py (refactored as an importable module)
import numpy as np
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from rdkit import Chem
from rdkit.Chem import AllChem, MACCSkeys


# --------------------------
# Core fingerprint computation functions (original logic preserved)
# --------------------------
def calculate_morgan_fingerprint(smiles_string, radius=2, nBits=2048, useChirality=False):
    """Compute Morgan fingerprint for a single SMILES string."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is not None:
        morgan_fingerprint = AllChem.GetMorganFingerprintAsBitVect(
            mol, radius, nBits=nBits, useChirality=useChirality
        )
        morgan_fingerprint_array = np.zeros((nBits,), dtype=np.float32)  # float32 for PyTorch compatibility
        AllChem.DataStructs.ConvertToNumpyArray(morgan_fingerprint, morgan_fingerprint_array)
        return morgan_fingerprint_array
    else:
        return None  # Return None for invalid SMILES


def calculate_maccs_fingerprint(smiles_string):
    """Compute MACCS fingerprint for a single SMILES string (optional)."""
    mol = Chem.MolFromSmiles(smiles_string)
    if mol is not None:
        maccs = MACCSkeys.GenMACCSKeys(mol)
        maccs_array = np.zeros((167,), dtype=np.float32)  # MACCS is fixed at 167 bits
        AllChem.DataStructs.ConvertToNumpyArray(maccs, maccs_array)
        return maccs_array
    else:
        return None


# --------------------------
# External API: batch fingerprint computation
# --------------------------
def compute_fingerprints_batch(
    smiles_list,
    fingerprint_type="morgan2048",  # "morgan" or "maccs"
    radius=2,
    nBits=2048,
    useChirality=False,
    num_processes=4,
    chunk_size=1000
):
    """
    Batch-compute fingerprints for a list of SMILES strings (external API).
    Args:
        smiles_list (list): List of SMILES strings
        fingerprint_type (str): Fingerprint type, "morgan" or "maccs"
        radius (int): Morgan fingerprint radius (morgan only)
        nBits (int): Morgan fingerprint dimension (morgan only)
        useChirality (bool): Whether to include chirality (morgan only)
        num_processes (int): Number of parallel processes
        chunk_size (int): Chunk size (avoids memory overflow)
    Returns:
        tuple: (valid_smiles, valid_fingerprints, invalid_smiles)
            - valid_smiles: List of valid SMILES
            - valid_fingerprints: Valid fingerprint array (shape: [n_valid, n_bits])
            - invalid_smiles: List of invalid SMILES
    """
    # Select fingerprint function and bind parameters
    if "morgan" in fingerprint_type:
        calc_fn = partial(
            calculate_morgan_fingerprint,
            radius=radius,
            nBits=nBits,
            useChirality=useChirality
        )
    elif fingerprint_type == "maccs":
        calc_fn = calculate_maccs_fingerprint
    else:
        raise ValueError(f"Unsupported fingerprint type: {fingerprint_type}; only 'morgan' or 'maccs' are supported")

    # Compute fingerprints in parallel chunks
    all_fingerprints = []
    for i in range(0, len(smiles_list), chunk_size):
        chunk_smiles = smiles_list[i:i+chunk_size]
        print(f"Computing fingerprints: processing SMILES {i}-{min(i+chunk_size, len(smiles_list))}")
        
        with ProcessPoolExecutor(max_workers=num_processes) as executor:
            chunk_fps = list(tqdm(
                executor.map(calc_fn, chunk_smiles),
                total=len(chunk_smiles),
                desc=f"Chunk {i//chunk_size + 1}"
            ))
        all_fingerprints.extend(chunk_fps)

    # Filter valid data (remove fingerprints for invalid SMILES)
    valid_smiles = []
    valid_fps = []
    invalid_smiles = []
    for smiles, fp in zip(smiles_list, all_fingerprints):
        if fp is not None:
            valid_smiles.append(smiles)
            valid_fps.append(fp)
        else:
            invalid_smiles.append(smiles)

    # Convert to numpy array for downstream processing
    valid_fps = np.array(valid_fps, dtype=np.float32) if valid_fps else np.array([])
    return valid_smiles, valid_fps, invalid_smiles