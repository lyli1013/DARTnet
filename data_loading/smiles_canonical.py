"""Shared SMILES canonicalization for GNN / Morgan pipelines."""

from __future__ import annotations

from rdkit import Chem


def canonicalize_smiles_for_data(smiles: str):
    """Canonical SMILES with stereochemistry."""
    try:
        mol = Chem.MolFromSmiles(smiles)
        if mol is None:
            return None
        # return Chem.MolToSmiles(mol, canonical=True, isomericSmiles=True)
        return Chem.MolToSmiles(mol)
    except Exception:
        return None
