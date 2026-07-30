#!/usr/bin/env python3
"""
MolFormer embedding extraction for MolTran_CUDA11 (notebook-equivalent).

Must be launched from the MolTran conda env, e.g.:

    python preprocessing/extract_embeddings_moltran.py \\
        --data-dir /path/to/split_1 --device cuda:0 --batch-size 32
"""

from __future__ import annotations

import argparse
import os
import sys
import time
from argparse import Namespace
from pathlib import Path
from typing import Iterable, List, Optional, Sequence, Tuple

import pandas as pd
import torch
import yaml
from rdkit import Chem

PREPROCESSING_DIR = Path(__file__).resolve().parent
MOLFORMER_CODE_ROOT = PREPROCESSING_DIR / "molformer"
DEFAULT_HPARAMS = str(MOLFORMER_CODE_ROOT / "hparams.yaml")
DEFAULT_CKPT = str(
    PREPROCESSING_DIR / "checkpoints" / "N-Step-Checkpoint_3_30000.ckpt"
)
DEFAULT_VOCAB = "bert_vocab.txt"

DEFAULT_TRAIN_DATASETS: Sequence[Tuple[str, str]] = (
    ("train_set.csv", "train_emb.pt"),
    ("val_set.csv", "val_emb.pt"),
    ("test_set.csv", "test_emb.pt"),
)


def get_molformer_code_root() -> Path:
    if not MOLFORMER_CODE_ROOT.is_dir():
        raise FileNotFoundError(f"MolFormer code not found: {MOLFORMER_CODE_ROOT}")
    return MOLFORMER_CODE_ROOT


def _setup_molformer_paths() -> None:
    code_root = str(get_molformer_code_root())
    if code_root not in sys.path:
        sys.path.insert(0, code_root)


def batch_split(data: List, batch_size: int = 64) -> Iterable[List]:
    i = 0
    while i < len(data):
        yield data[i : min(i + batch_size, len(data))]
        i += batch_size


def canonicalize(s: str) -> Optional[str]:
    try:
        return Chem.MolToSmiles(
            Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False
        )
    except Exception:
        return None


class Data:
    def __init__(self, smiles_id, smiles, smiles_embedding=None):
        self.smiles_id = smiles_id
        self.smiles = smiles
        self.smiles_embedding = smiles_embedding


def add_embedding_to_data(
    data_list: List[Data],
    hparams_path: str = DEFAULT_HPARAMS,
    vocab_path: str = DEFAULT_VOCAB,
    ckpt_path: str = DEFAULT_CKPT,
    batch_size: int = 32,
    device: str = "cuda" if torch.cuda.is_available() else "cpu",
    lm=None,
    tokenizer=None,
) -> List[Data]:
    start_time = time.time()
    print(
        f"Embedding start time: "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}"
    )

    smiles_list = [data.smiles for data in data_list]
    smiles_id_list = [data.smiles_id for data in data_list]
    print(f"Samples to embed: {len(smiles_list)}")

    valid_indices: List[int] = []
    valid_smiles: List[str] = []
    invalid_smiles: List[str] = []
    invalid_feature_indices: List[str] = []

    for idx, (s, fi) in enumerate(zip(smiles_list, smiles_id_list)):
        canon_s = canonicalize(s)
        if canon_s is not None:
            valid_indices.append(idx)
            valid_smiles.append(canon_s)
        else:
            invalid_smiles.append(s)
            invalid_feature_indices.append(fi)

    print(
        f"Valid embedding samples: {len(valid_smiles)}, "
        f"invalid: {len(invalid_smiles)}"
    )
    if invalid_smiles:
        print(
            "Invalid SMILES examples: "
            f"{list(zip(invalid_feature_indices[:5], invalid_smiles[:5]))}"
        )

    if lm is None or tokenizer is None:
        _setup_molformer_paths()
        from tokenizer.tokenizer import MolTranBertTokenizer
        from train_pubchem_light import LightningModule
        from fast_transformers.masking import LengthMask as LM

        with open(hparams_path, "r") as f:
            config = Namespace(**yaml.safe_load(f))

        molformer_code_root = get_molformer_code_root()
        cwd = os.getcwd()
        os.chdir(molformer_code_root)
        try:
            tokenizer = MolTranBertTokenizer(vocab_path)
            lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(
                ckpt_path, config=config, vocab=tokenizer.vocab
            )
            lm = lm.to(device)
            lm.eval()
        finally:
            os.chdir(cwd)
    else:
        _setup_molformer_paths()
        from fast_transformers.masking import LengthMask as LM

    print(f"Generating embeddings (batch_size={batch_size}, device={device})...")
    embeddings: List[torch.Tensor] = []
    with torch.no_grad():
        for batch in batch_split(valid_smiles, batch_size=batch_size):
            batch_enc = tokenizer.batch_encode_plus(
                batch, padding=True, add_special_tokens=True
            )
            idx = torch.tensor(batch_enc["input_ids"], device=device)
            mask = torch.tensor(batch_enc["attention_mask"], device=device)

            token_embeddings = lm.blocks(
                lm.tok_emb(idx), length_mask=LM(mask.sum(-1))
            )
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            batch_emb = sum_embeddings / sum_mask
            embeddings.append(batch_emb.cpu())

    valid_embeddings = torch.cat(embeddings, dim=0)

    data_with_emb: List[Data] = []
    for idx, emb in zip(valid_indices, valid_embeddings):
        data = data_list[idx]
        data.smiles_embedding = emb
        data_with_emb.append(data)

    end_time = time.time()
    total_time = end_time - start_time
    minutes = int(total_time // 60)
    seconds = round(total_time % 60, 2)
    print("\n=== Embedding complete ===")
    print(
        f"Embedding end time: "
        f"{time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}"
    )
    print(f"Total time: {minutes}m {seconds}s ({total_time:.2f}s)")
    if valid_smiles:
        print(f"Average time per valid sample: {total_time / len(valid_smiles):.4f}s")

    assert len(data_with_emb) == len(valid_smiles), (
        "Embedding count does not match Data objects"
    )
    return data_with_emb


def process_dataset(
    csv_path: Path,
    output_path: Path,
    batch_size: int,
    device: str,
    hparams_path: str,
    vocab_path: str,
    ckpt_path: str,
    lm=None,
    tokenizer=None,
) -> None:
    print("=====================================")
    print(f"Processing: {csv_path.name} -> {output_path.name}")
    print("=====================================")

    try:
        df = pd.read_csv(csv_path)
        print(f"Read CSV: {csv_path} ({len(df)} rows)")
    except Exception as exc:
        print(f"Failed to read CSV: {exc}; skipping dataset")
        return

    csv_data_list = [
        Data(smiles_id=row["FeatureIndex"], smiles=row["Smiles"])
        for _, row in df.iterrows()
    ]

    csv_data_with_emb = add_embedding_to_data(
        data_list=csv_data_list,
        hparams_path=hparams_path,
        vocab_path=vocab_path,
        ckpt_path=ckpt_path,
        batch_size=batch_size,
        device=device,
        lm=lm,
        tokenizer=tokenizer,
    )

    smile_to_embedding = {}
    fi_to_embedding = {}
    fi_to_smile = {}
    for item in csv_data_with_emb:
        smile_to_embedding[item.smiles] = item.smiles_embedding
        fi_to_embedding[item.smiles_id] = item.smiles_embedding
        fi_to_smile[item.smiles_id] = item.smiles

    output_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "smile_to_embedding": smile_to_embedding,
            "fi_to_embedding": fi_to_embedding,
            "fi_to_smile": fi_to_smile,
            "dataset_type": csv_path.name.split("_")[0],
        },
        output_path,
    )

    print(f"Saved embedding map to: {output_path}")
    print(f"   - Valid mappings: {len(smile_to_embedding)}")
    if csv_data_with_emb:
        sample_fi = csv_data_with_emb[0].smiles_id
        print(
            f"   - Example: FeatureIndex={sample_fi} -> "
            f"SMILES={fi_to_smile[sample_fi][:30]}..."
        )
        print(f"   - Embedding shape: {fi_to_embedding[sample_fi].shape}")
    print("=====================================\n")


def run_extraction(
    data_dir: str,
    ckpt_path: str = DEFAULT_CKPT,
    hparams_path: str = DEFAULT_HPARAMS,
    vocab_path: str = DEFAULT_VOCAB,
    device: str = "cuda:0",
    batch_size: int = 32,
    datasets: Optional[Sequence[Tuple[str, str]]] = None,
    skip_if_exists: bool = True,
) -> None:
    data_dir_path = Path(data_dir)
    dataset_pairs = list(datasets or DEFAULT_TRAIN_DATASETS)

    pending: List[Tuple[Path, Path]] = []
    for csv_name, pt_name in dataset_pairs:
        csv_path = data_dir_path / csv_name
        pt_path = data_dir_path / pt_name
        if not csv_path.is_file():
            print(f"[Embedding] Skip: CSV not found {csv_path}")
            continue
        if skip_if_exists and pt_path.is_file():
            print(f"[Embedding] Exists, skip {pt_path}")
            continue
        pending.append((csv_path, pt_path))

    if not pending:
        return

    print(
        f"[Embedding] MolTran env: processing {len(pending)} file(s) in {data_dir_path}"
    )
    molformer_code_root = get_molformer_code_root()
    cwd = os.getcwd()
    os.chdir(molformer_code_root)
    try:
        for csv_path, pt_path in pending:
            process_dataset(
                csv_path=csv_path,
                output_path=pt_path,
                batch_size=batch_size,
                device=device,
                hparams_path=hparams_path,
                vocab_path=vocab_path,
                ckpt_path=ckpt_path,
            )
    finally:
        os.chdir(cwd)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Extract MolFormer embeddings (MolTran env)")
    parser.add_argument("--data-dir", type=str, required=True)
    parser.add_argument(
        "--hparams-path",
        type=str,
        default=DEFAULT_HPARAMS,
    )
    parser.add_argument(
        "--ckpt-path",
        type=str,
        default=DEFAULT_CKPT,
    )
    parser.add_argument("--vocab-path", type=str, default=DEFAULT_VOCAB)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument(
        "--device",
        type=str,
        default="cuda:0" if torch.cuda.is_available() else "cpu",
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        default=None,
        help="CSV:PT pairs, e.g. train_set.csv:train_emb.pt",
    )
    parser.add_argument(
        "--skip-if-exists",
        dest="skip_if_exists",
        action="store_true",
        default=True,
    )
    parser.add_argument(
        "--no-skip-if-exists",
        dest="skip_if_exists",
        action="store_false",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.datasets:
        dataset_pairs = [tuple(item.split(":")) for item in args.datasets]
    else:
        dataset_pairs = DEFAULT_TRAIN_DATASETS

    run_extraction(
        data_dir=args.data_dir,
        ckpt_path=args.ckpt_path,
        hparams_path=args.hparams_path,
        vocab_path=args.vocab_path,
        device=args.device,
        batch_size=args.batch_size,
        datasets=dataset_pairs,
        skip_if_exists=args.skip_if_exists,
    )
    print("All datasets processed.")


if __name__ == "__main__":
    main()
