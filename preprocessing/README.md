# MolFormer Preprocessing

Embedding extraction is integrated into `data_loading.py`. When training or inference
uses a dataset mode that includes `emb`, DART will automatically generate missing
`*_emb.pt` files by delegating to the **MolTran_CUDA11** conda environment (or
`--molformer-env`), matching the historical notebook output.

## Layout

```
preprocessing/
├── extract_embeddings.py          # ensure_embeddings() + CLI (DARTnet subprocess orchestrator)
├── extract_embeddings_moltran.py  # runs inside MolTran env (notebook-equivalent logic)
├── molformer/                     # MolFormer inference code + hparams.yaml + bert_vocab.txt
└── checkpoints/
    └── N-Step-Checkpoint_3_30000.ckpt   # pretrained MolFormer weights (~536 MB)
```

Default paths are resolved relative to this directory; no environment variables required.
The checkpoint exceeds GitHub's 100 MB file limit — use **Git LFS** or host the `.ckpt` separately and place it under `preprocessing/checkpoints/`.


## Automatic usage (recommended)

Training:

```bash
python -m dartnet.train \
    --dataset-dir /path/to/split_4 \
    --molformer-env MolTran_CUDA11 \
    ...
```

Inference:

```bash
python -m dartnet.predict \
    --data-path /path/to/data \
    --infer-file-name my_infer_set \
    --molformer-env MolTran_CUDA11 \
    ...
```

Embeddings are saved beside the CSV files:

- `train_set.csv` → `train_emb.pt`
- `val_set.csv` → `val_emb.pt`
- `test_set.csv` → `test_emb.pt`
- `{infer_file_name}.csv` → `{infer_file_name}_emb.pt`

## Standalone CLI

From DARTnet (delegates to MolTran):

```bash
python preprocessing/extract_embeddings.py \
    --data-dir /path/to/split_4 \
    --molformer-env MolTran_CUDA11 \
    --device cuda:0
```

Directly in MolTran env:

```bash
conda run -n MolTran_CUDA11 python preprocessing/extract_embeddings_moltran.py \
    --data-dir /path/to/split_4 \
    --device cuda:0
```



## Notes

- DARTnet training and MolTran embedding generation use **separate conda envs**.
- Existing `.pt` files are skipped by default (`skip_if_exists=True`).
- Delete stale `.pt` files if the underlying CSV changes.
- `--molformer-ckpt-path` / `--molformer-hparams-path` are optional; defaults point to `preprocessing/checkpoints/` and `preprocessing/molformer/hparams.yaml`.
- Embedding GPU follows training `--gpu-devices` (e.g. `cuda:4`).
