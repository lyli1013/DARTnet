# MolFormer checkpoint

Place the pretrained checkpoint here:

- `N-Step-Checkpoint_3_30000.ckpt` (~536 MB)

This file is used by default by `extract_embeddings.py` and training when
`--molformer-ckpt-path` is omitted.

For GitHub, track with Git LFS:

```bash
git lfs track "preprocessing/checkpoints/*.ckpt"
```
