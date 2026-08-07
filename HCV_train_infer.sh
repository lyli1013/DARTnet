#!/bin/bash
# HCV example: train_tune -> train_final -> infer.
# Uses precomputed embeddings under dataset_HCV/ (*_emb.pt).
# MolTran / --molformer-* are NOT required for this script.
# For on-the-fly embedding generation, see HCV_train_infer_moltran.sh.

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
OUTPUT_ROOT="${DART_OUTPUT_ROOT:-${PROJECT_ROOT}/outputs}"
LOG_ROOT="${DART_LOG_ROOT:-${PROJECT_ROOT}/logs}"
GPU_DEVICES="${GPU_DEVICES:-0}"   # same meaning as --gpu-devices in train / predict

cd "${PROJECT_ROOT}"
mkdir -p "${OUTPUT_ROOT}" "${LOG_ROOT}"

data_id=(dataset_HCV)
# Must match get_gnn_wandb_name() with train.py c603 defaults
RUN_DIR="GNN+DEL+T=Label+S=42+GAT+GC=0.5+OPTD=1e-10+OH=True+NDIM=256+NL=4+GIDIM=256+GATH=2+GATD=0.9+BS=32+ESP=5+lr=0.0001_atteCat_embdrop0.1_fpdrop0.5_gnndrop0.0"

# ---------- stage1: train_tune ----------
for id in "${data_id[@]}"; do
    out_path="${OUTPUT_ROOT}/out_S5_${id}"
    nohup python -u -m dartnet.train \
        --train_stage train_tune \
        --dataset-dir "${PROJECT_ROOT}/${id}" \
        --dataset-id ${id} \
        --out-path "$out_path" \
        --gpu-devices "${GPU_DEVICES}" \
        >> "${LOG_ROOT}/out_S5_${id}.log" 2>&1 &
    wait
done
echo "train_tune: all jobs submitted"

# ---------- stage2: train_final ----------
for id in "${data_id[@]}"; do
    out_path="${OUTPUT_ROOT}/out_S5_${id}"
    nohup python -u -m dartnet.train \
        --train_stage train_final \
        --dataset-dir "${PROJECT_ROOT}/${id}" \
        --dataset-id ${id} \
        --out-path "$out_path" \
        --gpu-devices "${GPU_DEVICES}" \
        >> "${LOG_ROOT}/out_S5_${id}.log" 2>&1 &
    wait
done
echo "train_final: all jobs submitted"

# ---------- infer: HCV_screened_positive ----------
for id in "${data_id[@]}"; do
    out_final="${OUTPUT_ROOT}/out_S5_${id}_final"
    nohup python -m dartnet.predict \
        --infer \
        --ckpt-path "${out_final}/${RUN_DIR}/last.ckpt" \
        --dataset-dir "${PROJECT_ROOT}/${id}" \
        --infer-file-name HCV_screened_positive \
        --output-path "${out_final}/${RUN_DIR}" \
        --use-gpu \
        --gpu-devices "${GPU_DEVICES}" \
        >> "${LOG_ROOT}/out_S5_${id}_infer.log" 2>&1 &
    wait
done
echo "infer HCV_positive: all jobs submitted"

# ---------- infer: HCV_screened_negative ----------
for id in "${data_id[@]}"; do
    out_final="${OUTPUT_ROOT}/out_S5_${id}_final"
    nohup python -m dartnet.predict \
        --infer \
        --ckpt-path "${out_final}/${RUN_DIR}/last.ckpt" \
        --dataset-dir "${PROJECT_ROOT}/${id}" \
        --infer-file-name HCV_screened_negative \
        --output-path "${out_final}/${RUN_DIR}" \
        --use-gpu \
        --gpu-devices "${GPU_DEVICES}" \
        >> "${LOG_ROOT}/out_S5_${id}_infer.log" 2>&1 &
    wait
done
echo "infer HCV_negative: all jobs submitted"
