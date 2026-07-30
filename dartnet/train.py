import argparse, os, random
import copy
import sys
import torch
# 禁用TF32以获得更好的数值稳定性
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
# import wandb  # disabled: no local wandb/ logging
import numpy as np
import pytorch_lightning as pl

from pathlib import Path
from torch_geometric.seed import seed_everything
from torch_geometric.loader import DataLoader
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
# from pytorch_lightning.loggers import WandbLogger

# Imports from this project
sys.path.append(os.path.realpath("."))

from dartnet.model import DartNet
from data_loading.data_loading import get_dataset_train_val_test
from dartnet.config import (
    save_gnn_arguments_to_json,
    load_gnn_arguments_from_json,
    validate_gnn_argparse_arguments,
    get_gnn_wandb_name,
    get_epochs_from_ckpt_dir,
    save_classification_metrics,
    save_classification_metrics_tra_val,
    save_classification_metrics_val_best,
)


def fix_seed(seed):
    """
    修正版本的随机种子设置函数
    
    Args:
        seed: 随机种子
        full_deterministic: 是否启用完全确定性模式（可能降低性能）
    """
    if seed is None:
        seed = random.randint(1, 10000)
    
    # 1. 基础随机种子
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    
    # 2. 设置环境变量
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'  # 重要：对于CUDA 10.2+
    
    # 3. GPU相关设置
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        
        # if full_deterministic:
        # 完全确定性模式（牺牲性能换取可重复性）
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        
        
        
        # 启用CUDA确定性操作
        torch.use_deterministic_algorithms(True, warn_only=True)
        # else:
        #     # 性能模式（可能有一些随机性）
        #     torch.backends.cudnn.deterministic = True
        #     torch.backends.cudnn.benchmark = False
        #     torch.backends.cuda.matmul.allow_tf32 = True
        #     torch.backends.cudnn.allow_tf32 = True
    
    # 4. 线程设置
    torch.set_num_threads(1)  # 保持，有助于确定性
    
    # print(f"[Info] 随机种子已设置为: {seed}")
    # print(f"[Info] 确定性模式: {full_deterministic}")
    
    return seed

def main():
    # torch.set_num_threads(1)

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_stage", 
        type=str, 
        required=True, 
        choices=["train_tune", "train_final"],
        help="训练测试阶段：tune（调参）、train_final（合并训验证集训模型）"
    )
    parser.add_argument(
        "--eval",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Run evaluation after training (default: True)",
    )
    
    # Seed for seed_everything
    parser.add_argument("--seed", type=int, default=42)

    # Dataset arguments (c603 defaults; scripts should pass --dataset-dir / --dataset-id)
    # Legacy aliases kept: --dataset-download-dir / --dataset-split-id
    parser.add_argument("--dataset", type=str, default="DEL")
    parser.add_argument(
        "--dataset-dir",
        "--dataset-download-dir",
        dest="dataset_download_dir",
        type=str,
        help="Path to dataset folder containing train_set.csv / val_set.csv / test_set.csv",
    )
    parser.add_argument(
        "--dataset-id",
        "--dataset-split-id",
        dest="dataset_split_id",
        type=str,
        default="split_1",
        help="Dataset identifier used in logs / metric filenames",
    )
    parser.add_argument("--dataset-one-hot", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument("--dataset-target-name", type=str, default="Label")
    parser.add_argument("--fingerprint-type", type=str, default="morgan2048")
    parser.add_argument(
        "--molformer-ckpt-path",
        type=str,
        default=None,
        help="MolFormer pretrained checkpoint; embeddings are generated in dataset-dir if missing",
    )
    parser.add_argument(
        "--molformer-hparams-path",
        type=str,
        default=None,
        help="MolFormer hparams.yaml (optional, uses project default if omitted)",
    )
    parser.add_argument(
        "--molformer-env",
        type=str,
        default="MolTran_CUDA11",
        help="Conda env for MolFormer embedding extraction (subprocess)",
    )
    # emb arguments
    parser.add_argument("--emb-feat-dim-out", type=int, default=100)
    parser.add_argument("--emb-feat-out-drop", type=float, default=0.1)
    parser.add_argument("--fp-feat-out-drop", type=float, default=0.5)
    # KAN arguments
    parser.add_argument("--kan-grid-size", type=int, default=5)
    parser.add_argument("--kan-spline-order", type=int, default=3)
    parser.add_argument("--kan-hidden-dim", type=int, default=64)
    parser.add_argument("--kan-grid-range-min", type=float, default=-1.0)
    parser.add_argument("--kan-grid-range-max", type=float, default=1.0)
    parser.add_argument("--kan-reg-overall-weight", type=float, default=1e-5)
    parser.add_argument("--kan-reg-activation-weight", type=float, default=1.0)
    parser.add_argument("--kan-reg-entropy-weight", type=float, default=1.0)
    parser.add_argument("--kan-reg-use", default=True, action=argparse.BooleanOptionalAction)
    parser.add_argument(
        "--cross-dim",
        type=int,
        default=48,
        help="Projection dim for cross_interact pairwise features (c603 default)",
    )
    
    # GNN arguments (c603 defaults)
    parser.add_argument("--output-node-dim", type=int, default=256)
    parser.add_argument("--num-layers", type=int, default=4)
    parser.add_argument("--gnn-intermediate-dim", type=int, default=256)
    parser.add_argument("--gat-attn-heads", type=int, default=2)
    parser.add_argument("--gat-dropout", type=float, default=0.9)
    parser.add_argument("--gnn-feat-out-drop", type=float, default=0.0)

    # Learning hyperparameters (c603 defaults)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument(
        "--monitor-loss-name",
        type=str,
        default=None,
        help="If unset: Validation PRAUC for train_tune, train_loss for train_final",
    )
    parser.add_argument("--gradient-clip-val", type=float, default=0.5)
    parser.add_argument("--optimiser-weight-decay", type=float, default=1e-10)
    parser.add_argument("--early-stopping-patience", type=int, default=5)

    # Path/config arguments
    parser.add_argument("--ckpt-path", type=str)
    parser.add_argument("--out-path", type=str)
    parser.add_argument("--config-json-path", type=str)
    parser.add_argument("--wandb-project-name", type=str)
    parser.add_argument("--gpu-devices", type=int, default=2)

    args = parser.parse_args()

    # Stage-dependent monitor default (c603)
    if args.monitor_loss_name is None:
        args.monitor_loss_name = (
            "Validation PRAUC" if args.train_stage == "train_tune" else "train_loss"
        )
    
    if args.config_json_path:
        argsdict = load_gnn_arguments_from_json(args.config_json_path)
        validate_gnn_argparse_arguments(argsdict)
    else:
        argsdict = vars(args)
        validate_gnn_argparse_arguments(argsdict)   # 检查必要参数是否存在
        del argsdict["config_json_path"]

    # seed_everything(argsdict["seed"])   #两种固定方式均不能完全重复每一次训练，微小差异
    fix_seed(argsdict["seed"])

    # Dataset arguments
    dataset = argsdict["dataset"]  #数据集名称
    dataset_download_dir = argsdict["dataset_download_dir"]
    dataset_split_id = argsdict["dataset_split_id"]
    dataset_one_hot = argsdict["dataset_one_hot"]
    target_name = argsdict["dataset_target_name"]
    fingerprint_type = argsdict["fingerprint_type"]
    molformer_ckpt_path = argsdict.get("molformer_ckpt_path")
    molformer_hparams_path = argsdict.get("molformer_hparams_path")
    molformer_env = argsdict.get("molformer_env", "MolTran_CUDA11")
    argsdict.setdefault("feature_fusion", "atteCat")
    argsdict.setdefault("conv_type", "GAT")
    argsdict.setdefault("dataset_mode", "graph+fp+emb")
    argsdict.setdefault("mlpemb_type", "MLPemb")
    argsdict.setdefault("arch_variant", "cross_interact")
    argsdict.setdefault("train_regime", "gpu-32")

    # Learning hyperparameters
    batch_size = argsdict["batch_size"]
    early_stopping_patience = argsdict["early_stopping_patience"]
    gradient_clip_val = argsdict["gradient_clip_val"]
    monitor_loss_name = argsdict["monitor_loss_name"]
    # Path/config arguments
    ckpt_path = argsdict["ckpt_path"]
    out_path = argsdict["out_path"]
    wandb_project_name = argsdict.get("wandb_project_name")  # unused when wandb disabled
    gpu_devices = argsdict["gpu_devices"]

    if monitor_loss_name == "MCC" or "MCC" in monitor_loss_name:
        monitor_loss_name = "Validation MCC"

    if dataset == "DEL":
        assert target_name is not None, "A target must be specified for DEL!"

    if "emb" in argsdict["dataset_mode"] and not molformer_ckpt_path:
        raise ValueError(
            "dataset_mode includes 'emb' but --molformer-ckpt-path was not provided."
        )

    argsdict["kan_reg_config"] = {
        "use_regularization": argsdict.get("kan_reg_use", True),
        "activation_weight": argsdict.get("kan_reg_activation_weight", 1.0),
        "entropy_weight": argsdict.get("kan_reg_entropy_weight", 1.0),
        "overall_weight": argsdict.get("kan_reg_overall_weight", 1e-6),
    }
    argsdict["kan_grid_range"] = [
        argsdict.get("kan_grid_range_min", -1.0),
        argsdict.get("kan_grid_range_max", 1.0),
    ]

    ############## Data loading ##############
    # Graph-level task branch
    train, val, test, num_classes, task_type, scaler = get_dataset_train_val_test(
        dataset=dataset,
        dataset_dir=dataset_download_dir,
        one_hot=dataset_one_hot,
        target_name=target_name,
        # 传递指纹计算参数（可选，使用默认值则无需传递）
        fingerprint_type=fingerprint_type,    # 计算Morgan指纹
        morgan_radius=2,              # 半径=2（ECFP4）
        morgan_nBits=2048,            # 维度=1024（默认2048）
        useChirality=True,            # 考虑手性
        num_processes=4,              # 8个并行进程（根据CPU核心数调整）
        fp_chunk_size=2000,           # 每块处理2000个样本
        molformer_ckpt_path=molformer_ckpt_path,
        molformer_hparams_path=molformer_hparams_path,
        molformer_env=molformer_env,
        gpu_devices=gpu_devices,
    )
    print("type(train)", type(train))
    print("train[0]", train[0])
    if len(train) % batch_size == 1:
        batch_size += 1

    num_features = train[0].x.shape[-1]   # 每个结点的维度
    edge_dim = None
    if hasattr(train[0], "edge_attr") and train[0].edge_attr is not None:   #判断样本中是否具有edge_attr属性且不为空
        edge_dim = train[0].edge_attr.shape[-1]    # 每条边的维度

    if args.train_stage == "train_tune":
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        if val==None:
            val_loader=test_loader
        else:
            val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    elif args.train_stage == "train_final":
        train = train + val
        train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=False, drop_last=True)
        # val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
        test_loader = DataLoader(test, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=False)
    ############## Data loading ##############

    
    run_name = get_gnn_wandb_name(argsdict)
    if args.train_stage == "train_tune":
        output_save_dir = os.path.join(out_path, run_name)
        Path(output_save_dir).mkdir(exist_ok=True, parents=True)
    elif args.train_stage == "train_final":
        output_save_dir_tune = os.path.join(out_path, run_name)  # 调参时确定最优模型的保存路径，获取epoch
        try:
            ckpt_epochs, best_ckpt_path = get_epochs_from_ckpt_dir(output_save_dir_tune)
            print(f"{output_save_dir_tune}中找到的ckpt文件及对应的epoch:")
            for filename, epoch in ckpt_epochs.items():
                print(f"{filename} → epoch={epoch}")
                fixed_epochs = epoch
        except Exception as e:
            print(f"epoch获取处理失败: {e}")
    
        output_save_dir = os.path.join(f'{out_path}_final', run_name)
        Path(output_save_dir).mkdir(exist_ok=True, parents=True)
        
    config_json_path = save_gnn_arguments_to_json(argsdict, output_save_dir)
    print("monitor_loss_name:", monitor_loss_name)
    # Logging (wandb disabled — metrics still saved as CSV under output_save_dir)
    # logger = WandbLogger(
    #                     name=run_name,
    #                     project=wandb_project_name,
    #                     mode="offline"   # 设置为离线模式
    #                     )
    logger = False
    _ = wandb_project_name  # CLI arg retained for script compatibility

    # Callbacks: maximize scores (AUROC/PRAUC/F1/MCC/Recall), minimize losses
    monitor_mode = "min" if "loss" in monitor_loss_name.lower() else "max"
    if args.train_stage == "train_tune":
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_loss_name,  # 监控的指标名
            dirpath=output_save_dir,
            filename="{epoch:03d}",
            mode=monitor_mode,   # 监控模式（最大化或最小化）
            save_top_k=1,
        )

        early_stopping_callback = EarlyStopping(
            monitor=monitor_loss_name, patience=early_stopping_patience, mode=monitor_mode
        )
        callbacks = [checkpoint_callback, early_stopping_callback]
    elif args.train_stage == "train_final":
        checkpoint_callback = ModelCheckpoint(
            monitor=monitor_loss_name,  # 监控训练集指标
            dirpath=output_save_dir,
            filename="{epoch:03d}",  # 文件名包含总epoch，便于识别
            mode=monitor_mode,
            save_top_k=1,  # 仅保存最后一轮（或训练集最优）模型
            save_last=True,  # 额外保存最后一轮模型（推荐，确保训练满epoch）
        )
        callbacks = [checkpoint_callback]  # 无EarlyStopping
    
    ############## Learning and model set-up ##############
    gnn_args = copy.deepcopy(argsdict)
    gnn_args = gnn_args | dict(
        task_type=task_type,
        num_features=num_features,
        linear_output_size=num_classes,
        scaler=scaler,
        edge_dim=edge_dim,
        out_path=output_save_dir,
    )

    model = DartNet(**gnn_args)   #Python的“拆包”机制（**操作符），会自动匹配字典的键和模型中的参数名
    print("model:", model)
    precision = "32"

    trainer_args = dict(
        callbacks=callbacks,
        logger=logger,
        # min_epochs=10,
        # max_epochs=100,
        # min_epochs=1 if args.train_stage != "train_final" else early_stopping_patience,  #最终训练固定epoch
        # max_epochs=100 if args.train_stage != "train_final" else early_stopping_patience,
        min_epochs=1 if args.train_stage != "train_final" else fixed_epochs,  #最终训练固定epoch
        max_epochs=100 if args.train_stage != "train_final" else fixed_epochs,
        # devices=1,  #gpu数量
        devices=[gpu_devices],  #指定 GPU 编号
        check_val_every_n_epoch=1,
        num_sanity_val_steps=0,
        precision=precision,
        gradient_clip_val=gradient_clip_val,
    )
    
    trainer_args = trainer_args | dict(accelerator="gpu")
    ############## Learning and model set-up ##############

    trainer = pl.Trainer(**trainer_args)

    if args.train_stage == "train_tune":
        trainer.fit(
            model=model, train_dataloaders=train_loader, val_dataloaders=[val_loader, test_loader], ckpt_path=ckpt_path
        )
        
        
    elif args.train_stage == "train_final":
        trainer.fit(model=model, train_dataloaders=train_loader, ckpt_path=ckpt_path)


    # trainer.predict(model=model, dataloaders=test_loader, ckpt_path="best")
    if args.train_stage == "train_tune" and args.eval:
        trainer.test(model=model, dataloaders=val_loader, ckpt_path="best")

        val_metrics_best_path = os.path.join(out_path, f"val_metrics_best_{dataset_split_id}.csv")
        hyperparameters = {
                "conv_type": argsdict["conv_type"],
                "NL": argsdict["num_layers"],
                "GIDM": argsdict["gnn_intermediate_dim"],
                "GATH": argsdict["gat_attn_heads"],
                "GATD": argsdict["gat_dropout"],
                "BS": batch_size,
                "lr": argsdict["lr"]
        }
        save_classification_metrics_val_best(model.test_metrics, val_metrics_best_path, epoch_type="Val_best", hyperparameters=hyperparameters)
        # logger.experiment.save(val_metrics_best_path)

    elif args.train_stage == "train_final" and args.eval:   
        trainer.test(model=model, dataloaders=test_loader, ckpt_path="last")

        # Save test_set y_pred and y_true
        preds_path = os.path.join(output_save_dir, "test_y_pred.csv")
        # print("model.test_output:", model.test_output)
        test_output_tensor = model.test_output[1]  # 获取键为1的张量
        if isinstance(test_output_tensor, torch.Tensor):
            if test_output_tensor.device.type != 'cpu':
                test_output_np = test_output_tensor.cpu().numpy()
            else:
                test_output_np = test_output_tensor.numpy()
        else:
            test_output_np = test_output_tensor
            
        # print("model.test_true:", model.test_true)
        true_path = os.path.join(output_save_dir, "test_y_true.csv")
        test_true_tensor = model.test_true[1]
        if isinstance(test_true_tensor, torch.Tensor): # 若是张量，判断设备并转换
            if test_true_tensor.device.type != 'cpu':
                test_true_np = test_true_tensor.cpu().numpy()
            else:
                test_true_np = test_true_tensor.numpy()
        else:   # 若已是NumPy数组，直接使用
            test_true_np = test_true_tensor
            
        # test_output_np = test_output_tensor.cpu().numpy() if test_output_tensor.device.type != 'cpu' else test_output_tensor.numpy()  ## 转换为 NumPy 数组（若在GPU上，需先移到CPU）
        # test_true_np = test_true_tensor.cpu().numpy() if test_true_tensor.device.type != 'cpu' else test_true_tensor.numpy()
        np.savetxt(preds_path, test_output_np, delimiter=",", fmt="%.6f")  # 预测值
        np.savetxt(true_path, test_true_np, delimiter=",", fmt="%.6f")    # 真实值
    
        # 定义测试集指标保存路径
        # print("model.test_metrics:", model.test_metrics)
        test_metrics_path = os.path.join(output_save_dir, "test_metrics.csv")
        save_classification_metrics(model.test_metrics, test_metrics_path, epoch_type="Test")

        # wandb disabled
        # logger.experiment.save(preds_path)
        # logger.experiment.save(true_path)
        # logger.experiment.save(test_metrics_path)
    
    
    # 保存train_tune阶段的验证集指标
    if args.train_stage == "train_tune": 
        # 训练结束后保存验证集指标
        val_metrics_path = os.path.join(output_save_dir, "val_metrics_all_epochs.csv")  # 验证集指标路径（与测试集同目录）
        # print("model.val_metrics:", model.val_metrics)
        save_classification_metrics_tra_val(model.val_metrics, val_metrics_path, epoch_type="Validation")
        
        # 训练结束后保存验证-测试集指标
        valtest_metrics_path = os.path.join(output_save_dir, "valtest_metrics_all_epochs.csv")  # 验证集指标路径（与测试集同目录）
        save_classification_metrics_tra_val(model.val_test_metrics, valtest_metrics_path, epoch_type="ValidationTest")
        
        # logger.experiment.save(val_metrics_path)
        # logger.experiment.save(valtest_metrics_path)
    
    # 保存任何阶段的训练集指标
    train_metrics_path = os.path.join(output_save_dir, "train_metrics_all_epochs.csv")  # 验证集指标路径（与测试集同目录）
    save_classification_metrics_tra_val(model.train_metrics, train_metrics_path, epoch_type="Train")
        
    # wandb disabled — CSV / JSON already written to output_save_dir
    # logger.experiment.save(train_metrics_path)
    # logger.experiment.save(config_json_path)
    # logger.experiment.finish()
    
    

if __name__ == "__main__":
    main()
