# 上一版模型为graph_models_ekan.py。这版模型添加了kan损失
import os
import torch
import torch_geometric
import numpy as np
import pytorch_lightning as pl
import bitsandbytes as bnb


from collections import defaultdict
from torch import nn
from torch.nn import functional as F
from torch.nn import Linear
from torch_geometric.nn import (
    # GCNConv,
    # PNAConv,
    GATConv,
    GATv2Conv,
    # GINConv,
    # GINEConv,
    global_mean_pool,
    # global_add_pool,
    # global_max_pool,
)
from torch_geometric.utils import degree
from tqdm.auto import tqdm
from pathlib import Path
from typing import Optional

from utils.norm_layers import BN
from utils.reporting import (
    get_cls_metrics_binary_pt,
    get_cls_metrics_multilabel_pt,
    get_cls_metrics_multiclass_pt,
)
from dartnet.efficient_kan import KAN

import torch
import pandas as pd
import numpy as np
import os

# ===================== 新增：定义特征保存的辅助函数 =====================
def save_combined_feat_to_csv(combined_feat, csv_path, batch_idx):
    """
    将单批次的combined_feat保存到CSV文件（追加模式）
    :param combined_feat: 形状为[batch, feat_dim]的torch张量（cuda/cpu）
    :param csv_path: 保存的CSV文件路径
    :param batch_idx: 批次索引（用于标注每行数据所属批次）
    """
    # 1. 将张量从CUDA移到CPU，转为numpy数组
    if combined_feat.is_cuda:
        feat_np = combined_feat.detach().cpu().numpy()
    else:
        feat_np = combined_feat.detach().numpy()
    
    # 2. 构建DataFrame：每一行是一个样本的特征，添加批次索引列
    # 生成特征列名（feat_0, feat_1, ..., feat_455）
    feat_cols = [f"feat_{i}" for i in range(feat_np.shape[1])]
    df_feat = pd.DataFrame(feat_np, columns=feat_cols)
    # 添加批次索引列（便于后续溯源）
    df_feat['batch_idx'] = batch_idx
    # 添加样本在批次内的索引列
    df_feat['sample_in_batch_idx'] = range(len(df_feat))
    
    # 3. 追加保存到CSV
    # 判断文件是否存在：不存在则写入表头，存在则追加（不写表头）
    header = not os.path.exists(csv_path)
    df_feat.to_csv(
        csv_path,
        mode='a',          # 追加模式
        header=header,    # 仅首次写入表头
        index=False,      # 不保存DataFrame的索引
        float_format='%.6f'  # 保留6位小数，保证精度
    )
    print(f"✅ 第{batch_idx}批次特征已追加保存至: {csv_path} (本批次样本数: {len(df_feat)})")



class MLP_morgan2048(nn.Module):
    def __init__(
        self,
        input_features: int = 2048,
        dnn_module_layer1_out: int = 100,
        dnn_module_layer2_out: int = 150,
        dnn_module_layer3_out: int = 150,
        dnn_module_layer4_out: int = 100,
        dnn_module_fp_final_out: int = 1,
        mode: str = "graph+fp+emb",
    ):
        super(MLP_morgan2048, self).__init__()
        self.activation = nn.ReLU()
        # mode kept for call-site compat; graph+fp+emb always returns layer4 features
        
        self.layer1 = nn.Linear(input_features, dnn_module_layer1_out)
        self.layer2 = nn.Linear(dnn_module_layer1_out, dnn_module_layer2_out)
        self.layer3 = nn.Linear(dnn_module_layer2_out, dnn_module_layer3_out)
        self.layer4 = nn.Linear(dnn_module_layer3_out, dnn_module_layer4_out)
        
        self.bn1 = nn.LayerNorm(dnn_module_layer1_out)
        self.bn2 = nn.LayerNorm(dnn_module_layer2_out)
        self.bn3 = nn.LayerNorm(dnn_module_layer3_out)
        self.bn4 = nn.LayerNorm(dnn_module_layer4_out)
        
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
    
    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.activation(self.bn1(self.layer1(x)))
        x = self.activation(self.bn2(self.layer2(x)))
        x = self.activation(self.bn3(self.layer3(x)))
        x = self.activation(self.bn4(self.layer4(x)))
        return x


class MLPemb(nn.Module):
    def __init__(
        self,
        input_features: int = 768,
        dnn_module_layer1_out: int = 100,
        dnn_module_layer2_out: int = 150,
        dnn_module_layer3_out: int = 150,
        dnn_module_layer4_out: int = 100,
        mode: str = "graph+fp",
    ):
        super(MLPemb, self).__init__()
        self.activation = nn.ReLU()  # 保留原有激活函数
        self.mode = mode
        
        # 1. 定义原有线性层（与原代码一致）
        self.layer1 = nn.Linear(input_features, dnn_module_layer1_out)
        self.layer2 = nn.Linear(dnn_module_layer1_out, dnn_module_layer2_out)
        self.layer3 = nn.Linear(dnn_module_layer2_out, dnn_module_layer3_out)
        self.layer4 = nn.Linear(dnn_module_layer3_out, dnn_module_layer4_out)
        
        # 2. 为每个线性层添加对应的LayerNorm层
        # 批归一化维度与线性层输出维度严格一致
        self.bn1 = nn.LayerNorm(dnn_module_layer1_out)
        self.bn2 = nn.LayerNorm(dnn_module_layer2_out)
        self.bn3 = nn.LayerNorm(dnn_module_layer3_out)
        self.bn4 = nn.LayerNorm(dnn_module_layer4_out)
        
        # 保留权重初始化函数
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # 线性层权重初始化（与原代码一致）
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # 新增：批归一化层的标准初始化
                nn.init.ones_(m.weight)  # 缩放因子γ初始化为1
                nn.init.zeros_(m.bias)   # 偏移量β初始化为0
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平输入特征（适配批次维度）
        
        # 前向传播流程：线性层 → 批归一化 → 激活函数（核心修改）
        x1 = self.layer1(x)
        x1 = self.bn1(x1)  # 线性层1后添加批归一化
        x1 = self.activation(x1)
        
        x2 = self.layer2(x1)
        x2 = self.bn2(x2)  # 线性层2后添加批归一化
        x2 = self.activation(x2)
        
        x3 = self.layer3(x2)
        x3 = self.bn3(x3)  # 线性层3后添加批归一化
        x3 = self.activation(x3)
        
        x4 = self.layer4(x3)
        x4 = self.bn4(x4)  # 线性层4后添加批归一化
        x4 = self.activation(x4)
        
        return x4


class GATorGATv2SC(nn.Module):
    def __init__(self, GATConvConstructor, in_channels, out_channels, attn_heads, concat, dropout, edge_dim):
        super(GATorGATv2SC, self).__init__()

        self.gnn_layer = GATConvConstructor(
                            in_channels=in_channels,
                            out_channels=out_channels,
                            heads=attn_heads,
                            concat=concat,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        )

    def forward(self, x, edge_index, edge_attr=None):
        try:
            return x + self.gnn_layer(x, edge_index, edge_attr=edge_attr)
        except:
            return self.gnn_layer(x, edge_index, edge_attr=edge_attr)


class GATorGATv2(pl.LightningModule):
    def __init__(
        self,
        gat_or_gatv2: str,
        in_channels: int,
        intermediate_dim: int,
        out_channels: int,
        num_layers: int,
        attn_heads: int,
        dropout: float,
        edge_dim: int = None,
        out_path: str = None,
    ):
        super(GATorGATv2, self).__init__()
        self.edge_dim = edge_dim

        assert gat_or_gatv2 in ["GAT", "GATv2"]
        GATConvConstructor = GATConv if gat_or_gatv2 == "GAT" else GATv2Conv

        self.node_input_mlp = nn.Sequential(
            nn.Linear(in_channels, intermediate_dim, bias=True),
            nn.Mish(),
            nn.Linear(intermediate_dim, intermediate_dim, bias=True),
        )

        if self.edge_dim is not None:
            self.edge_input_mlp = nn.Sequential(
                nn.Linear(edge_dim, edge_dim, bias=True),
                nn.Mish(),
                nn.Linear(edge_dim, edge_dim, bias=True),
            )

        modules = []

        seq_str = "x, edge_index, edge_attr -> x" if edge_dim else "x, edge_index -> x"

        for i in range(num_layers):
            if i == 0:
                modules.append(
                    (
                        GATorGATv2SC(
                            GATConvConstructor,
                            in_channels=intermediate_dim,
                            out_channels=intermediate_dim,
                            attn_heads=attn_heads,
                            concat=True,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        ),
                        seq_str,
                    )
                )

                modules.append(BN(intermediate_dim * attn_heads))

            elif i != num_layers - 1:
                modules.append(
                    (
                        GATorGATv2SC(
                            GATConvConstructor,
                            in_channels=intermediate_dim * attn_heads,
                            out_channels=intermediate_dim,
                            attn_heads=attn_heads,
                            concat=True,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        ),
                        seq_str,
                    )
                )

                modules.append(BN(intermediate_dim * attn_heads))

            else:
                modules.append(
                    (
                        GATorGATv2SC(
                            GATConvConstructor,
                            in_channels=intermediate_dim * attn_heads,
                            out_channels=intermediate_dim,
                            attn_heads=attn_heads,
                            concat=False,
                            dropout=dropout,
                            edge_dim=edge_dim,
                        ),
                        seq_str,
                    )
                )

                modules.append(BN(out_channels))

            modules.append(nn.Mish())

        if edge_dim:
            self.convs = torch_geometric.nn.Sequential("x, edge_index, edge_attr", modules)
        else:
            self.convs = torch_geometric.nn.Sequential("x, edge_index", modules)

    def forward(self, x, edge_index, edge_attr=None):
        x_proj = self.node_input_mlp(x)
        if edge_attr is not None:
            edge_attr_proj = self.edge_input_mlp(edge_attr)

        if edge_attr is not None:
            x_conv = self.convs(x_proj, edge_index, edge_attr=edge_attr_proj)
        else:
            x_conv = self.convs(x_proj, edge_index)

        return x_proj + x_conv


class DartNet(pl.LightningModule):
    def __init__(
        self,
        task_type: str,
        num_features: int,
        gnn_intermediate_dim: int = 256,
        output_node_dim: int = 256,
        batch_size: int = 32,
        lr: float = 1e-4,
        gat_attn_heads: int = 2,
        gat_dropout: float = 0.9,
        linear_output_size: int = 1,
        output_intermediate_dim: int = 768,
        scaler=None,
        monitor_loss_name: str = "Validation PRAUC",
        num_layers: int = 4,
        edge_dim: int = None,
        out_path: str = None,
        early_stopping_patience: int = 5,
        optimiser_weight_decay: float = 1e-10,
        gnn_feat_out_drop: float = 0.0,
        
        fingerprint_type: str = "morgan2048",  # 分子指纹类型
        fp_dim_morgan2048: int = 2048,  # 分子指纹维度
        fp_dim_morgan1024: int = 1024,
        fp_dim_maccs: int = 167,
        dnn_module_layer1_out: int = 100,
        dnn_module_layer2_out: int = 150,
        dnn_module_layer3_out: int = 150,
        dnn_module_layer4_out: int = 100,
        dnn_module_fp_final_out: int = 1,
        fp_feat_out_drop: float = 0.5,
        
        emb_feat_dim_input: int = 768,
        emb_feat_dim_out: int = 100,
        emb_feat_out_drop: float = 0.1,
        
        # KAN regularization (c603 defaults)
        kan_reg_config: dict = None,
        kan_grid_size: int = 5,
        kan_spline_order: int = 3,
        kan_hidden_dim: int = 64,
        kan_grid_range: list = None,
        cross_dim: int = 48,
        **kwargs,
    ):
        super().__init__()
        self.cross_dim = cross_dim
        assert task_type in ["binary_classification", "multi_classification"]

        self.edge_dim = edge_dim   #self将局部变量定义为类的实例属性，可以在类的其他方法中调用
        self.task_type = task_type
        self.num_features = num_features
        self.lr = lr
        self.batch_size = batch_size
        self.output_node_dim = output_node_dim
        self.gnn_intermediate_dim = gnn_intermediate_dim
        self.output_intermediate_dim = output_intermediate_dim
        self.num_layers = num_layers
        self.scaler = scaler
        self.linear_output_size = linear_output_size
        self.monitor_loss_name = monitor_loss_name
        self.out_path = out_path
        self.early_stopping_patience = early_stopping_patience
        self.optimiser_weight_decay = optimiser_weight_decay
        self.gat_attn_heads = gat_attn_heads
        self.gat_dropout = gat_dropout
        self.gnn_feat_out_drop = gnn_feat_out_drop
        
        # Parameters of the DNN Module
        self.fingerprint_type = fingerprint_type  # 分子指纹类型
        self.fp_dim_morgan2048 = fp_dim_morgan2048  # 分子指纹维度
        self.fp_dim_morgan1024 = fp_dim_morgan1024
        self.fp_dim_maccs = fp_dim_maccs
        self.dnn_module_layer1_out = dnn_module_layer1_out
        self.dnn_module_layer2_out = dnn_module_layer2_out
        self.dnn_module_layer3_out = dnn_module_layer3_out
        self.dnn_module_layer4_out = dnn_module_layer4_out
        self.dnn_module_fp_final_out = dnn_module_fp_final_out
        self.fp_feat_out_drop = fp_feat_out_drop
        
        self.emb_feat_dim_out = emb_feat_dim_out
        self.emb_feat_dim_input = emb_feat_dim_input
        self.emb_feat_out_drop = emb_feat_out_drop
        print("emb_feat_out_drop:", emb_feat_out_drop)
        print("emb_feat_dim_out:", emb_feat_dim_out)
        
        # 训练参数
        self.fixed_epochs = kwargs.get("fixed_epochs")
        
        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics。defaultdict 是 Python 标准库 collections 中的一种特殊字典，其特点是：当访问一个不存在的键时，会自动为该键创建默认值（此处为 list 空列表）
        self.train_output = defaultdict(list)
        self.val_output = defaultdict(list)
        self.val_test_output = defaultdict(list)
        self.test_output = defaultdict(list)

        self.val_preds = defaultdict(list)

        self.test_true = defaultdict(list)
        self.val_true = defaultdict(list)

        # Keep track of how many times we called test
        self.num_called_test = 1

        # Metrics per epoch (for train, valid); for test use above variable to register metrics per test-run
        self.train_metrics = {}
        self.val_metrics = {}
        self.val_test_metrics = {}
        self.test_metrics = {}

        # Holds final graphs embeddings
        self.test_graph_embeddings = defaultdict(list)
        self.val_graph_embeddings = defaultdict(list)
        self.train_graph_embeddings = defaultdict(list)

        # kan loss
        # 设置KAN正则化配置
        if kan_reg_config is None:
            kan_reg_config = {
                'use_regularization': True,
                'activation_weight': 1.0,
                'entropy_weight': 1.0,
                'overall_weight': 1e-5,  # train.py --kan-reg-overall-weight default
            }
        self.kan_reg_config = kan_reg_config
        self.kan_grid_size = kan_grid_size
        self.kan_spline_order = kan_spline_order
        self.kan_hidden_dim = kan_hidden_dim
        self.kan_grid_range = kan_grid_range or [-1, 1]

        ########################################################################
        # 1. GAT graph encoder (graph+fp+emb)
        ########################################################################
        gnn_args = dict(
            in_channels=num_features,
            out_channels=output_node_dim,
            intermediate_dim=gnn_intermediate_dim,
            num_layers=num_layers,
            out_path=out_path,
        )
        if self.edge_dim:
            gnn_args = gnn_args | dict(edge_dim=edge_dim)
        gnn_args = gnn_args | dict(attn_heads=gat_attn_heads, dropout=gat_dropout)
        self.gnn_model = GATorGATv2(gat_or_gatv2="GAT", **gnn_args)
        graph_feat_dim = output_node_dim
        self.gat_norm = nn.LayerNorm(graph_feat_dim)
        # self.gnn_dropout = nn.Dropout(p=gnn_feat_out_drop)

        ########################################################################
        # 2. Morgan fingerprint MLP
        ########################################################################
        dnn_args = dict(
            dnn_module_layer1_out=dnn_module_layer1_out,
            dnn_module_layer2_out=dnn_module_layer2_out,
            dnn_module_layer3_out=dnn_module_layer3_out,
            dnn_module_layer4_out=dnn_module_layer4_out,
            dnn_module_fp_final_out=dnn_module_fp_final_out,
            mode="graph+fp+emb",
        )
        self.morgan_mlp = MLP_morgan2048(input_features=fp_dim_morgan2048, **dnn_args)
        self.fp_dropout = nn.Dropout(p=fp_feat_out_drop)

        ########################################################################
        # 3. Smile embedding MLP
        ########################################################################
        self.emb_mlp = MLPemb(input_features=emb_feat_dim_input)
        self.smiles_emb_dropout = nn.Dropout(p=emb_feat_out_drop)

        ########################################################################
        # 4. Hybrid multimodal fusion module
        #    - Gated multimodal fusion branch (attention scoring + weighted concat)
        #    - Cross-modal interaction branch (low-dim pairwise Hadamard products)
        ########################################################################
        # Gated multimodal fusion branch: attention scoring network (456 -> 128 -> 3)
        self.attention = nn.Sequential(
            nn.Linear(graph_feat_dim + dnn_module_layer4_out + emb_feat_dim_out, 128),
            nn.ReLU(),
            nn.Linear(128, 3),
        )
        self.output_mlp_in_dim = graph_feat_dim + dnn_module_layer4_out + emb_feat_dim_out
        # Cross-modal interaction branch: project each modality to cross_dim
        self.cross_proj_gnn = nn.Linear(graph_feat_dim, self.cross_dim)
        self.cross_proj_fp = nn.Linear(dnn_module_layer4_out, self.cross_dim)
        self.cross_proj_emb = nn.Linear(emb_feat_dim_out, self.cross_dim)
        self.output_mlp_in_dim += 3 * self.cross_dim

        # LayerNorm on concatenated multimodal features before attention scoring
        attn_in_dim = graph_feat_dim + dnn_module_layer4_out + emb_feat_dim_out
        self.attn_input_norm = nn.LayerNorm(attn_in_dim)

        self.output_mlp = KAN(
            layers_hidden=[self.output_mlp_in_dim, self.kan_hidden_dim, linear_output_size],
            grid_size=self.kan_grid_size,
            spline_order=self.kan_spline_order,
            base_activation=nn.SiLU,
            grid_range=self.kan_grid_range,
        )

    def _cross_interact_tensor(self, gnn_out, morgan_feat, smiles_embedding):
        """Cross-modal interaction branch: project to shared subspace and form pairwise products."""
        g = self.cross_proj_gnn(gnn_out)
        f = self.cross_proj_fp(morgan_feat)
        e = self.cross_proj_emb(smiles_embedding)
        return torch.cat([g * f, g * e, f * e], dim=-1)

    def _build_weighted_cat(self, gnn_out, morgan_feat, smiles_embedding):
        """Gated multimodal fusion branch: LN + attention scoring + weighted concat."""
        raw_cat = torch.cat([gnn_out, morgan_feat, smiles_embedding], dim=-1)
        cat_feat = self.attn_input_norm(raw_cat)
        attn_weights = torch.softmax(self.attention(cat_feat), dim=-1)
        gnn_weighted = gnn_out * attn_weights[:, 0].unsqueeze(-1)
        morgan_weighted = morgan_feat * attn_weights[:, 1].unsqueeze(-1)
        smiles_weighted = smiles_embedding * attn_weights[:, 2].unsqueeze(-1)
        return torch.cat([gnn_weighted, morgan_weighted, smiles_weighted], dim=-1)

    def _fuse_attecat(self, gnn_out, morgan_feat, smiles_embedding):
        """Hybrid multimodal fusion module: gated branch || cross-modal interaction branch."""
        weighted_cat = self._build_weighted_cat(gnn_out, morgan_feat, smiles_embedding)
        interact = self._cross_interact_tensor(gnn_out, morgan_feat, smiles_embedding)
        combined_feat = torch.cat([weighted_cat, interact], dim=-1)
        return combined_feat

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        batch: torch.Tensor,
        edge_attr: Optional[torch.Tensor] = None,
        fingerprint: Optional[torch.Tensor] = None,  # 新增：接收分子指纹
        smiles_embedding: Optional[torch.Tensor] = None,
    ):
        x = x.float()
        edge_attr = edge_attr.float() if edge_attr is not None else None

        # 1. Obtain node embeddings (GAT)
        if self.edge_dim:
            z = self.gnn_model.forward(x, edge_index, edge_attr=edge_attr)
        else:
            z = self.gnn_model.forward(x, edge_index)

        # 2. Readout layer (global mean pooling)
        emb_avg_pool = global_mean_pool(z, batch)
        gnn_out = self.gat_norm(emb_avg_pool)
        # gnn_out = self.gnn_dropout(gnn_out)

        # 3. Morgan fingerprint
        batch_size = gnn_out.shape[0]
        fingerprint = fingerprint.view(batch_size, -1)
        morgan_feat = self.morgan_mlp(fingerprint)
        morgan_feat = self.fp_dropout(morgan_feat)

        # 4. Smile embedding
        smiles_embedding = smiles_embedding.view(batch_size, -1)
        smiles_embedding = self.emb_mlp(smiles_embedding)
        smiles_embedding = self.smiles_emb_dropout(smiles_embedding)

        # 5. Hybrid multimodal fusion module
        combined_feat = self._fuse_attecat(gnn_out, morgan_feat, smiles_embedding)

        # 6. Final KAN classifier
        if self.training:
            kan_output = self.output_mlp(combined_feat, update_grid=True)
        else:
            kan_output = self.output_mlp(combined_feat, update_grid=False)
        predictions = torch.flatten(kan_output)

        return z, emb_avg_pool, predictions

        
    def configure_optimizers(self):
        opt = bnb.optim.AdamW8bit(self.parameters(), lr=self.lr, weight_decay=self.optimiser_weight_decay)

        # mode = "max" if any(metric in self.monitor_loss_name for metric in ("AUROC", "MCC")) else "min"
        mode = "min" if "loss" in self.monitor_loss_name.lower() else "max"
        
        opt_dict = {
            "optimizer": opt,
            "monitor": self.monitor_loss_name,
        }

        sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode=mode, factor=0.5, patience=self.early_stopping_patience // 2, verbose=True
        )
        
        # if self.monitor_loss_name != "train_loss":
        #     opt_dict["lr_scheduler"] = sched

        return opt_dict
    

    def _batch_loss(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        y: Optional[torch.Tensor] = None,
        batch_mapping: Optional[torch.Tensor] = None,
        edge_attr: Optional[torch.Tensor] = None,
        fingerprint: Optional[torch.Tensor] = None,
        smiles_embedding: Optional[torch.Tensor] = None,
        step_type: str = None,
    ):
        # Forward pass (graph_embeddings not used here so far, after forward)
        z, graph_embeddings, predictions = self.forward(x, edge_index, batch_mapping, edge_attr=edge_attr, fingerprint=fingerprint, smiles_embedding=smiles_embedding)
        # print("predictions.shape:", predictions.shape)
        # print("y.shape:", y.shape)

        # 任务损失（分类）
        task_loss = F.binary_cross_entropy_with_logits(predictions.float(), y.float())
        
        # KAN正则化损失（仅在训练时且启用时）
        if step_type == "train" and self.kan_reg_config['use_regularization']:
            kan_reg_loss = self.output_mlp.regularization_loss(
                regularize_activation=self.kan_reg_config['activation_weight'],
                regularize_entropy=self.kan_reg_config['entropy_weight']
            )
            
            total_loss = task_loss + self.kan_reg_config['overall_weight'] * kan_reg_loss
            
            # 记录各项损失
            self.log(f"{step_type}_task_loss", task_loss, batch_size=self.batch_size)
            self.log(f"{step_type}_kan_reg_loss", kan_reg_loss, batch_size=self.batch_size)
            self.log(f"{step_type}_total_loss", total_loss, batch_size=self.batch_size)
        else:
            total_loss = task_loss
            if step_type != "train":
                self.log(f"{step_type}_loss", total_loss, batch_size=self.batch_size)
        
        return total_loss, z, graph_embeddings, predictions, y


    def _step(self, batch: torch.Tensor, step_type: str):
        assert step_type in ["train", "validation", "test", "validation_test"]

        x, edge_index, y, batch_mapping, edge_attr, fingerprint, smiles_embedding=batch.x, batch.edge_index, batch.y, batch.batch, batch.edge_attr, batch.fingerprint, batch.smiles_embedding

        total_loss, z, graph_embeddings, predictions, y = self._batch_loss(
            x, edge_index, y, batch_mapping, edge_attr=edge_attr, fingerprint=fingerprint, smiles_embedding=smiles_embedding, step_type=step_type,
        )

        output = (predictions, y)

        if step_type == "train":
            self.train_output[self.current_epoch].append(output)   # self.current_epoch是pl.LightningModule的内置属性，无需初始化中手动定义
        elif step_type == "validation":
            self.val_output[self.current_epoch].append(output)   # 将每个batch的输出output1 = (predictions1, labels1)添加到列表中
        elif step_type == "validation_test":
            self.val_test_output[self.current_epoch].append(output)
        elif step_type == "test":
            self.test_output[self.num_called_test].append(output)

        return total_loss


    def training_step(self, batch: torch.Tensor, batch_idx: int):
        train_loss = self._step(batch, "train")

        self.log("train_loss", train_loss, prog_bar=True, batch_size=self.batch_size)

        return train_loss


    def validation_step(self, batch: torch.Tensor, batch_idx: int, dataloader_idx: int = 0):
        if dataloader_idx == 0:
            val_loss = self._step(batch, "validation")

            self.log("val_loss", val_loss, batch_size=self.batch_size)

            return val_loss

        if dataloader_idx == 1:
            val_test_loss = self._step(batch, "validation_test")

            self.log("val_test_loss", val_test_loss, batch_size=self.batch_size)

            return val_test_loss


    def test_step(self, batch: torch.Tensor, batch_idx: int):
        test_loss = self._step(batch, "test")

        self.log("test_loss", test_loss, batch_size=self.batch_size)

        return test_loss

    # 在DartNet类中添加以下方法
    def predict_step(self, batch, batch_idx):
        """定义推理步骤，明确传递图数据参数"""
        # 从batch中提取图数据的关键参数（与训练时的forward参数一致）
        smiles_id = batch.smiles_id
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        fingerprint = batch.fingerprint  
        smiles_embedding = batch.smiles_embedding  
        batch = batch.batch  
        print("smiles_id:", smiles_id)
        # print("Batch内容:", batch)  # 查看是否包含smiles_id
        # 调用模型的forward方法，传入所有必需参数
        output = self.forward(x, edge_index, batch, edge_attr=edge_attr, fingerprint=fingerprint, smiles_embedding=smiles_embedding)  # 与你的forward签名匹配
        print("output:", output)
        
        # 根据模型输出格式，返回预测结果（这里假设output是元组，第三个元素是预测值）
        if isinstance(output, tuple):
            return output[2]  # 对应原代码中的predictions = output[2]
        return output

    def _epoch_end_report(self, epoch_outputs, epoch_type):
        assert epoch_type in ["Train", "Validation", "Test", "ValidationTest"]

        y_pred = torch.cat([item[0] for item in epoch_outputs], dim=0)
        y_true = torch.cat([item[1] for item in epoch_outputs], dim=0)
        # print("y_pred:", y_pred)
        # print("y_true:", y_true)
        if self.scaler:
            if self.linear_output_size > 1:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, self.linear_output_size))
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, self.linear_output_size))
            else:
                y_pred = self.scaler.inverse_transform(y_pred.reshape(-1, 1)).flatten()
                y_true = self.scaler.inverse_transform(y_true.reshape(-1, 1)).flatten()

            y_pred = torch.from_numpy(y_pred)
            y_true = torch.from_numpy(y_true)

        if self.task_type == "binary_classification" and self.linear_output_size == 1:
            # print("y_pred:", y_pred)
            y_pred = torch.sigmoid(y_pred)
            # print("y_pred:", y_pred)
            metrics = get_cls_metrics_binary_pt(y_true, y_pred)

            self.log(f"{epoch_type} AUROC", metrics[0], batch_size=self.batch_size)
            self.log(f"{epoch_type} PRAUC", metrics[1], batch_size=self.batch_size)
            self.log(f"{epoch_type} MCC", metrics[2], batch_size=self.batch_size)
            self.log(f"{epoch_type} Accuracy", metrics[3], batch_size=self.batch_size)
            self.log(f"{epoch_type} F1", metrics[4], batch_size=self.batch_size)
            tp, tn, fp, fn = metrics[5], metrics[6], metrics[7], metrics[8]
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            specificity = tn / (tn + fp) if (tn + fp) > 0 else 0.0
            self.log(
                f"{epoch_type} Sensitivity",
                torch.tensor(float(sensitivity), dtype=torch.float32),
                batch_size=self.batch_size,
            )
            self.log(
                f"{epoch_type} Specificity",
                torch.tensor(float(specificity), dtype=torch.float32),
                batch_size=self.batch_size,
            )


        return metrics, y_pred, y_true


    def on_train_epoch_end(self):
        self.train_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
            self.train_output[self.current_epoch], epoch_type="Train"
        )

        del y_pred
        del y_true
        del self.train_output[self.current_epoch]


    def on_validation_epoch_end(self):
        if len(self.val_output[self.current_epoch]) > 0:
            self.val_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_output[self.current_epoch], epoch_type="Validation"
            )

            del y_pred
            del y_true
            del self.val_output[self.current_epoch]

        if len(self.val_test_output[self.current_epoch]) > 0:
            self.val_test_metrics[self.current_epoch], y_pred, y_true = self._epoch_end_report(
                self.val_test_output[self.current_epoch], epoch_type="ValidationTest"
            )

            del y_pred
            del y_true
            del self.val_test_output[self.current_epoch]


    def on_test_epoch_end(self):
        test_outputs_per_epoch = self.test_output[self.num_called_test]
        self.test_metrics[self.num_called_test], y_pred, y_true = self._epoch_end_report(
            test_outputs_per_epoch, epoch_type="Test"
        )
        self.test_output[self.num_called_test] = y_pred
        self.test_true[self.num_called_test] = y_true

        self.num_called_test += 1
