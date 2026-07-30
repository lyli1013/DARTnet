# Previous model version: graph_models_ekan.py. This version adds KAN loss.
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

# ===================== Added: helper to save features =====================
def save_combined_feat_to_csv(combined_feat, csv_path, batch_idx):
    """
    Save one batch of combined_feat to a CSV file (append mode).
    :param combined_feat: torch tensor of shape [batch, feat_dim] (cuda/cpu)
    :param csv_path: CSV file path to save to
    :param batch_idx: batch index (to label which batch each row belongs to)
    """
    # 1. Move tensor from CUDA to CPU and convert to numpy
    if combined_feat.is_cuda:
        feat_np = combined_feat.detach().cpu().numpy()
    else:
        feat_np = combined_feat.detach().numpy()
    
    # 2. Build DataFrame: each row is one sample's features; add batch index column
    # Generate feature column names (feat_0, feat_1, ..., feat_N)
    feat_cols = [f"feat_{i}" for i in range(feat_np.shape[1])]
    df_feat = pd.DataFrame(feat_np, columns=feat_cols)
    # Add batch index column (for later tracing)
    df_feat['batch_idx'] = batch_idx
    # Add within-batch sample index column
    df_feat['sample_in_batch_idx'] = range(len(df_feat))
    
    # 3. Append-save to CSV
    # Write header only if file does not exist; otherwise append without header
    header = not os.path.exists(csv_path)
    df_feat.to_csv(
        csv_path,
        mode='a',          # Append mode
        header=header,    # Write header only on first write
        index=False,      # Do not save DataFrame index
        float_format='%.6f'  # Keep 6 decimal places for precision
    )
    print(f"[OK] Batch {batch_idx} features appended to: {csv_path} (samples in batch: {len(df_feat)})")



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
        self.activation = nn.ReLU()  # Keep original activation
        self.mode = mode
        
        # 1. Define original linear layers (same as prior code)
        self.layer1 = nn.Linear(input_features, dnn_module_layer1_out)
        self.layer2 = nn.Linear(dnn_module_layer1_out, dnn_module_layer2_out)
        self.layer3 = nn.Linear(dnn_module_layer2_out, dnn_module_layer3_out)
        self.layer4 = nn.Linear(dnn_module_layer3_out, dnn_module_layer4_out)
        
        # 2. Add a LayerNorm after each linear layer
        # Norm dimension strictly matches linear layer output dim
        self.bn1 = nn.LayerNorm(dnn_module_layer1_out)
        self.bn2 = nn.LayerNorm(dnn_module_layer2_out)
        self.bn3 = nn.LayerNorm(dnn_module_layer3_out)
        self.bn4 = nn.LayerNorm(dnn_module_layer4_out)
        
        # Keep weight initialization
        self._initialize_weights()
        
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                # Linear weight init (same as prior code)
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.LayerNorm):
                # Added: standard LayerNorm initialization
                nn.init.ones_(m.weight)  # Scale factor gamma init to 1
                nn.init.zeros_(m.bias)   # Bias beta init to 0
    
    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten input features (batch-aware)
        
        # Forward: linear -> LayerNorm -> activation (core change)
        x1 = self.layer1(x)
        x1 = self.bn1(x1)  # LayerNorm after linear 1
        x1 = self.activation(x1)
        
        x2 = self.layer2(x1)
        x2 = self.bn2(x2)  # LayerNorm after linear 2
        x2 = self.activation(x2)
        
        x3 = self.layer3(x2)
        x3 = self.bn3(x3)  # LayerNorm after linear 3
        x3 = self.activation(x3)
        
        x4 = self.layer4(x3)
        x4 = self.bn4(x4)  # LayerNorm after linear 4
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
        
        fingerprint_type: str = "morgan2048",  # Molecular fingerprint type
        fp_dim_morgan2048: int = 2048,  # Molecular fingerprint dimension
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

        self.edge_dim = edge_dim   # Bind local var as instance attribute for use in other methods
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
        self.fingerprint_type = fingerprint_type  # Molecular fingerprint type
        self.fp_dim_morgan2048 = fp_dim_morgan2048  # Molecular fingerprint dimension
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
        
        # Training parameters
        self.fixed_epochs = kwargs.get("fixed_epochs")
        
        # Store model outputs per epoch (for train, valid) or test run; used to compute the reporting metrics. defaultdict auto-creates a default (here: empty list) for missing keys.
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
        # Set KAN regularization config
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
        fingerprint: Optional[torch.Tensor] = None,  # Added: accept molecular fingerprint
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

        # Task loss (classification)
        task_loss = F.binary_cross_entropy_with_logits(predictions.float(), y.float())
        
        # KAN regularization loss (only during training when enabled)
        if step_type == "train" and self.kan_reg_config['use_regularization']:
            kan_reg_loss = self.output_mlp.regularization_loss(
                regularize_activation=self.kan_reg_config['activation_weight'],
                regularize_entropy=self.kan_reg_config['entropy_weight']
            )
            
            total_loss = task_loss + self.kan_reg_config['overall_weight'] * kan_reg_loss
            
            # Log individual loss terms
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
            self.train_output[self.current_epoch].append(output)   # self.current_epoch is a built-in pl.LightningModule attribute
        elif step_type == "validation":
            self.val_output[self.current_epoch].append(output)   # Append each batch output (predictions, labels) to the list
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

    # The following method is part of DartNet
    def predict_step(self, batch, batch_idx):
        """Define predict step; explicitly pass graph data arguments."""
        # Extract key graph args from batch (same as training forward args)
        smiles_id = batch.smiles_id
        x = batch.x
        edge_index = batch.edge_index
        edge_attr = batch.edge_attr
        fingerprint = batch.fingerprint  
        smiles_embedding = batch.smiles_embedding  
        batch = batch.batch  
        print("smiles_id:", smiles_id)
        # print("Batch contents:", batch)  # Check whether smiles_id is present
        # Call model forward with all required arguments
        output = self.forward(x, edge_index, batch, edge_attr=edge_attr, fingerprint=fingerprint, smiles_embedding=smiles_embedding)  # Matches forward signature
        print("output:", output)
        
        # Return predictions based on output format (assume tuple; 3rd element is predictions)
        if isinstance(output, tuple):
            return output[2]  # Corresponds to predictions = output[2] in original code
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
