# from argparse import Namespace
# import yaml
# import torch
# import pandas as pd
# from rdkit import Chem
# from tokenizer.tokenizer import MolTranBertTokenizer
# from train_pubchem_light import LightningModule
# from fast_transformers.masking import LengthMask as LM


# # -------------------------- 工具函数（内部使用）--------------------------
# def batch_split(data, batch_size=64):
#     """按批次分割数据，避免内存溢出"""
#     i = 0
#     while i < len(data):
#         yield data[i:min(i + batch_size, len(data))]
#         i += batch_size

# def canonicalize(s):
#     """SMILES 标准化：统一格式，去除异构体标记"""
#     return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)


# def add_embedding_to_data(
#     data_list,  # 输入：带smiles的Data对象列表（与_add_fingerprint_to_data的输入格式一致）
#     hparams_path="/mnt/data/phd/lly/02_Project/PrismNet/embedding/Pretrained MoLFormer/hparams.yaml",
#     vocab_path="bert_vocab.txt",
#     ckpt_path="/mnt/data/phd/lly/02_Project/PrismNet/embedding/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt",
#     batch_size=64,
#     device="cuda" if torch.cuda.is_available() else "cpu"
# ):
#     """
#     为Data对象列表计算并添加SMILES嵌入（与_add_fingerprint_to_data保持一致的输入输出格式）
    
#     参数：
#         data_list: list[Data] - 含smiles属性的Data对象列表（同_add_fingerprint_to_data的输入）
#         其他参数：同之前的get_smiles_embedding
    
#     返回：
#         list[Data] - 含smiles_embedding属性的Data对象列表（移除无效SMILES样本）
#     """
#     # 1. 提取SMILES并过滤无效样本（与指纹函数逻辑对齐）
#     smiles_list = [data.smiles for data in data_list]
#     print(f"待处理样本数(生成SMILES嵌入): {len(smiles_list)}")

#     # 2. 标准化SMILES（同时记录无效样本索引）
#     valid_indices = []  # 有效样本在原data_list中的索引
#     valid_smiles = []   # 有效SMILES列表
#     invalid_smiles = [] # 无效SMILES列表

#     for idx, s in enumerate(smiles_list):
#         canon_s = canonicalize(s)
#         if canon_s is not None:
#             valid_indices.append(idx)
#             valid_smiles.append(canon_s)
#         else:
#             invalid_smiles.append(s)

#     print(f"embedding有效样本数: {len(valid_smiles)}, embedding无效样本数: {len(invalid_smiles)}")
#     if len(invalid_smiles) > 0:
#         print(f"embedding无效SMILES示例: {invalid_smiles[:5]}")

#     # 3. 加载模型和分词器
#     with open(hparams_path, 'r') as f:
#         config = Namespace(**yaml.safe_load(f))
#     tokenizer = MolTranBertTokenizer(vocab_path)
#     lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt_path, config=config, vocab=tokenizer.vocab)
#     lm = lm.to(device)
#     lm.eval()

#     # 4. 批量生成嵌入（仅处理有效样本）
#     embeddings = []
#     with torch.no_grad():
#         for batch in batch_split(valid_smiles, batch_size=batch_size):
#             batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
#             idx = torch.tensor(batch_enc['input_ids'], device=device)
#             mask = torch.tensor(batch_enc['attention_mask'], device=device)
            
#             token_embeddings = lm.blocks(lm.tok_emb(idx), length_mask=LM(mask.sum(-1)))
#             input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
#             sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
#             sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
#             batch_emb = sum_embeddings / sum_mask
#             embeddings.append(batch_emb.cpu())

#     # 合并嵌入结果（形状：[有效样本数, 768]）
#     valid_embeddings = torch.cat(embeddings, dim=0)

#     # 5. 为有效样本添加嵌入属性（保持与原data_list的顺序对应）
#     data_with_emb = []
#     for idx, emb in zip(valid_indices, valid_embeddings):
#         # 从原data_list中取出对应Data对象，添加嵌入属性
#         data = data_list[idx]
#         data.smiles_embedding = emb  # 新增smiles_embedding字段
#         data_with_emb.append(data)

#     # 验证：有效样本数与输出Data列表长度一致
#     assert len(data_with_emb) == len(valid_smiles), "嵌入与Data对象匹配失败，数量不对应"
#     return data_with_emb





# # 在文件顶部添加（必须在 from embedding import ... 之前）
# import sys
# import os

# # 填写 embedding.py 所在的文件夹路径（注意是文件夹路径，不是文件路径）
# embedding_dir = "/home/phd/lly/02_Project/PrismNet/embedding/molformer-main/notebooks/pretrained_molformer"  # 替换为你的实际路径

# # 将路径添加到 Python 搜索路径
# if embedding_dir not in sys.path:
#     sys.path.append(embedding_dir)
#     print(f"已添加嵌入模块路径：{embedding_dir}")
import torch
import pandas as pd
from rdkit import Chem
from argparse import Namespace
import yaml
import torch
import time  # 导入计时模块
from rdkit import Chem
from tokenizer.tokenizer import MolTranBertTokenizer
from train_pubchem_light import LightningModule
from fast_transformers.masking import LengthMask as LM


def batch_split(data, batch_size=64):
    i = 0
    while i < len(data):
        yield data[i:min(i + batch_size, len(data))]
        i += batch_size


def canonicalize(s):
    try:
        return Chem.MolToSmiles(Chem.MolFromSmiles(s), canonical=True, isomericSmiles=False)
    except:
        return None


def add_embedding_to_data(
    data_list,
    hparams_path="/mnt/data/phd/lly/02_Project/PrismNet/embedding/Pretrained MoLFormer/hparams.yaml",
    vocab_path="bert_vocab.txt",
    ckpt_path="/mnt/data/phd/lly/02_Project/PrismNet/embedding/Pretrained MoLFormer/checkpoints/N-Step-Checkpoint_3_30000.ckpt",
    batch_size=64,
    device="cuda" if torch.cuda.is_available() else "cpu"
):
    # -------------------------- 新增：记录开始时间 --------------------------
    start_time = time.time()  # 单位：秒
    print(f"嵌入处理开始时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(start_time))}")

    # 1. 提取SMILES并过滤无效样本
    smiles_list = [data.smiles for data in data_list]
    print(f"待处理样本数(生成SMILES嵌入): {len(smiles_list)}")

    # 2. 标准化SMILES
    valid_indices = []
    valid_smiles = []
    invalid_smiles = []
    for idx, s in enumerate(smiles_list):
        canon_s = canonicalize(s)
        if canon_s is not None:
            valid_indices.append(idx)
            valid_smiles.append(canon_s)
        else:
            invalid_smiles.append(s)

    print(f"嵌入有效样本数: {len(valid_smiles)}, 嵌入无效样本数: {len(invalid_smiles)}")
    if len(invalid_smiles) > 0:
        print(f"嵌入无效SMILES示例: {invalid_smiles[:5]}")

    # 3. 加载模型和分词器
    with open(hparams_path, 'r') as f:
        config = Namespace(**yaml.safe_load(f))
    tokenizer = MolTranBertTokenizer(vocab_path)
    lm = LightningModule(config, tokenizer.vocab).load_from_checkpoint(ckpt_path, config=config, vocab=tokenizer.vocab)
    lm = lm.to(device)
    lm.eval()

    # 4. 批量生成嵌入
    print(f"开始生成嵌入（批次大小：{batch_size}，设备：{device}）...")
    embeddings = []
    with torch.no_grad():
        for batch in batch_split(valid_smiles, batch_size=batch_size):
            batch_enc = tokenizer.batch_encode_plus(batch, padding=True, add_special_tokens=True)
            idx = torch.tensor(batch_enc['input_ids'], device=device)
            mask = torch.tensor(batch_enc['attention_mask'], device=device)
            
            token_embeddings = lm.blocks(lm.tok_emb(idx), length_mask=LM(mask.sum(-1)))
            input_mask_expanded = mask.unsqueeze(-1).expand(token_embeddings.size()).float()
            sum_embeddings = torch.sum(token_embeddings * input_mask_expanded, dim=1)
            sum_mask = torch.clamp(input_mask_expanded.sum(dim=1), min=1e-9)
            batch_emb = sum_embeddings / sum_mask
            embeddings.append(batch_emb.cpu())

    valid_embeddings = torch.cat(embeddings, dim=0)

    # 5. 为有效样本添加嵌入属性
    data_with_emb = []
    for idx, emb in zip(valid_indices, valid_embeddings):
        data = data_list[idx]
        data.smiles_embedding = emb
        data_with_emb.append(data)

    # -------------------------- 新增：计算并输出耗时 --------------------------
    end_time = time.time()
    total_time = end_time - start_time  # 总耗时（秒）
    # 格式化为“分:秒”，方便阅读
    minutes = int(total_time // 60)
    seconds = round(total_time % 60, 2)
    print(f"\n=== 嵌入处理完成 ===")
    print(f"嵌入处理结束时间: {time.strftime('%Y-%m-%d %H:%M:%S', time.localtime(end_time))}")
    print(f"嵌入总耗时: {minutes}分{seconds}秒（共{total_time:.2f}秒）")
    print(f"平均每个有效样本耗时: {total_time/len(valid_smiles):.4f}秒")  # 新增：平均耗时（可选）

    assert len(data_with_emb) == len(valid_smiles), "嵌入与Data对象匹配失败，数量不对应"
    return data_with_emb




# -------------------------- 1. 处理原始CSV数据，生成嵌入 --------------------------
# 读取原始CSV（含 FeatureIndex、Smiles 列）
df = pd.read_csv('/home/phd/lly/02_Project/PrismNet/00data_SMD/data_smallMoleculer_lib/00database_merged/merged_all_small_mol_deduplicated_example.csv')

# 定义 Data 类（与你的 train_data 中 Data 结构一致，仅需 smiles 和 embedding）
class Data:
    def __init__(self, smiles, smiles_embedding=None):
        self.smiles = smiles  # 原始SMILES（用于后续匹配）
        self.smiles_embedding = smiles_embedding  # 嵌入向量

# 将 CSV 转换为 Data 列表（仅含 smiles，用于生成嵌入）
csv_data_list = [Data(smiles=row["Smiles"]) for _, row in df.iterrows()]

# 生成嵌入（复用之前的 add_embedding_to_data 函数，自动过滤无效SMILES）
csv_data_with_emb = add_embedding_to_data(
    data_list=csv_data_list,
    batch_size=32,
    device="cuda" if torch.cuda.is_available() else "cpu"
)

# -------------------------- 2. 构建 SMILES-嵌入 映射并保存为 .pt 文件 --------------------------
# 构建映射：key=原始SMILES，value=嵌入向量（便于后续快速匹配）
smile_to_embedding = {
    data.smiles: data.smiles_embedding 
    for data in csv_data_with_emb
}

# 保存映射到 .pt 文件（直接存字典，加载后可直接用）
save_path = '/home/phd/lly/02_Project/PrismNet/00data_SMD/data_smallMoleculer_lib/00database_merged/smiles_embedding_map.pt'
torch.save(smile_to_embedding, save_path)
print(f"SMILES-嵌入映射已保存到 {save_path}")
print(f"有效映射数：{len(smile_to_embedding)}（每个SMILES对应一个768维嵌入）")






import torch
# 假设你的 Data 类已在代码中定义（含 y、smiles、fingerprint 等属性）


def add_embedding_from_pt(train_data, emb_pt_path):
    """
    从 .pt 文件加载 SMILES-嵌入映射，按 SMILES 匹配到 train_data 中
    
    参数：
        train_data: list[Data] - 你的训练集 Data 列表（已含 fingerprint）
        emb_pt_path: str - 保存 SMILES-嵌入映射的 .pt 文件路径
    
    返回：
        list[Data] - 已添加 smiles_embedding 的 train_data（过滤无匹配嵌入的样本）
    """
    # -------------------------- 1. 加载 .pt 文件中的 SMILES-嵌入映射 --------------------------
    print(f"\n=== 从 {emb_pt_path} 加载 SMILES-嵌入映射 ===")
    try:
        smile_to_embedding = torch.load(emb_pt_path)
    except Exception as e:
        raise ValueError(f"加载 .pt 文件失败：{str(e)}") from e
    print(f"加载到 {len(smile_to_embedding)} 个 SMILES-嵌入映射")

    # -------------------------- 2. 按 SMILES 匹配并添加嵌入 --------------------------
    train_data_with_emb = []
    no_match_smiles = []  # 记录无匹配嵌入的SMILES（便于调试）

    for data in train_data:
        # 按 Data 对象的 smiles 字段匹配嵌入
        if data.smiles in smile_to_embedding:
            # 添加嵌入到 Data 对象
            data.smiles_embedding = smile_to_embedding[data.smiles]
            train_data_with_emb.append(data)
        else:
            no_match_smiles.append(data.smiles)

    # -------------------------- 3. 输出匹配结果 --------------------------
    print(f"train_data 原始样本数：{len(train_data)}")
    print(f"成功添加嵌入的样本数：{len(train_data_with_emb)}")
    if len(no_match_smiles) > 0:
        print(f"无匹配嵌入的样本数：{len(no_match_smiles)}，示例：{no_match_smiles[:5]}")

    return train_data_with_emb


# -------------------------- 你的原有代码中调用 --------------------------
# 4. 从 .pt 文件加载嵌入并添加到 train_data（替换之前的 add_embedding_to_data 调用）
# 无需再导入 add_embedding_to_data，直接调用上面的函数
emb_pt_path = '/home/phd/lly/02_Project/PrismNet/00data_SMD/data_smallMoleculer_lib/00database_merged/smiles_embedding_map.pt'
train_data = add_embedding_from_pt(
    train_data=train_data,  # 你的原始 train_data（已含 fingerprint）
    emb_pt_path=emb_pt_path
)

# 验证结果
print("\n=== 嵌入添加验证 ===")
print("train_data[0] (含 embedding)：", train_data[0])
# 输出示例：Data(y=1, smiles='CCOc1ncc...', fingerprint=[2048], smiles_embedding=tensor([...]))
print("train_data[0] 嵌入形状：", train_data[0].smiles_embedding.shape)  # torch.Size([768])
print("train_data[1600] 嵌入形状：", train_data[1600].smiles_embedding.shape)  # torch.Size([768])