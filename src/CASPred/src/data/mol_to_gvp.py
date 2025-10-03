import torch
from rdkit import Chem
from rdkit.Chem import AllChem
import numpy as np

from .smile_to_3D import smiles_to_3d_conformer

# --- 特征提取函数 ---

def get_atom_features(atom: Chem.Atom) -> np.ndarray: # 31 维
    """
    为单个原子生成丰富的标量特征向量。
    
    特征包括:
    - 原子符号 (One-hot)
    - 形式电荷 (数值)
    - 杂化类型 (One-hot)
    - 是否为芳香性 (布尔值)
    - 连接的氢原子数 (One-hot)
    - 是否在环中 (布尔值)
    """
    # 原子符号 (eg: C, N, O, F, ...)
    possible_atom_symbols = ['B', 'Br', 'C', 'Ca', 'Cl', 'F', 'Fe', 'H', 'I', 'K', 'Li', 'Mg', 'N', 'Na', 'O', 'P', 'S', 'Se', 'Sn', 'Zn']
    atom_symbol = [1 if atom.GetSymbol() == s else 0 for s in possible_atom_symbols]
    
    # 杂化类型
    possible_hybridizations = [Chem.rdchem.HybridizationType.SP, Chem.rdchem.HybridizationType.SP2, Chem.rdchem.HybridizationType.SP3]
    hybridization = [1 if atom.GetHybridization() == h else 0 for h in possible_hybridizations]
    
    # 连接的氢原子数
    num_hs = [1 if atom.GetTotalNumHs() == i else 0 for i in range(5)] # 0-4
    
    # 其他数值/布尔特征
    formal_charge = atom.GetFormalCharge()
    is_aromatic = atom.GetIsAromatic()
    is_in_ring = atom.IsInRing()

    features = np.array(
        atom_symbol + 
        [formal_charge] + 
        hybridization + 
        [is_aromatic, is_in_ring] +
        num_hs
    )
    return features

def get_bond_features(bond: Chem.Bond) -> np.ndarray: # 6 维
    """
    为单个化学键生成标量特征向量。
    
    特征包括:
    - 键类型 (One-hot)
    - 是否共轭 (布尔值)
    - 是否在环中 (布尔值)
    """
    # 键类型
    bond_type_to_idx = {
        Chem.rdchem.BondType.SINGLE: 0,
        Chem.rdchem.BondType.DOUBLE: 1,
        Chem.rdchem.BondType.TRIPLE: 2,
        Chem.rdchem.BondType.AROMATIC: 3,
    }
    bond_type = [1 if bond.GetBondType() == bt else 0 for bt in bond_type_to_idx.keys()]

    # 共轭
    is_conjugated = bond.GetIsConjugated()

    # 在环中
    is_in_ring = bond.IsInRing()
    
    features = np.array(
        bond_type + 
        [is_conjugated, is_in_ring]
    )
    return features


def rbf_encode_dist(dist, num_rbf=16, min_dist=0.0, max_dist=5.0):
    """
    使用径向基函数 (RBF) 对距离进行编码。
    """
    dist_range = np.linspace(min_dist, max_dist, num_rbf)
    # sigma 控制了基函数的宽度
    sigma = (max_dist - min_dist) / num_rbf
    rbf = np.exp(-((dist.reshape(-1, 1) - dist_range) ** 2) / (2 * sigma ** 2))
    return rbf

# --- 主转换函数 ---

def mol_to_gvp_graph(mol_3d: Chem.Mol, num_rbf=16):
    """
    将 RDKit 3D 分子对象转换为 GVP 模型所需的、包含完整特征的图数据格式。
    """
    if not mol_3d or mol_3d.GetNumConformers() == 0:
        return None

    conformer = mol_3d.GetConformer(0)
    coords = torch.tensor(conformer.GetPositions(), dtype=torch.float32)

    # --- 1. 节点特征 ---
    # 节点标量特征 (node_s)
    node_s_list = [get_atom_features(atom) for atom in mol_3d.GetAtoms()]
    node_s = torch.tensor(np.vstack(node_s_list), dtype=torch.float32)

    # 节点向量特征 (node_v) - 原子坐标
    node_v = coords.unsqueeze(1)  # Shape: [num_nodes, 1, 3]

    # --- 2. 边特征 ---
    edge_index_list = []
    edge_s_list = []
    edge_v_list = []
    
    # 临时存储键长用于 RBF 编码
    bond_distances = []

    for bond in mol_3d.GetBonds():
        # a. 获取索引和构建双向边
        start_idx, end_idx = bond.GetBeginAtomIdx(), bond.GetEndAtomIdx()
        edge_index_list.extend([[start_idx, end_idx], [end_idx, start_idx]])

        # b. 提取基础边标量特征 (化学属性)
        bond_feats = get_bond_features(bond)
        edge_s_list.extend([bond_feats, bond_feats]) # 双向边使用相同特征
        
        # c. 计算相对位置向量 (边向量特征)
        start_pos, end_pos = coords[start_idx], coords[end_idx]
        relative_pos = end_pos - start_pos
        edge_v_list.extend([relative_pos, -relative_pos]) # B->A 是 A->B 的反向
        
        # d. 存储键长
        distance = torch.norm(relative_pos).item()
        bond_distances.extend([distance, distance])

    # 如果没有边，直接返回
    if not edge_index_list:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        edge_s_dim = 6 + num_rbf 
        edge_s = torch.empty((0, edge_s_dim), dtype=torch.float32)
        edge_v = torch.empty((0, 1, 3))
    else:
        # e. 对键长进行 RBF 编码
        rbf_distances = torch.tensor(rbf_encode_dist(np.array(bond_distances), num_rbf=num_rbf), dtype=torch.float32)
        
        # f. 拼接化学边特征和RBF几何特征
        edge_s_chem = torch.tensor(np.vstack(edge_s_list), dtype=torch.float32)
        edge_s = torch.cat([edge_s_chem, rbf_distances], dim=1)
        
        # g. 转换为最终的张量格式
        edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
        edge_v = torch.stack(edge_v_list).unsqueeze(1)  


    return (node_s, node_v), edge_index, (edge_s, edge_v)

# --- 测试 ---
if __name__ == "__main__":
    smiles = "O=C(C)Oc1ccccc1C(=O)O"  # 阿司匹林
    name = "阿司匹林"
    # smiles = "C" # 甲烷，用于测试无边的情况
    
    mol_3d = smiles_to_3d_conformer(smiles)

    if mol_3d:
        graph_data = mol_to_gvp_graph(mol_3d)
        if graph_data:
            (node_s, node_v), edge_index, (edge_s, edge_v) = graph_data
            print("--- SMILE2GVP 图数据  ---")
            print(f"SMILES: {smiles}")
            print(f"名称：{name}")
            print(f"节点总数: {node_s.shape[0]}")
            print(f"边总数: {edge_index.shape[1]}")
            print("-" * 20)
            print(f"节点标量特征形状 (node_s): {node_s.shape}")
            print(f"节点向量特征形状 (node_v): {node_v.shape}")
            print(f"边索引形状 (edge_index):   {edge_index.shape}")
            print(f"边标量特征形状 (edge_s):   {edge_s.shape}")
            print(f"边向量特征形状 (edge_v):   {edge_v.shape}")
            print("-" * 20)
            print(f"node_s 的一个样本: {node_s[0]}")
            if edge_s.shape[0] > 0:
                print(f"edge_s 的一个样本: {edge_s[0]}")