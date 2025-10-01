from rdkit import Chem
from rdkit.Chem import AllChem

def smiles_to_3d_conformer(smiles_string):
    """
    将单个 SMILES 字符串转换为 RDKit 的 3D 分子对象。

    参数:
        smiles_string (str): 输入的 SMILES 字符串。

    返回:
        rdkit.Chem.Mol: 带有 3D 构象的 RDKit 分子对象，如果失败则返回 None。
    """
    try:
        # 1. 从 SMILES 创建分子对象
        mol = Chem.MolFromSmiles(smiles_string)
        if mol is None:
            return None

        # 2. 添加氢原子
        mol = Chem.AddHs(mol)

        # 3. 生成 3D 构象 (返回构象 ID)
        # 使用 ETKDG v3 算法并设置多个尝试次数
        ps = AllChem.ETKDGv3()
        ps.randomSeed = 42  # 设置随机种子以获得可重复的结果
        ps.numThreads = 0   # 使用所有可用的线程
        ps.maxIterations = 1000  # 增加最大迭代次数
        ps.useRandomCoords = True  # 在无法使用特征坐标时使用随机坐标
        
        # 尝试多次生成构象
        cid = AllChem.EmbedMultipleConfs(mol, numConfs=10, params=ps)
        if len(cid) == 0:
            # 如果ETKDGv3失败，尝试使用随机坐标生成
            ps.useRandomCoords = True
            ps.randomSeed = 42
            cid = AllChem.EmbedMolecule(mol, ps)
            if cid == -1:
                print(f"警告: 无法为 SMILES '{smiles_string}' 生成构象。")
                return None
        
        # (可选) 使用力场优化构象
        AllChem.MMFFOptimizeMolecule(mol, confId=0)
        
        return mol
    except Exception as e:
        print(f"处理 SMILES '{smiles_string}' 时出错: {e}")
        return None

# --- 测试 ---
if __name__ == "__main__":
    smiles = "CCO"  # 乙醇的 SMILES
    mol_3d = smiles_to_3d_conformer(smiles)

    if mol_3d:
        print(f"成功为 '{smiles}' 生成 3D 构象。")
        # 获取第一个构象
        conformer = mol_3d.GetConformer(0)
        # 打印每个原子的坐标
        for atom in mol_3d.GetAtoms():
            pos = conformer.GetAtomPosition(atom.GetIdx())
            print(f"原子 {atom.GetSymbol()}({atom.GetIdx()}) 的坐标: ({pos.x:.4f}, {pos.y:.4f}, {pos.z:.4f})")