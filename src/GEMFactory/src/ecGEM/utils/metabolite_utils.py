import pandas as pd
import pubchempy as pcp
import os
import json
import time
from tqdm import tqdm

def get_metabolite_smiles_from_pubchem(metabolite_name):
    """
    根据代谢物名称从PubChem获取canonical SMILES
    
    Args:
        metabolite_name: 代谢物名称
    
    Returns:
        canonical_smiles: SMILES字符串，如果未找到则返回None
    """
    if pd.isna(metabolite_name) or metabolite_name == '':
        return None
        
    try:
        # 查询PubChem数据库
        results = pcp.get_compounds(metabolite_name, 'name')
        
        if results:
            # 返回第一个结果的SMILES
            compound = results[0]
            if hasattr(compound, 'smiles') and compound.smiles:
                return compound.smiles
            else:
                return None
        
    except Exception as e:
        # Error getting SMILES - silently handle to avoid I/O errors in daemon threads
        return None

def get_smiles_for_gprdf(gprdf, model, cache_file = 'src/GEMFactory/src/ecGEM/utils/smiles_cache.json'):
    """
    为gprdf中的每个代谢物获取SMILES

    Args:
        gprdf: DataFrame包含代谢物反应信息
        model: CobraModel对象
        cache_file: 缓存文件路径，用于存储metabolite name和SMILES的映射关系

    Returns:
        smiles_list: SMILES字符串列表
    """
    # 加载缓存文件
    if os.path.exists(cache_file):
        try:
            with open(cache_file, 'r', encoding='utf-8') as f:
                processed_metabolites = json.load(f)
            # Loaded cache - removed print to avoid I/O errors in daemon threads
        except Exception as e:
            # Failed to load cache - removed print to avoid I/O errors in daemon threads
            processed_metabolites = {}
    else:
        processed_metabolites = {}

    smiles_list = []
    new_entries = 0  # 记录新查询的条目数

    # Getting SMILES - removed print to avoid I/O errors in daemon threads

    for idx, row in tqdm(gprdf.iterrows(), total=len(gprdf), desc="获取SMILES"):
        metabolite_id = row['metabolites']  # 第二列是metabolites

        # 从model中获取代谢物对象
        metabolite = model.metabolites.get_by_id(metabolite_id)
        metabolite_name = metabolite.name if hasattr(metabolite, 'name') and metabolite.name else metabolite_id

        try:
            # 如果已经处理过这个代谢物，直接使用缓存的结果
            if metabolite_name in processed_metabolites:
                smiles_list.append(processed_metabolites[metabolite_name])
                continue

            # 如果缓存中没有，则查询PubChem获取SMILES
            smiles = get_metabolite_smiles_from_pubchem(metabolite_name)

            # 更新缓存
            processed_metabolites[metabolite_name] = smiles
            smiles_list.append(smiles)
            new_entries += 1

            # 添加小延迟避免过于频繁的API调用
            time.sleep(0.1)

            # 每查询10个新条目就保存一次缓存（防止意外中断丢失数据）
            if new_entries % 10 == 0:
                try:
                    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
                    with open(cache_file, 'w', encoding='utf-8') as f:
                        json.dump(processed_metabolites, f, ensure_ascii=False, indent=2)
                    # Saved cache - removed print to avoid I/O errors in daemon threads
                except Exception as e:
                    # Failed to save cache - removed print to avoid I/O errors in daemon threads
                    pass

        except Exception as e:
            # Error processing metabolite - removed print to avoid I/O errors in daemon threads
            # 对于出错的情况，也要记录到缓存中避免重复查询
            if metabolite_name not in processed_metabolites:
                processed_metabolites[metabolite_name] = None
                new_entries += 1
            smiles_list.append(None)

    # SMILES retrieval complete - removed print to avoid I/O errors in daemon threads
    return smiles_list