def get_protein_sequences_from_fasta(fasta_file):
    """
    从FASTA文件中读取蛋白质序列
    返回一个字典，键为蛋白质ID，值为序列
    """
    protein_sequences = {}
    current_protein = None
    current_sequence = ""
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # 保存之前的蛋白质序列
                if current_protein:
                    protein_sequences[current_protein] = current_sequence
                # 开始新的蛋白质
                current_protein = line[1:]  # 去掉'>'符号
                current_sequence = ""
            else:
                current_sequence += line
        
        # 保存最后一个蛋白质序列
        if current_protein:
            protein_sequences[current_protein] = current_sequence
    
    return protein_sequences

def calculate_protein_molecular_weight(sequence):
    """
    计算蛋白质分子量（单位：Da）
    使用标准氨基酸分子量
    """
    # 氨基酸分子量字典（单位：Da）
    aa_weights = {
        'A': 89.09, 'R': 174.20, 'N': 132.12, 'D': 133.10, 'C': 121.15,
        'E': 147.13, 'Q': 146.15, 'G': 75.07, 'H': 155.16, 'I': 131.17,
        'L': 131.17, 'K': 146.19, 'M': 149.21, 'F': 165.19, 'P': 115.13,
        'S': 105.09, 'T': 119.12, 'W': 204.23, 'Y': 181.19, 'V': 117.15
    }
    
    # 计算分子量
    total_weight = 0.0
    for aa in sequence.upper():
        if aa in aa_weights:
            total_weight += aa_weights[aa]
    
    # 减去水分子重量（肽键形成时脱水）
    if len(sequence) > 1:
        total_weight -= (len(sequence) - 1) * 18.015
    
    return total_weight