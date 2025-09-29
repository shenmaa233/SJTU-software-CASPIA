import cobra
from cobra.io import read_sbml_model
from Bio import SeqIO
import csv
import requests

# 载入模型
model = read_sbml_model("src/GEMFactory/data/CarveMe/ecoli_draft.xml")

# 读取蛋白质序列（GeneMarkS输出）
faa_file = "src/GEMFactory/data/GeneMarkS/GCF_000005845.2_ASM584v2_genomic/GCF_000005845.2_ASM584v2_genomic_protein_clean.fasta"
gene_to_seq = {}
for record in SeqIO.parse(faa_file, "fasta"):
    gene_to_seq[record.id] = str(record.seq)

# 简单 PubChem 查询函数（代谢物 → SMILES）
def get_smiles_from_pubchem(name):
    try:
        url = f"https://pubchem.ncbi.nlm.nih.gov/rest/pug/compound/name/{name}/property/CanonicalSMILES/TXT"
        r = requests.get(url, timeout=5)
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        return None
    return None

# 输出 CSV
with open("gem_reactions_with_sequences_and_smiles.csv", "w", newline="", encoding="utf-8") as f:
    writer = csv.writer(f)
    writer.writerow(["Reaction ID", "Reaction Formula", "Protein Sequence", "Substrate SMILES"])

    for rxn in model.reactions:
        # 反应式
        formula = rxn.reaction

        # 蛋白序列（根据基因-反应规则）
        seqs = []
        for gene in rxn.genes:
            if gene.id in gene_to_seq:
                seqs.append(gene_to_seq[gene.id])
        seqs_str = ";".join(seqs) if seqs else "NA"

        # 底物 SMILES
        substrates = [m.name for m in rxn.reactants]  # 取底物名字
        smiles_list = []
        for s in substrates:
            smi = get_smiles_from_pubchem(s)
            smiles_list.append(smi if smi else "NA")
        smiles_str = ";".join(smiles_list) if smiles_list else "NA"

        # 写入一行
        writer.writerow([rxn.id, formula, seqs_str, smiles_str])

print("✅ CSV 已生成: gem_reactions_with_sequences_and_smiles.csv")
