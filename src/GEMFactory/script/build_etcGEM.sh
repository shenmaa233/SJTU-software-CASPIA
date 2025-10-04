#!/bin/bash
# ==========================
# Shell script to run ecGEM pipeline
# ==========================

# === 用户自定义参数 ===
MODEL_FILE="src/GEMFactory/data/CarveMe/GCF_000005845.2_ASM584v2_genomic_draft.xml"   # 输入 GEM 模型文件
RESULT_FOLDER="src/GEMFactory/data/ecGEM/GCF_000005845.2_ASM584v2_genomic"        # 输出结果文件夹
is_etc=True
T=37.0

# 优化相关参数
SUBSTRATE="EX_glc__D_e"
CONCENTRATION=10.0
F=0.405
PTOT=0.56
SIGMA=1.0
LOWERBOUND=0.0

# === 执行 Python 脚本 ===
python -m src.GEMFactory.src.ecGEM.build_ecGEM \
    --model_file "$MODEL_FILE" \
    --result_folder "$RESULT_FOLDER" \
    --substrate "$SUBSTRATE" \
    --concentration $CONCENTRATION \
    --f $F \
    --ptot $PTOT \
    --sigma $SIGMA \
    --lowerbound $LOWERBOUND \
    --is_etc $is_etc \
    --T $T
