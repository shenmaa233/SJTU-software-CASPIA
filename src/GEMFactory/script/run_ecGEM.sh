python -m src.GEMFactory.src.ecGEM.run_ecGEM --ecModel_output_file src/GEMFactory/data/ecGEM/GCF_000005845.2_ASM584v2_genomic/ecModel.json 


python -m src.GEMFactory.src.ecGEM.run_ecGEM \
    --ecModel_output_file src/GEMFactory/data/CarveMe/GCF_000005845.2_ASM584v2_genomic_draft.xml \
    --result_folder src/GEMFactory/data/ecGEM/GCF_000005845.2_ASM584v2_genomic \
    --algorithm "FBA"

python -m src.GEMFactory.src.ecGEM.run_ecGEM \
    --ecModel_output_file src/GEMFactory/data/CarveMe/GCF_000005845.2_ASM584v2_genomic_draft.xml \
    --result_folder src/GEMFactory/data/ecGEM/GCF_000005845.2_ASM584v2_genomic \
    --algorithm "pFBA"

python -m src.GEMFactory.src.ecGEM.run_ecGEM \
    --ecModel_output_file src/GEMFactory/data/CarveMe/GCF_000005845.2_ASM584v2_genomic_draft.xml \
    --result_folder src/GEMFactory/data/ecGEM/GCF_000005845.2_ASM584v2_genomic \
    --algorithm "FVA"