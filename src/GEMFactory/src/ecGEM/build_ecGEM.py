import re
import pandas as pd
import cobra
import argparse
import os
from .utils import *

def split_and_pair_substrate_with_protein(model_file, result_folder = './ec_result/'):
    '''
    Splits the substrate of reactions to match the gene of reactions and saves the results to a file.

    Arguments
    ----------
    * model: cobra.Model ~ A Model object which will be modified in place.
    * metabolites_reactions_gpr_file: str ~ File path to save the resulting DataFrame as a CSV file.

    Returns
    ----------
    * gprdf: pandas.DataFrame ~ DataFrame with metabolite information.
    '''
    model = load_model(model_file)

    metabolites_reactions_gpr_file = '%s/metabolites_reactions_gpr.csv'%result_folder
    currency_metabolites = get_currency_metabolites()
    
    # Perform necessary modifications to the model
    convert_to_irreversible(model)
    model = isoenzyme_split(model)

    # Preprocess the metabolites and create the DataFrame
    gprdf = preprocess_metabolites(model, currency_metabolites)
    
    # Save the DataFrame to a file
    gprdf.to_csv(metabolites_reactions_gpr_file, index=False)

    return gprdf

def build_ecGEM(model_file, result_folder, f=0.405, ptot=0.56, sigma=1.0, lowerbound=0):
    """
    Build enzyme-constrained GEM and save as JSON.

    Returns
    -------
    ecModel_output_file: str ~ Path to the saved ecGEM JSON file
    """
    reaction_kcat_mw_file = f"{result_folder}/reaction_kcat_mw.csv"
    ecModel_output_file = f"{result_folder}/ecModel.json"
    upperbound = round(ptot * f * sigma, 3)

    # 构建 enzyme-constrained GEM
    trans_model2enz_json_model_split_isoenzyme(
        model_file,
        reaction_kcat_mw_file,
        f,
        ptot,
        sigma,
        lowerbound,
        upperbound,
        ecModel_output_file
    )

    return ecModel_output_file

def main():
    parser = argparse.ArgumentParser(
        description="Pipeline for constructing and optimizing enzyme-constrained GEM"
    )

    # 输入输出相关参数
    parser.add_argument("--protein_clean_file", type=str, default=None,
                        help="Path to protein clean FASTA file")
    parser.add_argument("--model_file", type=str, required=True,
                        help="Path to draft GEM model file (e.g., .xml)")
    parser.add_argument("--result_folder", type=str, default=None,
                        help="Directory to save intermediate and final results")
    parser.add_argument("--bigg_met_file", type=str, default='src/GEMFactory/src/ecGEM/bigg_models_metabolites.txt',
                        help="Path to BiGG metabolites file")
    
    # 优化相关参数
    parser.add_argument("--substrate", type=str, default="EX_glc__D_e",
                        help="Exchange reaction ID for carbon source")
    parser.add_argument("--concentration", type=float, default=10.0,
                        help="Substrate concentration (mM)")
    parser.add_argument("--f", type=float, default=0.405,
                        help="Fraction of enzymes with available kcat values")
    parser.add_argument("--ptot", type=float, default=0.56,
                        help="Total protein fraction of cell dry weight (g/gDW)")
    parser.add_argument("--sigma", type=float, default=1.0,
                        help="Average enzyme saturation factor")
    parser.add_argument("--lowerbound", type=float, default=0.0,
                        help="Lower bound of enzyme usage constraints")
    
    args = parser.parse_args()

    if args.result_folder is None:
        args.result_folder = f"src/GEMFactory/data/ecGEM/{os.path.basename(args.model_file).split('_draft.xml')[0]}"

    if args.protein_clean_file is None:
        args.protein_clean_file = f"src/GEMFactory/data/GeneMarkS/{os.path.basename(args.model_file).split('_draft.xml')[0]}/{os.path.basename(args.model_file).split('_draft.xml')[0]}_protein_clean.fasta"

    # 创建结果目录
    os.makedirs(args.result_folder, exist_ok=True)

    # Step 1: 检查模型是否适合构建ecGEM
    suitability, messages = Determine_suitable_ecGEM(args.model_file, args.bigg_met_file)
    print(f"Model suitability for EC construction: {suitability}")
    for message in messages:
        print(f"- {message}")
    
    if suitability == 'No':
        print("The model is not suitable for enzyme-constrained model construction. Exiting.")
        exit(1)

    # Step 2: 拆分并匹配底物与蛋白
    gprdf = split_and_pair_substrate_with_protein(args.model_file, args.result_folder)

    # Step 3: 预测 kcat
    gprdf_with_kcat = kcat_predict(gprdf, args.protein_clean_file, args.model_file, args.result_folder)
    gprdf_with_kcat = pd.read_csv(f"{args.result_folder}/full_metabolites_reactions.csv")
    
    # Step 4: 获得 kcat 与分子量
    reaction_kcat_mw = get_kcat_mw(gprdf_with_kcat, args.result_folder)

    # Step 5: 构建 ecGEM
    ecModel_output_file = build_ecGEM(
        args.model_file, args.result_folder,
        f=args.f, ptot=args.ptot, sigma=args.sigma, lowerbound=args.lowerbound
    )
    print(f"ecGEM built and saved at {ecModel_output_file}")


if __name__ == "__main__":
    main()