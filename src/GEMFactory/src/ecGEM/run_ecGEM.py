from ast import main
import re
import cobra
import argparse
from utils import *

def run_ecGEM_fba(ecModel_output_file, result_folder, obj=None,
                  use_substrate="EX_glc__D_e", concentration=10):
    """
    Run FBA on an existing ecGEM.

    Returns
    -------
    flux_value: float ~ Optimal flux value of the objective function
    """

    # 加载 enzyme-constrained model
    enz_model = get_enzyme_constraint_model(ecModel_output_file)

    # 设置目标函数（默认 biomass）
    if obj is None:
        for reaction in enz_model.reactions:
            if 'biomass' in reaction.name.lower():
                obj = reaction.id
                break
    enz_model.objective = obj

    # 修改底物供给条件
    ori_obj_id, ori_substrate_id_list, ori_sub_concentration, ori_ATPM = get_model_substrate_obj(enz_model)
    for eachsubid in ori_substrate_id_list:
        if re.search('_reverse', eachsubid):
            r_id_new = eachsubid.split('_reverse')[0]
            enz_model.reactions.get_by_id(eachsubid).bounds = (0, 0)
            enz_model.reactions.get_by_id(r_id_new).bounds = (0, 0)
        else:
            r_id_new = eachsubid + '_reverse'
            enz_model.reactions.get_by_id(eachsubid).bounds = (0, 0)
            enz_model.reactions.get_by_id(r_id_new).bounds = (0, 0)

    enz_model.reactions.get_by_id(use_substrate).bounds = (-concentration, 0)
    enz_model.reactions.get_by_id(use_substrate + '_reverse').bounds = (0, 0)

    # FBA
    fluxes_outfile = f'{result_folder}/ECMpy_solution_{obj}_pfba.csv'
    enz_model_pfba_solution = cobra.flux_analysis.pfba(enz_model)
    enz_model_pfba_solution = get_fluxes_detail_in_model(
        enz_model, enz_model_pfba_solution, fluxes_outfile, ecModel_output_file
    )

    return enz_model_pfba_solution.fluxes[obj]

def main():
    parser = argparse.ArgumentParser(
        description="Run FBA on an existing ecGEM."
    )
    parser.add_argument("--ecModel_output_file", type=str, required=True,
                        help="Path to the ecGEM JSON file")
    parser.add_argument("--result_folder", type=str, required=True,
                        help="Path to the result folder")
    parser.add_argument("--obj", type=str, default=None,
                        help="The objective function")
    parser.add_argument("--substrate", type=str, default="EX_glc__D_e",
                        help="The substrate")
    parser.add_argument("--concentration", type=float, default=10,
                        help="The concentration")
    args = parser.parse_args()

    optimal_value = run_ecGEM_fba(
        args.ecModel_output_file, args.result_folder,
        obj=args.obj,
        use_substrate=args.substrate,
        concentration=args.concentration
    )

    print(f"Optimal value of the objective function: {optimal_value}")

if __name__ == "__main__":
    main()