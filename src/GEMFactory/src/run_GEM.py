import os
import re
import sys
import jpype
import argparse
import pandas as pd
import cobra
from cobra.io.json import from_json

# === 在导入 straindesign 之前配置 JAVA_HOME 并启动 JVM ===
java_install_path = os.getenv("JAVA_HOME")
os.environ['JAVA_HOME'] = java_install_path
print(f"JAVA_HOME successfully set to: {os.environ['JAVA_HOME']}", file=sys.stderr)

if not jpype.isJVMStarted():
    jpype.startJVM(
        jpype.getDefaultJVMPath(),
        "--enable-native-access=ALL-UNNAMED"
    )

import straindesign as sd
from .ecGEM.utils import (
    get_enzyme_constraint_model,
    get_model_substrate_obj,
    get_fluxes_detail_in_model,
    load_model
)

# ===============================================================
# ecGEM FBA 功能
# ===============================================================
def run_ecGEM_fba(ecModel_output_file, result_folder, obj=None,
                  use_substrate="EX_glc__D_e", concentration=10):
    """
    Run FBA (default pFBA) on an existing ecGEM.

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

    # pFBA
    fluxes_outfile = f'{result_folder}/ECMpy_solution_{obj}_pfba.csv'
    enz_model_pfba_solution = cobra.flux_analysis.pfba(enz_model)
    enz_model_pfba_solution = get_fluxes_detail_in_model(
        enz_model, enz_model_pfba_solution, fluxes_outfile, ecModel_output_file
    )

    return enz_model_pfba_solution.fluxes[obj]

# ===============================================================
# straindesign 算法选择接口
# ===============================================================
def run_straindesign(model_file, algorithm="FBA", target_reaction_id=None, result_folder="./result"):
    """
    Run straindesign FBA / pFBA / FVA on a COBRA model.
    """
    model = load_model(model_file)
    output_df = None
    objective_value = None

    if target_reaction_id is None:
        for reaction in model.reactions:
            if 'biomass' in reaction.name.lower():
                target_reaction_id = reaction.id
                break

    if algorithm == "FBA":
        fba_sol = sd.fba(model, solver='scip', obj=target_reaction_id)
        print(f"Maximum growth: {fba_sol.objective_value}.", file=sys.stderr)
        objective_value = fba_sol.objective_value
        output_df = pd.DataFrame(list(fba_sol.fluxes.items()), columns=['Reaction', 'Flux'])

    elif algorithm == "pFBA":
        pfba_sol = sd.fba(model, solver='scip', pfba=1, obj=target_reaction_id)
        objective_value = pfba_sol.objective_value
        output_df = pd.DataFrame(list(pfba_sol.fluxes.items()), columns=['Reaction', 'Flux'])

    elif algorithm == "FVA":
        fva_sol = sd.fva(model)
        output_df = fva_sol
        objective_value = None  # FVA 没有单一目标值

    # 保存结果
    os.makedirs(result_folder, exist_ok=True)
    out_csv = os.path.join(result_folder, f"straindesign_{algorithm}_solution.csv")
    output_df.to_csv(out_csv, index=False)
    print(f"Results saved to {out_csv}", file=sys.stderr)

    return objective_value

# ===============================================================
# 主函数
# ===============================================================
def main():
    parser = argparse.ArgumentParser(description="Run FBA on an existing ecGEM or via straindesign.")
    parser.add_argument("--ecModel_output_file", type=str, required=True,
                        help="Path to the ecGEM JSON file")
    parser.add_argument("--result_folder", type=str, default=None,
                        help="Path to the result folder")
    parser.add_argument("--obj", type=str, default=None,
                        help="The objective function")
    parser.add_argument("--substrate", type=str, default="EX_glc__D_e",
                        help="The substrate")
    parser.add_argument("--concentration", type=float, default=10,
                        help="The concentration")
    parser.add_argument("--algorithm", type=str, default="ecGEM",
                        choices=["ecGEM", "FBA", "pFBA", "FVA"],
                        help="Which algorithm to run: ecGEM (default pFBA), FBA, pFBA, FVA")

    args = parser.parse_args()
    if args.result_folder is None:
        args.result_folder = os.path.dirname(args.ecModel_output_file) + "/result"
    os.makedirs(args.result_folder, exist_ok=True)

    if args.algorithm == "ecGEM":
        optimal_value = run_ecGEM_fba(
            args.ecModel_output_file, args.result_folder,
            obj=args.obj, use_substrate=args.substrate, concentration=args.concentration
        )
        print(f"Optimal value of the objective function (ecGEM pFBA): {optimal_value}")
    else:
        optimal_value = run_straindesign(
            args.ecModel_output_file, algorithm=args.algorithm,
            target_reaction_id=args.obj, result_folder=args.result_folder
        )
        if optimal_value is not None:
            print(f"Optimal value of the objective function ({args.algorithm}): {optimal_value}")

if __name__ == "__main__":
    main()
