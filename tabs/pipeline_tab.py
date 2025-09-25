# tabs/pipeline_tab.py

import gradio as gr
import time
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.GeneMarks.GeneMarkSRunner import GeneMarkSRunner
from typing import Tuple

# --- 后端模拟函数 ---
def run_gene_prediction_real(genome_file) -> Tuple[str, str]:
    if genome_file is None:
        return "请先上传基因组文件。", ""
    
    genome_path = genome_file.name
    output_dir = "./genemarks_output"
    runner = GeneMarkSRunner(gms_script_path="/home/shenmaa/gms2_linux_64/gms2.pl")  # 使用默认 gms2.pl 路径
    try:
        results = runner.run(
            input_fasta=genome_path,
            output_dir=output_dir
        )
        protein_faa = results.get("faa", "")
        msg = f"GeneMarkS 注释完成！蛋白质序列保存在 `{protein_faa}`"
        return msg, protein_faa
    except Exception as e:
        return f"GeneMarkS 执行出错: {e}", ""

def build_draft_model(protein_file_path):
    if not protein_file_path:
        return "请先完成上一步的基因预测。", ""
    print(f"开始构建草稿模型，输入文件: {protein_file_path}")
    time.sleep(3)
    output_path = "model.xml"
    print(f"模型构建完成，输出到: {output_path}")
    return f"草稿模型构建完成！模型保存在 `{output_path}`。", output_path

def integrate_kcat(model_path, protein_path):
    if not model_path or not protein_path:
        return "请先完成前两步。", ""
    print(f"开始整合kcat值，输入: {model_path}, {protein_path}")
    time.sleep(4)
    output_path = "ecGEM.json"
    print(f"kcat整合完成，输出到: {output_path}")
    return f"kcat 值整合完成！最终模型保存在 `{output_path}`。", output_path

def run_analysis(ecgem_path, target_reaction, algorithm):
    if not ecgem_path:
        return "请先生成 ecGEM 模型。"
    print(f"开始代谢分析，模型: {ecgem_path}, 目标: {target_reaction}, 算法: {algorithm}")
    time.sleep(3)
    result = f"""
### 分析报告
- **模型**: `{ecgem_path}`
- **目标反应**: `{target_reaction}`
- **算法**: `{algorithm}`
- **分析结果**: 模拟通量为 0.87 mmol/gDW/h。
- **改造建议**: 建议敲除基因 `YOL001W` 以提高目标产量。
"""
    print("分析完成。")
    return result

# --- UI 构建 ---
def create_pipeline_tab():
    with gr.Blocks() as pipeline_interface:
        gr.Markdown("## 代谢网络模型构建与分析流程 (手动)")
        gr.Markdown("请按顺序执行以下步骤。每一步的输出将作为下一步的输入。")

        protein_path_state = gr.State("")
        model_path_state = gr.State("")
        ecgem_path_state = gr.State("")

        # --- 步骤 1: 基因预测 ---
        with gr.Group():
            gr.Markdown("### 1. 基因预测 (GeneMarkS)")
            with gr.Row():
                genome_input = gr.File(label="上传基因组文件 (fasta格式)")
                gene_pred_btn = gr.Button("运行基因预测", variant="primary")
            gene_pred_output_status = gr.Markdown(value="*等待任务...*")

        # --- 步骤 2: 构建草稿模型 ---
        with gr.Group():
            gr.Markdown("### 2. 构建草稿模型 (CarveMe)")
            build_model_btn = gr.Button("构建草稿模型", variant="primary")
            build_model_output_status = gr.Markdown(value="*等待上一步完成...*")

        # --- 步骤 3: 整合 kcat 值 ---
        with gr.Group():
            gr.Markdown("### 3. 整合 kcat 预测")
            integrate_kcat_btn = gr.Button("整合 kcat 值", variant="primary")
            integrate_kcat_output_status = gr.Markdown(value="*等待上一步完成...*")

        # --- 步骤 4: 代谢分析 ---
        with gr.Group():
            gr.Markdown("### 4. 代谢分析 (FBA等)")
            with gr.Row():
                target_reaction_input = gr.Textbox(label="目标反应 (Reaction ID)", placeholder="例如: `EX_succ_e`")
                algorithm_input = gr.Radio(["FBA", "pFBA", "FVA"], label="选择算法", value="FBA")
            run_analysis_btn = gr.Button("运行代谢分析", variant="primary")
            analysis_output = gr.Markdown(value="*等待分析任务...*")

        # --- 事件绑定 ---
        gene_pred_btn.click(
            fn=run_gene_prediction_real,
            inputs=[genome_input],
            outputs=[gene_pred_output_status, protein_path_state]
        )
        build_model_btn.click(
            fn=build_draft_model,
            inputs=[protein_path_state],
            outputs=[build_model_output_status, model_path_state]
        )
        integrate_kcat_btn.click(
            fn=integrate_kcat,
            inputs=[model_path_state, protein_path_state],
            outputs=[integrate_kcat_output_status, ecgem_path_state]
        )
        run_analysis_btn.click(
            fn=run_analysis,
            inputs=[ecgem_path_state, target_reaction_input, algorithm_input],
            outputs=[analysis_output]
        )

    return pipeline_interface