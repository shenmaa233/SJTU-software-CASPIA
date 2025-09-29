import gradio as gr
import os
from src.GEMFactory.src.build_GEM_tool import build_gem

# === 主函数：带进度条 ===
def run_pipeline(genome_file, gapfill_medium):
    logs = []
    try:
        if genome_file is None:
            return "❌ Please upload a genome file (.fna).", ""

        genome_path = genome_file.name

        # === Step 1: GeneMarkS ===
        logs.append("🔬 Step 1: Running GeneMarkS...")
        yield "\n".join(logs), ""

        results = build_gem(
            genome_fasta=genome_path,
            gms_script="/home/shenmaa/gms2_linux_64/gms2.pl",
            gapfill=gapfill_medium if gapfill_medium != "None" else None
        )

        logs.append("✅ GeneMarkS annotation completed.")
        yield "\n".join(logs), ""

        # === Step 2: Clean FASTA ===
        logs.append("🧹 Step 2: Cleaning protein FASTA headers...")
        yield "\n".join(logs), ""
        logs.append(f"✅ Clean FASTA saved: {results['clean_faa']}")
        yield "\n".join(logs), ""

        # === Step 3: CarveMe ===
        logs.append("🛠️ Step 3: Running CarveMe reconstruction...")
        yield "\n".join(logs), ""
        logs.append(f"✅ GEM built successfully: {results['gem']}")
        yield "\n".join(logs), results['gem']

    except Exception as e:
        logs.append(f"❌ Error: {e}")
        yield "\n".join(logs), ""


# === Gradio Tab 界面 ===
def gemfactory_tab():
    with gr.Tab("🧬 GEM Factory"):
        gr.Markdown("## Genome → GEM Pipeline\nUpload a genome and build GEM automatically with GeneMarkS + CarveMe.")

        with gr.Row():
            genome_file = gr.File(label="Upload Genome (.fna)", type="filepath", file_types=[".fna"])
            gapfill_medium = gr.Dropdown(
                choices=["None", "M9", "LB", "M9,LB"],
                value="None",
                label="Gapfill Medium"
            )

        run_btn = gr.Button("🚀 Run Pipeline")

        progress = gr.Textbox(label="Progress Log", lines=15, interactive=False)
        result = gr.Textbox(label="Generated GEM Path")

        run_btn.click(
            fn=run_pipeline,
            inputs=[genome_file, gapfill_medium],
            outputs=[progress, result],
        )
