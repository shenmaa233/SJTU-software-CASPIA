import shutil
from pathlib import Path

import gradio as gr

from src.GEMFactory.src.build_GEM import clean_faa, run_carveme
from src.GEMFactory.src.utils.GeneMarkS import GeneMarkSRunner
from src.utils import LogManager, TaskRunner

GENOME_DIR = "src/GEMFactory/data/Genome"


def list_genomes():
    p = Path(GENOME_DIR)
    p.mkdir(parents=True, exist_ok=True)
    exts = {".fna", ".fa", ".fasta"}
    return [str(f.resolve()) for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]


def save_and_refresh(uploaded_path):
    if uploaded_path is None or uploaded_path == "":
        return gr.update()
    src = Path(uploaded_path)
    if not src.exists():
        return gr.update()
    dest_dir = Path(GENOME_DIR)
    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / src.name
    try:
        shutil.copy(src, dest)
    except Exception:
        pass
    genomes = list_genomes()
    return gr.update(choices=genomes, value=str(dest.resolve()))


def gem_pipeline(logger, genome_path: str, gapfill: str):
    logger.info(f"🚀 Pipeline started. genome={genome_path}, medium={gapfill}")

    # 1. Run GeneMarkS
    logger.info("🔬 Step 1: Running GeneMarkS...")
    gms_runner = GeneMarkSRunner(gms_script_path="/home/shenmaa/gms2_linux_64/gms2.pl")
    gms_outputs = gms_runner.run(
        input_fasta=genome_path,
        output_dir="src/GEMFactory/data/GeneMarkS",
        genome_type="bacteria",
        gcode="11",
    )
    logger.info("✅ GeneMarkS annotation completed.")

    # 2. Clean FASTA
    logger.info("🧹 Step 2: Cleaning protein FASTA headers...")
    clean_faa_path = clean_faa(gms_outputs["faa"])
    logger.info(f"✅ Clean FASTA saved: {clean_faa_path}")

    # 3. Run CarveMe
    prefix = Path(genome_path).stem
    gem_output = f"src/GEMFactory/data/CarveMe/{prefix}_draft.xml"
    logger.info("🛠️ Step 3: Running CarveMe reconstruction...")
    run_carveme(clean_faa_path, gem_output, gapfill=gapfill, tmpdir="src/GEMFactory/data/temp")
    logger.info(f"✅ GEM built successfully: {gem_output}")

    return gem_output


# --- Setup ---
logs = LogManager("./logs")
runner = TaskRunner(logs)


# --- Gradio Callbacks ---
def start_pipeline(genome_file, gapfill, _sid):
    if genome_file is None or genome_file == "":
        return "", "❌ Please upload a genome file (.fna).", ""
    sid = runner.start(gem_pipeline, genome_file, gapfill, prefix="gem-")
    return sid, f"🚧 Running (sid={sid})", logs.read_tail(sid)


def poll_pipeline(sid: str):
    return runner.poll(sid)


# --- Gradio UI ---
def gemfactory_tab():
    with gr.Blocks():
        gr.Markdown("## Genome → GEM Pipeline\nUpload a genome and build GEM automatically with GeneMarkS + CarveMe.")

        sid_state = gr.State("")

        with gr.Row():
            genome_upload = gr.File(label="Upload Genome (.fna/.fa/.fasta)", type="filepath", file_types=[".fna", ".fa", ".fasta"])
            existing = list_genomes()
            genome_file = gr.Dropdown(choices=existing, value=(existing[0] if existing else None), label="Genome")
            gapfill = gr.Dropdown(choices=["None", "M9", "LB", "M9,LB"], value="None", label="Gapfill Medium")

        genome_upload.change(fn=save_and_refresh, inputs=[genome_upload], outputs=[genome_file])

        run_btn = gr.Button("🚀 Run Pipeline")

        logs_box = gr.Textbox(label="Progress Log", lines=15, interactive=False)
        status_box = gr.Textbox(label="Status", interactive=False)
        result_box = gr.Textbox(label="Generated GEM Path", interactive=False)

        run_btn.click(fn=start_pipeline, inputs=[genome_file, gapfill, sid_state], outputs=[sid_state, status_box, logs_box])

        timer = gr.Timer(1.0)
        timer.tick(fn=poll_pipeline, inputs=[sid_state], outputs=[logs_box, status_box, result_box])
