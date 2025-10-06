"""
GEMFactory Tab - Complete Multi-Tab Interface
==============================================

A comprehensive interface for:
1. Draft GEM building (GeneMarkS + CarveMe)
2. ecGEM construction (enzyme-constrained)
3. etcGEM construction (enzyme-temperature-constrained)
4. Model viewing and analysis
5. Results management

Author: SJTU-Software Team
Date: 2025-10
"""

import shutil
import os
import dotenv
from pathlib import Path
from datetime import datetime
from typing import Optional

import gradio as gr
import pandas as pd

from src.GEMFactory.src.build_GEM import clean_faa, run_carveme
from src.GEMFactory.src.utils.GeneMarkS import GeneMarkSRunner
from src.GEMFactory.src.ecGEM.ecgem_service import ECGEMService
from src.utils import get_task_manager

dotenv.load_dotenv()

# ==================== Constants ====================
GENOME_DIR = "src/GEMFactory/data/Genome"
GMS_SCRIPT = os.environ.get("GMS_SCRIPT_PATH")

# ==================== Global Services ====================
task_manager = get_task_manager()
ecgem_service = ECGEMService()

# Global mapping from display name to full path for draft models
_model_path_mapping = {}

# ==================== Utility Functions ====================

def get_model_path_from_display(display_name: str) -> str:
    """Extract actual file path from display name
    
    Display format: "filename.xml (1.2 MB, 2025-10-06 12:00)"
    Returns the full path from the mapping.
    """
    if not display_name:
        return ""
    
    # Extract filename from display string (everything before the first '(')
    filename = display_name.split(' (')[0].strip()
    
    # Get full path from mapping
    return _model_path_mapping.get(filename, display_name)


def format_size(bytes_size):
    """Format file size in human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB']:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def format_timestamp(timestamp):
    """Format Unix timestamp to readable date"""
    return datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d %H:%M:%S')


def list_genomes():
    """List all genome files in the genome directory"""
    p = Path(GENOME_DIR)
    p.mkdir(parents=True, exist_ok=True)
    exts = {".fna", ".fa", ".fasta"}
    return [str(f) for f in sorted(p.iterdir()) if f.is_file() and f.suffix.lower() in exts]


def save_and_refresh_genome(uploaded_path):
    """Save uploaded genome file and refresh dropdown"""
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


# ==================== Tab 1: Draft GEM Builder ====================

def gem_pipeline_task(logger, genome_path: str, gapfill: str):
    """Background task for draft GEM construction"""
    logger.info("="*60)
    logger.info("🚀 Draft GEM Construction Pipeline Started")
    logger.info("="*60)
    logger.info(f"Genome: {genome_path}")
    logger.info(f"Gap-filling medium: {gapfill}")

    # Step 1: GeneMarkS annotation
    logger.info("\n🔬 Step 1: Running GeneMarkS gene annotation...")
    gms_runner = GeneMarkSRunner(gms_script_path=GMS_SCRIPT)
    gms_outputs = gms_runner.run(
        input_fasta=genome_path,
        output_dir="src/GEMFactory/data/GeneMarkS",
        genome_type="bacteria",
        gcode="11",
    )
    logger.info(f"  ✅ GFF: {gms_outputs['gff']}")
    logger.info(f"  ✅ Genes: {gms_outputs['fnn']}")
    logger.info(f"  ✅ Proteins: {gms_outputs['faa']}")

    # Step 2: Clean FASTA headers
    logger.info("\n🧹 Step 2: Cleaning protein FASTA headers...")
    clean_faa_path = clean_faa(gms_outputs["faa"])
    logger.info(f"  ✅ Clean FASTA: {clean_faa_path}")

    # Step 3: CarveMe GEM reconstruction
    prefix = Path(genome_path).stem
    gem_output = f"src/GEMFactory/data/CarveMe/{prefix}_draft.xml"
    logger.info("\n🛠️  Step 3: Running CarveMe reconstruction...")
    logger.info(f"  Output: {gem_output}")
    logger.info(f"  Gap-filling: {gapfill if gapfill != 'None' else 'No'}")
    
    run_carveme(
        clean_faa_path,
        gem_output,
        gapfill=gapfill if gapfill != "None" else None,
        tmpdir="src/GEMFactory/data/temp"
    )
    
    logger.info(f"\n✅ Draft GEM saved: {gem_output}")
    logger.info("="*60)
    logger.info("✅ Pipeline Complete!")
    logger.info("="*60)

    return gem_output


def start_draft_gem_pipeline(genome_file, gapfill, _sid):
    """Start draft GEM building pipeline"""
    if genome_file is None or genome_file == "":
        return "", "❌ Please select or upload a genome file.", ""
    
    genome_name = Path(genome_file).name
    
    task_id = task_manager.start(
        gem_pipeline_task,
        genome_file, 
        gapfill, 
        prefix="gem-",
        task_name=f"Draft GEM: {genome_name}",
        task_type="draft_gem"
    )
    
    logs, status, _ = task_manager.poll(task_id)
    return task_id, status, logs


def poll_draft_gem(sid: str):
    """Poll draft GEM task status"""
    if not sid:
        return "", "", ""
    return task_manager.poll(sid)


def draft_gem_tab():
    """Create Draft GEM Builder tab"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Input")
            genome_upload = gr.File(
                label="Upload Genome",
                type="filepath",
                file_types=[".fna", ".fa", ".fasta"]
            )
            existing = list_genomes()
            genome_dropdown = gr.Dropdown(
                choices=existing,
                value=(existing[0] if existing else None),
                label="Or Select Existing Genome",
                interactive=True
            )
            gapfill_dropdown = gr.Dropdown(
                choices=["None", "M9", "LB", "M9,LB"],
                value="None",
                label="Gap-filling Medium"
            )
            
            genome_upload.change(
                fn=save_and_refresh_genome,
                inputs=[genome_upload],
                outputs=[genome_dropdown]
            )
            
            gr.Markdown("### ⚙️ Actions")
            run_btn = gr.Button("🚀 Build Draft GEM", variant="primary", size="lg")
            
            gr.Markdown("### ℹ️ Info")
            gr.Markdown("""
            **Pipeline Steps:**
            1. 🔬 Gene annotation (GeneMarkS)
            2. 🧹 FASTA cleaning
            3. 🛠️ GEM reconstruction (CarveMe)
            
            **Output:** Draft GEM in SBML format
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Progress")
            sid_state = gr.State("")
            status_box = gr.Textbox(label="Status", interactive=False)
            result_box = gr.Textbox(label="Output File", interactive=False)
            logs_box = gr.Textbox(label="Logs", lines=20, interactive=False)
            
            run_btn.click(
                fn=start_draft_gem_pipeline,
                inputs=[genome_dropdown, gapfill_dropdown, sid_state],
                outputs=[sid_state, status_box, logs_box]
            )
            
            timer = gr.Timer(2.0)
            timer.tick(
                fn=poll_draft_gem,
                inputs=[sid_state],
                outputs=[logs_box, status_box, result_box]
            )


# ==================== Tab 2: ecGEM Builder ====================

def refresh_draft_models():
    """Refresh list of draft models, show file name with metadata"""
    models = ecgem_service.list_draft_models()
    # Create display labels with file name, size, and timestamp
    choices = [f"{Path(m['path']).name} ({format_size(m['size'])}, {format_timestamp(m['modified'])})" for m in models]
    # Store mapping from display name to full path
    global _model_path_mapping
    _model_path_mapping = {Path(m['path']).name: m['path'] for m in models}
    return gr.update(choices=choices, value=choices[0] if choices else None)


def check_model_suitability(model_file):
    """Check if model is suitable for ecGEM construction"""
    if not model_file:
        return "⚠️ Please select a model"
    
    # Convert display name to actual path
    model_path = get_model_path_from_display(model_file)
    is_suitable, messages = ecgem_service.check_model_suitability(model_path)
    
    if is_suitable:
        result = "✅ Model is SUITABLE for ecGEM construction\n\n"
    else:
        result = "❌ Model is NOT SUITABLE for ecGEM construction\n\n"
    
    result += "\n".join(f"• {msg}" for msg in messages)
    return result


def start_ecgem_build(model_file, f, ptot, sigma, lowerbound, _sid):
    """Start ecGEM building task"""
    if not model_file:
        return "", "❌ Please select a model", ""
    
    # Convert display name to actual path
    model_path = get_model_path_from_display(model_file)
    
    # Check suitability first
    is_suitable, messages = ecgem_service.check_model_suitability(model_path)
    if not is_suitable:
        return "", "❌ Model not suitable: " + "; ".join(messages), ""
    
    task_id = ecgem_service.build_ecgem(
        model_file=model_path,
        f=f,
        ptot=ptot,
        sigma=sigma,
        lowerbound=lowerbound
    )
    
    logs, status, _ = task_manager.poll(task_id)
    return task_id, status, logs


def poll_ecgem(sid: str):
    """Poll ecGEM task status"""
    if not sid:
        return "", "", ""
    return task_manager.poll(sid)


def ecgem_builder_tab():
    """Create ecGEM Builder tab"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Model Selection")
            
            refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            model_dropdown = gr.Dropdown(
                label="Select Draft GEM Model",
                choices=[],
                interactive=True
            )
            
            refresh_btn.click(
                fn=refresh_draft_models,
                outputs=[model_dropdown]
            )
            
            check_btn = gr.Button("🔍 Check Suitability", size="sm")
            suitability_box = gr.Textbox(
                label="Suitability Check",
                lines=6,
                interactive=False
            )
            
            check_btn.click(
                fn=check_model_suitability,
                inputs=[model_dropdown],
                outputs=[suitability_box]
            )
            
            gr.Markdown("### ⚙️ Parameters")
            f_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.405,
                step=0.01,
                label="f (Enzyme Fraction)",
                info="Fraction of enzymes with available kcat"
            )
            ptot_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.56,
                step=0.01,
                label="Ptot (Total Protein)",
                info="Total protein fraction (g/gDW)"
            )
            sigma_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="σ (Saturation)",
                info="Average enzyme saturation factor"
            )
            lowerbound_slider = gr.Slider(
                minimum=0.0,
                maximum=0.1,
                value=0.0,
                step=0.01,
                label="Lower Bound",
                info="Lower bound for enzyme constraints"
            )
            
            build_btn = gr.Button("🏗️ Build ecGEM", variant="primary", size="lg")
            
            gr.Markdown("### ℹ️ Info")
            gr.Markdown("""
            **ecGEM Features:**
            - Enzyme kinetic constraints
            - Kcat parameter prediction
            - Protein allocation modeling
            
            **Required:** Draft GEM + Protein sequences
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Progress")
            sid_state = gr.State("")
            status_box = gr.Textbox(label="Status", interactive=False)
            result_box = gr.Textbox(label="Output File", interactive=False)
            logs_box = gr.Textbox(label="Logs", lines=20, interactive=False)
            
            build_btn.click(
                fn=start_ecgem_build,
                inputs=[model_dropdown, f_slider, ptot_slider, sigma_slider, lowerbound_slider, sid_state],
                outputs=[sid_state, status_box, logs_box]
            )
            
            timer = gr.Timer(2.0)
            timer.tick(
                fn=poll_ecgem,
                inputs=[sid_state],
                outputs=[logs_box, status_box, result_box]
            )


# ==================== Tab 3: etcGEM Builder ====================

def start_etcgem_build(model_file, temperature, f, ptot, sigma, lowerbound, _sid):
    """Start etcGEM building task"""
    if not model_file:
        return "", "❌ Please select a model", ""
    
    if temperature is None or temperature < 0 or temperature > 100:
        return "", "❌ Please specify a valid temperature (0-100°C)", ""
    
    # Convert display name to actual path
    model_path = get_model_path_from_display(model_file)
    
    # Check suitability first
    is_suitable, messages = ecgem_service.check_model_suitability(model_path)
    if not is_suitable:
        return "", "❌ Model not suitable: " + "; ".join(messages), ""
    
    task_id = ecgem_service.build_etcgem(
        model_file=model_path,
        temperature=temperature,
        f=f,
        ptot=ptot,
        sigma=sigma,
        lowerbound=lowerbound
    )
    
    logs, status, _ = task_manager.poll(task_id)
    return task_id, status, logs


def etcgem_builder_tab():
    """Create etcGEM Builder tab"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Model Selection")
            
            refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            model_dropdown = gr.Dropdown(
                label="Select Draft GEM Model",
                choices=[],
                interactive=True
            )
            
            refresh_btn.click(
                fn=refresh_draft_models,
                outputs=[model_dropdown]
            )
            
            check_btn = gr.Button("🔍 Check Suitability", size="sm")
            suitability_box = gr.Textbox(
                label="Suitability Check",
                lines=4,
                interactive=False
            )
            
            check_btn.click(
                fn=check_model_suitability,
                inputs=[model_dropdown],
                outputs=[suitability_box]
            )
            
            gr.Markdown("### 🌡️ Temperature")
            temp_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=37.0,
                step=0.5,
                label="Temperature (°C)",
                info="Optimal growth temperature"
            )
            
            gr.Markdown("### ⚙️ Parameters")
            f_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.405, step=0.01,
                label="f (Enzyme Fraction)"
            )
            ptot_slider = gr.Slider(
                minimum=0.1, maximum=1.0, value=0.56, step=0.01,
                label="Ptot (Total Protein)"
            )
            sigma_slider = gr.Slider(
                minimum=0.1, maximum=2.0, value=1.0, step=0.1,
                label="σ (Saturation)"
            )
            lowerbound_slider = gr.Slider(
                minimum=0.0, maximum=0.1, value=0.0, step=0.01,
                label="Lower Bound"
            )
            
            build_btn = gr.Button("🌡️ Build etcGEM", variant="primary", size="lg")
            
            gr.Markdown("### ℹ️ Info")
            gr.Markdown("""
            **etcGEM Features:**
            - All ecGEM features
            - Temperature-dependent kinetics
            - Topt (optimal temperature) prediction
            - Thermal adaptation modeling
            """)
        
        with gr.Column(scale=2):
            gr.Markdown("### 📊 Progress")
            sid_state = gr.State("")
            status_box = gr.Textbox(label="Status", interactive=False)
            result_box = gr.Textbox(label="Output File", interactive=False)
            logs_box = gr.Textbox(label="Logs", lines=20, interactive=False)
            
            build_btn.click(
                fn=start_etcgem_build,
                inputs=[model_dropdown, temp_slider, f_slider, ptot_slider, sigma_slider, lowerbound_slider, sid_state],
                outputs=[sid_state, status_box, logs_box]
            )
            
            timer = gr.Timer(2.0)
            timer.tick(
                fn=poll_ecgem,
                inputs=[sid_state],
                outputs=[logs_box, status_box, result_box]
            )


# ==================== Tab 4: Model Viewer ====================

def list_all_models():
    """List all models (Draft, ecGEM, etcGEM)"""
    draft_models = ecgem_service.list_draft_models()
    ecgem_models = ecgem_service.list_ecgem_models()
    etcgem_models = ecgem_service.list_etcgem_models()
    
    data = []
    
    for m in draft_models:
        data.append([
            "Draft GEM",
            m["name"],
            format_size(m["size"]),
            format_timestamp(m["modified"]),
            m["path"]
        ])
    
    for m in ecgem_models:
        data.append([
            "ecGEM",
            m["name"],
            format_size(m["size"]),
            format_timestamp(m["modified"]),
            m["path"]
        ])
    
    for m in etcgem_models:
        data.append([
            "etcGEM",
            f"{m['name']} @ {m['temperature']}°C",
            format_size(m["size"]),
            format_timestamp(m["modified"]),
            m["path"]
        ])
    
    return data


def view_model_details(model_path):
    """View details of a selected model"""
    if not model_path:
        return "No model selected"
    
    model_path = Path(model_path)
    
    if not model_path.exists():
        return "❌ Model file not found"
    
    details = f"📄 **Model:** {model_path.name}\n"
    details += f"📁 **Path:** {model_path}\n"
    details += f"📦 **Size:** {format_size(model_path.stat().st_size)}\n"
    details += f"🕒 **Modified:** {format_timestamp(model_path.stat().st_mtime)}\n\n"
    
    # For ecGEM/etcGEM, show additional stats
    if "ecGEM" in str(model_path) or "etcGEM" in str(model_path):
        folder = model_path.parent
        stats = ecgem_service.get_model_stats(str(folder))
        
        details += "### 📊 Model Statistics\n\n"
        
        for filename, info in stats.get("files", {}).items():
            if info.get("exists"):
                details += f"**{filename}:**\n"
                details += f"  - Size: {format_size(info['size'])}\n"
                if "rows" in info:
                    details += f"  - Rows: {info['rows']}\n"
                if "columns" in info:
                    details += f"  - Columns: {info['columns']}\n"
                details += "\n"
    
    return details


def model_viewer_tab():
    """Create Model Viewer tab"""
    with gr.Row():
        with gr.Column(scale=2):
            gr.Markdown("### 📚 All Models")
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            models_table = gr.Dataframe(
                headers=["Type", "Name", "Size", "Modified", "Path"],
                label="Available Models",
                interactive=False,
                wrap=True
            )
            
            refresh_btn.click(
                fn=list_all_models,
                outputs=[models_table]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Model Details")
            selected_path = gr.Textbox(label="Selected Model Path", visible=False)
            details_box = gr.Markdown("Select a model to view details")
            
            def on_model_select(evt: gr.SelectData):
                if evt and evt.value:
                    # Get the path from the selected row (last column)
                    return evt.value
                return ""
            
            models_table.select(
                fn=on_model_select,
                outputs=[selected_path]
            )
            
            selected_path.change(
                fn=view_model_details,
                inputs=[selected_path],
                outputs=[details_box]
            )


# ==================== Tab 5: Results Manager ====================

def list_result_folders():
    """List all result folders with statistics"""
    folders = []
    
    # ecGEM folders
    ecgem_dir = Path("src/GEMFactory/data/ecGEM")
    if ecgem_dir.exists():
        for folder in ecgem_dir.iterdir():
            if folder.is_dir():
                size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
                folders.append({
                    "type": "ecGEM",
                    "name": folder.name,
                    "path": str(folder),
                    "size": format_size(size),
                    "modified": format_timestamp(folder.stat().st_mtime)
                })
    
    # etcGEM folders
    etcgem_dir = Path("src/GEMFactory/data/etcGEM")
    if etcgem_dir.exists():
        for folder in etcgem_dir.iterdir():
            if folder.is_dir():
                size = sum(f.stat().st_size for f in folder.rglob('*') if f.is_file())
                folders.append({
                    "type": "etcGEM",
                    "name": folder.name,
                    "path": str(folder),
                    "size": format_size(size),
                    "modified": format_timestamp(folder.stat().st_mtime)
                })
    
    return [[f["type"], f["name"], f["size"], f["modified"], f["path"]] for f in folders]


def list_folder_files(folder_path):
    """List all files in a result folder"""
    if not folder_path:
        return []
    
    folder = Path(folder_path)
    if not folder.exists():
        return []
    
    files = []
    for file in sorted(folder.rglob('*')):
        if file.is_file():
            files.append([
                file.name,
                format_size(file.stat().st_size),
                format_timestamp(file.stat().st_mtime),
                str(file)
            ])
    
    return files


def results_manager_tab():
    """Create Results Manager tab"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📁 Result Folders")
            refresh_btn = gr.Button("🔄 Refresh", size="sm")
            
            folders_table = gr.Dataframe(
                headers=["Type", "Name", "Size", "Modified", "Path"],
                label="Result Folders",
                interactive=False
            )
            
            refresh_btn.click(
                fn=list_result_folders,
                outputs=[folders_table]
            )
        
        with gr.Column(scale=1):
            gr.Markdown("### 📄 Folder Contents")
            selected_folder = gr.Textbox(label="Selected Folder", visible=False)
            files_table = gr.Dataframe(
                headers=["File", "Size", "Modified", "Path"],
                label="Files",
                interactive=False
            )
            
            def on_folder_select(evt: gr.SelectData):
                if evt and evt.value:
                    # Get the path from the selected row (last column)
                    return evt.value
                return ""
            
            folders_table.select(
                fn=on_folder_select,
                outputs=[selected_folder]
            )
            
            selected_folder.change(
                fn=list_folder_files,
                inputs=[selected_folder],
                outputs=[files_table]
            )
            
            gr.Markdown("""
            ### 💡 Tips
            - Click on a folder to view its contents
            - All intermediate files are preserved
            - Key files:
              - `metabolites_reactions_gpr.csv` - Metabolite-reaction pairs
              - `full_metabolites_reactions.csv` - With predicted parameters
              - `reaction_kcat_mw.csv` - Final kcat/MW values
              - `ecModel.json` - Final ecGEM model
            """)


# ==================== Main Interface ====================

def gemfactory_tab():
    """Create the complete GEMFactory multi-tab interface"""
    
    with gr.Blocks() as interface:
        gr.Markdown("""
        # 🏭 GEMFactory - Genome-Scale Metabolic Model Factory
        
        **Build and enhance GEMs with state-of-the-art tools and constraints**
        """)
        
        with gr.Tabs() as tabs:
            with gr.Tab("🧬 Draft GEM"):
                gr.Markdown("""
                ### Build Draft GEM from Genome
                Upload a genome sequence and automatically build a draft metabolic model using:
                - **GeneMarkS** for gene annotation
                - **CarveMe** for GEM reconstruction
                """)
                draft_gem_tab()
            
            with gr.Tab("⚗️ ecGEM Builder"):
                gr.Markdown("""
                ### Build Enzyme-Constrained GEM
                Add enzyme kinetic constraints to your draft GEM:
                - Predicts kcat values using deep learning
                - Integrates protein allocation constraints
                - Improves flux predictions
                """)
                ecgem_builder_tab()
            
            with gr.Tab("🌡️ etcGEM Builder"):
                gr.Markdown("""
                ### Build Enzyme-Temperature-Constrained GEM
                Add both enzyme kinetics AND temperature constraints:
                - All ecGEM features
                - Temperature-dependent kinetics
                - Thermal adaptation modeling
                """)
                etcgem_builder_tab()
            
            with gr.Tab("🔍 Model Viewer"):
                gr.Markdown("""
                ### View and Analyze Models
                Browse all constructed models and view their statistics
                """)
                model_viewer_tab()
            
            with gr.Tab("📊 Results Manager"):
                gr.Markdown("""
                ### Manage Build Results
                Access all intermediate and final files from model construction
                """)
                results_manager_tab()
        
        gr.Markdown("""
        ---
        **Documentation:** See `TASK_LOGGING_SYSTEM_GUIDE.md` for usage details
        
        **Note:** All tasks run asynchronously in the background. Check the Tasks Monitor tab for overall progress.
        """)
    
    return interface


# For backward compatibility
def gemfactory_tab_legacy():
    """Legacy single-tab interface"""
    return gemfactory_tab()
