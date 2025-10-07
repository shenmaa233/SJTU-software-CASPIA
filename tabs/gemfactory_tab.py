"""GEMFactory Tab - Complete Multi-Tab Interface
==============================================

A comprehensive interface for:
1. Draft GEM building (GeneMarkS + CarveMe)
2. ecGEM construction (enzyme-constrained)
3. etcGEM construction (enzyme-temperature-constrained)
4. Model viewing and analysis
5. Download Manager

Author: SJTU-Software Team
Date: 2025-10
"""

import os
import re
import shutil
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import dotenv
import gradio as gr
import requests

from src.GEMFactory.src.build_GEM import clean_faa, run_carveme
from src.GEMFactory.src.ecGEM.ecgem_service import ECGEMService
from src.GEMFactory.src.utils.GeneMarkS import GeneMarkSRunner
from src.utils import get_task_manager

dotenv.load_dotenv()

# ==================== Constants ====================
GENOME_DIR = "src/GEMFactory/data/Genome"
CARVEME_DIR = "src/GEMFactory/data/CarveMe"
GMS_SCRIPT = os.environ.get("GMS_SCRIPT_PATH")

# ==================== Global Services ====================
task_manager = get_task_manager()
ecgem_service = ECGEMService()

# Global mapping from display name to full path for draft models
_model_path_mapping = {}

# Global cache for BiGG models
_bigg_models_cache = []

# ==================== Utility Functions ====================


def get_model_path_from_display(display_name: str) -> str:
    """Extract actual file path from display name

    Display format: "filename.xml (1.2 MB, 2025-10-06 12:00)"
    Returns the full path from the mapping.
    """
    if not display_name:
        return ""

    # Extract filename from display string (everything before the first '(')
    filename = display_name.split(" (")[0].strip()

    # Get full path from mapping
    return _model_path_mapping.get(filename, display_name)


def format_size(bytes_size):
    """Format file size in human-readable format"""
    for unit in ["B", "KB", "MB", "GB"]:
        if bytes_size < 1024.0:
            return f"{bytes_size:.1f} {unit}"
        bytes_size /= 1024.0
    return f"{bytes_size:.1f} TB"


def format_timestamp(timestamp):
    """Format Unix timestamp to readable date"""
    return datetime.fromtimestamp(timestamp).strftime("%Y-%m-%d %H:%M:%S")


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


# ==================== BiGG Database Functions ====================


def get_all_bigg_models():
    """Get the list of all models in the BiGG database"""
    api_url = "http://bigg.ucsd.edu/api/v2/models"
    try:
        response = requests.get(api_url, timeout=10)
        response.raise_for_status()
        data = response.json()
        return data
    except requests.exceptions.RequestException as e:
        print(f"Error occurred while requesting BiGG API: {e}")
        return None


def get_all_bigg_model_ids() -> List[str]:
    """Get the list of all model IDs in the BiGG database"""
    models_data = get_all_bigg_models()
    if models_data:
        all_models_list = models_data.get("results", [])
        return [model.get("bigg_id") for model in all_models_list if model.get("bigg_id")]
    return []


def load_bigg_models():
    """Load BiGG model list into cache"""
    global _bigg_models_cache
    if not _bigg_models_cache:
        models_data = get_all_bigg_models()
        if models_data:
            _bigg_models_cache = models_data.get("results", [])
    return _bigg_models_cache


def search_bigg_models(search_query: str):
    """Search BiGG models"""
    models = load_bigg_models()
    if not search_query:
        # Return all models
        filtered = models
    else:
        # Search by bigg_id and organism
        search_lower = search_query.lower()
        filtered = [m for m in models if search_lower in m.get("bigg_id", "").lower() or search_lower in m.get("organism", "").lower()]

    # Format as table data
    data = []
    for model in filtered:
        data.append(
            [
                model.get("bigg_id", ""),
                model.get("organism", ""),
                str(model.get("metabolite_count", 0)),
                str(model.get("reaction_count", 0)),
                str(model.get("gene_count", 0)),
            ]
        )

    return data, data


def download_protein_sequences(model_id: str) -> Optional[str]:
    """Download protein sequences for a BiGG model and save as FASTA

    Args:
        model_id: BiGG model ID

    Returns:
        Path to saved FASTA file or None if failed

    """
    try:
        # Get all genes for this model
        genes_url = f"http://bigg.ucsd.edu/api/v2/models/{model_id}/genes"
        response = requests.get(genes_url, timeout=30)
        response.raise_for_status()
        genes_data = response.json()
        genes_list = genes_data.get("results", [])

        if not genes_list:
            print(f"No genes found for model {model_id}")
            return None

        # Create output directory
        protein_dir = Path(f"src/GEMFactory/data/GeneMarkS/{model_id}")
        protein_dir.mkdir(parents=True, exist_ok=True)

        # Output FASTA file
        fasta_path = protein_dir / f"{model_id}_protein_clean.fasta"

        protein_count = 0
        with open(fasta_path, "w") as fasta_file:
            for gene in genes_list:
                gene_id = gene.get("bigg_id")
                if not gene_id:
                    continue

                # Get detailed gene info including protein sequence
                gene_detail_url = f"http://bigg.ucsd.edu/api/v2/models/{model_id}/genes/{gene_id}"
                try:
                    gene_response = requests.get(gene_detail_url, timeout=10)
                    gene_response.raise_for_status()
                    gene_detail = gene_response.json()

                    protein_seq = gene_detail.get("protein_sequence")
                    if protein_seq:
                        # Write in FASTA format
                        fasta_file.write(f">{gene_id}\n")
                        # Split sequence into lines of 60 characters
                        fasta_file.writelines(f"{protein_seq[i : i + 60]}\n" for i in range(0, len(protein_seq), 60))
                        protein_count += 1
                except Exception as e:
                    print(f"Failed to get protein sequence for gene {gene_id}: {e}")
                    continue

        if protein_count > 0:
            print(f"Downloaded {protein_count} protein sequences for model {model_id}")
            return str(fasta_path)
        print(f"No protein sequences found for model {model_id}")
        return None

    except Exception as e:
        print(f"Error downloading protein sequences: {e}")
        return None


def download_bigg_model(model_id: str) -> str:
    """Download model from BiGG database"""
    if not model_id:
        return "❌ Please select a model"

    # Ensure output directory exists
    output_dir = Path(CARVEME_DIR)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Set download URL and output path
    download_url = f"http://bigg.ucsd.edu/static/models/{model_id}.xml"
    output_path = output_dir / f"{model_id}_draft.xml"

    status_message = ""

    try:
        # Step 1: Download model XML
        status_message += "📥 Downloading model XML file...\n"
        response = requests.get(download_url, timeout=30)
        response.raise_for_status()

        # Save file
        with open(output_path, "wb") as f:
            f.write(response.content)

        file_size = output_path.stat().st_size
        status_message += f"✅ Model XML saved: {output_path}\n"
        status_message += f"📦 File size: {format_size(file_size)}\n\n"

        # Step 2: Download protein sequences
        status_message += "🧬 Downloading protein sequences from BiGG API...\n"
        protein_file = download_protein_sequences(model_id)

        if protein_file:
            protein_size = Path(protein_file).stat().st_size
            status_message += f"✅ Protein sequences saved: {protein_file}\n"
            status_message += f"📦 File size: {format_size(protein_size)}\n\n"
            status_message += "🎉 Model and protein sequences downloaded successfully!\n"
            status_message += "✨ This model is now ready for ecGEM/etcGEM construction!"
        else:
            status_message += "⚠️ Could not download protein sequences.\n"
            status_message += "💡 This model can only be used for basic analysis, not ecGEM/etcGEM construction."

        return status_message

    except requests.exceptions.RequestException as e:
        return f"❌ Error occurred while downloading model {model_id}:\n{str(e)}"
    except Exception as e:
        return f"❌ Error occurred while saving file:\n{str(e)}"


# ==================== Tab 1: Draft GEM Builder ====================


def gem_pipeline_task(logger, genome_path: str, gapfill: str):
    """Background task for draft GEM construction"""
    logger.info("=" * 60)
    logger.info("🚀 Draft GEM Construction Pipeline Started")
    logger.info("=" * 60)
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
        tmpdir="src/GEMFactory/data/temp",
    )

    logger.info(f"\n✅ Draft GEM saved: {gem_output}")
    logger.info("=" * 60)
    logger.info("✅ Pipeline Complete!")
    logger.info("=" * 60)

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
        task_type="draft_gem",
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
    # Use internal Tabs to separate two build methods
    with gr.Tabs():
        # Tab 1.1: Build from Genome
        with gr.Tab("📝 Build from Genome"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Input")
                    genome_upload = gr.File(
                        label="Upload Genome File",
                        type="filepath",
                        file_types=[".fna", ".fa", ".fasta"],
                    )
                    existing = list_genomes()
                    genome_dropdown = gr.Dropdown(
                        choices=existing,
                        value=(existing[0] if existing else None),
                        label="Or select existing genome",
                        interactive=True,
                    )
                    gapfill_dropdown = gr.Dropdown(
                        choices=["None", "M9", "LB", "M9,LB"],
                        value="None",
                        label="Gap-filling Medium",
                    )

                    genome_upload.change(
                        fn=save_and_refresh_genome,
                        inputs=[genome_upload],
                        outputs=[genome_dropdown],
                    )

                    gr.Markdown("### ⚙️ Actions")
                    run_btn = gr.Button("🚀 Build Draft GEM", variant="primary", size="lg")

                    gr.Markdown("### ℹ️ Instructions")
                    gr.Markdown("""
                    **Workflow Steps:**
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
                        outputs=[sid_state, status_box, logs_box],
                    )

                    timer = gr.Timer(2.0)
                    timer.tick(
                        fn=poll_draft_gem,
                        inputs=[sid_state],
                        outputs=[logs_box, status_box, result_box],
                    )

        # Tab 1.2: Download from BiGG
        with gr.Tab("🌐 Download from BiGG Database"):
            with gr.Row():
                with gr.Column(scale=1):
                    gr.Markdown("### 🔍 Search Models")
                    search_box = gr.Textbox(
                        label="Search Keyword",
                        placeholder="Enter model ID or organism name...",
                        interactive=True,
                    )
                    search_btn = gr.Button("🔍 Search", size="sm")
                    load_all_btn = gr.Button("📋 Load All Models", size="sm")

                    gr.Markdown("### ℹ️ Instructions")
                    gr.Markdown("""
                    **BiGG Database:**
                    - Contains many published high-quality metabolic models
                    - Covers various microbes and model organisms
                    - Direct download in SBML format

                    **How to use:**
                    1. Enter a keyword to search or load all models
                    2. Click to select a model in the table
                    3. Click the "Download Model" button

                    **Note:** Downloaded models will be automatically saved to the CarveMe folder, with file name format `{model_id}_draft.xml`
                    """)

                with gr.Column(scale=2):
                    gr.Markdown("### 📊 Search Results")
                    models_table = gr.Dataframe(
                        headers=["Model ID", "Organism", "Metabolite Count", "Reaction Count", "Gene Count"],
                        label="Available Models",
                        interactive=False,
                        wrap=True,
                    )

                    selected_model_id = gr.Textbox(
                        label="Selected Model ID",
                        interactive=False,
                    )

                    # Store current search results in state
                    search_results_state = gr.State([])

                    download_btn = gr.Button("📥 Download Model", variant="primary", size="lg")
                    download_status = gr.Textbox(
                        label="Download Status",
                        lines=5,
                        interactive=False,
                    )

                    # Bind events
                    search_btn.click(
                        fn=search_bigg_models,
                        inputs=[search_box],
                        outputs=[models_table, search_results_state],
                    )

                    load_all_btn.click(
                        fn=search_bigg_models,
                        inputs=[gr.Textbox(value="", visible=False)],
                        outputs=[models_table, search_results_state],
                    )

                    def on_model_select(evt: gr.SelectData, search_results):
                        """When a user selects a row in the table, get the model ID"""
                        if evt and evt.index[0] is not None:
                            row_idx = evt.index[0]
                            # Get the model ID from the first column (index 0) of the selected row
                            if 0 <= row_idx < len(search_results):
                                row_data = search_results[row_idx]
                                if row_data and len(row_data) > 0:
                                    return row_data[0]  # Return the model ID from column 0
                        return ""

                    models_table.select(
                        fn=on_model_select,
                        inputs=[search_results_state],
                        outputs=[selected_model_id],
                    )

                    download_btn.click(
                        fn=download_bigg_model,
                        inputs=[selected_model_id],
                        outputs=[download_status],
                    )


# ==================== Tab 2: ecGEM Builder ====================


def refresh_draft_models():
    """Refresh list of draft models, show file name with metadata"""
    models = ecgem_service.list_draft_models()
    # Create display labels with file name, size, and timestamp
    choices = [f"{Path(m['path']).name} ({format_size(m['size'])}, {format_timestamp(m['modified'])})" for m in models]
    # Store mapping from display name to full path
    global _model_path_mapping
    _model_path_mapping = {Path(m["path"]).name: m["path"] for m in models}
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
        lowerbound=lowerbound,
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
                interactive=True,
            )

            refresh_btn.click(
                fn=refresh_draft_models,
                outputs=[model_dropdown],
            )

            check_btn = gr.Button("🔍 Check Suitability", size="sm")
            suitability_box = gr.Textbox(
                label="Suitability Check",
                lines=6,
                interactive=False,
            )

            check_btn.click(
                fn=check_model_suitability,
                inputs=[model_dropdown],
                outputs=[suitability_box],
            )

            gr.Markdown("### ⚙️ Parameters")
            f_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.405,
                step=0.01,
                label="f (Enzyme Fraction)",
                info="Fraction of enzymes with available kcat",
            )
            ptot_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.56,
                step=0.01,
                label="Ptot (Total Protein)",
                info="Total protein fraction (g/gDW)",
            )
            sigma_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="σ (Saturation)",
                info="Average enzyme saturation factor",
            )
            lowerbound_slider = gr.Slider(
                minimum=0.0,
                maximum=0.1,
                value=0.0,
                step=0.01,
                label="Lower Bound",
                info="Lower bound for enzyme constraints",
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
                outputs=[sid_state, status_box, logs_box],
            )

            timer = gr.Timer(2.0)
            timer.tick(
                fn=poll_ecgem,
                inputs=[sid_state],
                outputs=[logs_box, status_box, result_box],
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
        lowerbound=lowerbound,
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
                interactive=True,
            )

            refresh_btn.click(
                fn=refresh_draft_models,
                outputs=[model_dropdown],
            )

            check_btn = gr.Button("🔍 Check Suitability", size="sm")
            suitability_box = gr.Textbox(
                label="Suitability Check",
                lines=4,
                interactive=False,
            )

            check_btn.click(
                fn=check_model_suitability,
                inputs=[model_dropdown],
                outputs=[suitability_box],
            )

            gr.Markdown("### 🌡️ Temperature")
            temp_slider = gr.Slider(
                minimum=0,
                maximum=100,
                value=37.0,
                step=0.5,
                label="Temperature (°C)",
                info="Optimal growth temperature",
            )

            gr.Markdown("### ⚙️ Parameters")
            f_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.405,
                step=0.01,
                label="f (Enzyme Fraction)",
            )
            ptot_slider = gr.Slider(
                minimum=0.1,
                maximum=1.0,
                value=0.56,
                step=0.01,
                label="Ptot (Total Protein)",
            )
            sigma_slider = gr.Slider(
                minimum=0.1,
                maximum=2.0,
                value=1.0,
                step=0.1,
                label="σ (Saturation)",
            )
            lowerbound_slider = gr.Slider(
                minimum=0.0,
                maximum=0.1,
                value=0.0,
                step=0.01,
                label="Lower Bound",
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
                outputs=[sid_state, status_box, logs_box],
            )

            timer = gr.Timer(2.0)
            timer.tick(
                fn=poll_ecgem,
                inputs=[sid_state],
                outputs=[logs_box, status_box, result_box],
            )


# ==================== Tab 4: Model Viewer ====================


def list_all_models():
    """List all models (Draft, ecGEM, etcGEM)"""
    draft_models = ecgem_service.list_draft_models()
    ecgem_models = ecgem_service.list_ecgem_models()
    etcgem_models = ecgem_service.list_etcgem_models()

    data = []

    for m in draft_models:
        data.append(
            [
                "Draft GEM",
                m["name"],
                format_size(m["size"]),
                format_timestamp(m["modified"]),
                m["path"],
            ]
        )

    for m in ecgem_models:
        data.append(
            [
                "ecGEM",
                m["name"],
                format_size(m["size"]),
                format_timestamp(m["modified"]),
                m["path"],
            ]
        )

    for m in etcgem_models:
        data.append(
            [
                "etcGEM",
                f"{m['name']} @ {m['temperature']}°C",
                format_size(m["size"]),
                format_timestamp(m["modified"]),
                m["path"],
            ]
        )

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
                wrap=True,
            )

            refresh_btn.click(
                fn=list_all_models,
                outputs=[models_table],
            )

        with gr.Column(scale=1):
            gr.Markdown("### 🔍 Model Details")
            selected_path = gr.Textbox(label="Selected Model Path", visible=False)
            details_box = gr.Markdown("Select a model to view details")

            gr.Markdown("""
            ### 📥 Download Models
            To download models, go to the **"Download Manager"** tab.
            """)

            def on_model_select(evt: gr.SelectData):
                if evt and evt.value:
                    # Get the path from the selected row (last column)
                    return evt.value
                return ""

            models_table.select(
                fn=on_model_select,
                outputs=[selected_path],
            )

            selected_path.change(
                fn=view_model_details,
                inputs=[selected_path],
                outputs=[details_box],
            )


# ==================== Tab 5: KO/OE Analysis ====================


def start_ko_oe_analysis(selected_model, analysis_type, target_reaction, knockout_threshold, oe_fold_changes_text, _sid):
    """Start KO/OE analysis task"""
    global _analysis_model_map

    if not selected_model or selected_model not in _analysis_model_map:
        return "", "❌ Please select a model", "", "No results yet"

    if not analysis_type:
        return "", "❌ Please select an analysis type", "", "No results yet"

    model_info = _analysis_model_map[selected_model]
    model_path = model_info["path"]

    # Parse fold changes
    oe_fold_changes = None
    if analysis_type in ["overexpression", "comprehensive"]:
        try:
            oe_fold_changes = [float(x.strip()) for x in oe_fold_changes_text.split(",") if x.strip()]
            if not oe_fold_changes:
                oe_fold_changes = [2.0, 5.0, 10.0]
        except:
            oe_fold_changes = [2.0, 5.0, 10.0]

    try:
        task_id = ecgem_service.run_ko_oe_analysis(
            model_file=model_path,
            analysis_type=analysis_type,
            target_ids=None,  # Analyze all by default
            production_target=target_reaction if target_reaction else None,
            knockout_threshold=knockout_threshold,
            oe_fold_changes=oe_fold_changes,
        )

        logs, status, _ = task_manager.poll(task_id)
        return task_id, status, logs, "Analysis running..."
    except Exception as e:
        return "", f"❌ Failed to start: {str(e)}", "", "No results yet"


def poll_ko_oe_analysis(sid: str):
    """Poll KO/OE analysis task status"""
    if not sid:
        return "", "Waiting to start...", "No results yet"

    logs, status, result = task_manager.poll(sid)

    # If task is complete, extract results
    if "✅" in status or "Complete" in status:
        result_summary = extract_ko_oe_results(logs, result)
        return logs, status, result_summary

    return logs, status, "Analysis in progress..."


def extract_ko_oe_results(logs: str, result: str) -> str:
    """Extract key results from KO/OE analysis logs"""
    summary = "📊 **KO/OE Analysis Results:**\n\n"

    # Extract statistics from logs
    reaction_match = re.search(r"Reaction knockouts:\s*(\d+)", logs)
    gene_match = re.search(r"Gene knockouts:\s*(\d+)", logs)
    essential_rxn_match = re.search(r"Essential reactions:\s*(\d+)", logs)
    essential_gene_match = re.search(r"Essential genes:\s*(\d+)", logs)
    oe_match = re.search(r"Overexpression scenarios:\s*(\d+)", logs)

    if reaction_match:
        summary += f"🔬 **Reaction Knockouts Analyzed:** {reaction_match.group(1)}\n"
    if gene_match:
        summary += f"🧬 **Gene Knockouts Analyzed:** {gene_match.group(1)}\n"
    if essential_rxn_match:
        summary += f"⚠️ **Essential Reactions Found:** {essential_rxn_match.group(1)}\n"
    if essential_gene_match:
        summary += f"⚠️ **Essential Genes Found:** {essential_gene_match.group(1)}\n"
    if oe_match:
        summary += f"📈 **Overexpression Scenarios:** {oe_match.group(1)}\n"

    # Extract result folder
    if result:
        summary += f"\n📁 **Results Folder:** `{result}`\n"

    # Add download instruction
    summary += "\n💾 **To download:** Go to **'Download Manager'** tab and refresh to see the latest results.\n"

    return summary if any([reaction_match, gene_match, oe_match]) else "Analysis completed. Check logs for details."


def ko_oe_analysis_tab():
    """Create KO/OE Analysis tab"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Model Selection")

            refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=[],
                interactive=True,
            )

            refresh_btn.click(
                fn=refresh_analysis_models,
                outputs=[model_dropdown],
            )

            gr.Markdown("### ⚙️ Analysis Type")
            analysis_type_dropdown = gr.Dropdown(
                label="Analysis Type",
                choices=[
                    "knockout_reaction",
                    "knockout_gene",
                    "overexpression",
                    "comprehensive",
                ],
                value="comprehensive",
                interactive=True,
            )

            gr.Markdown("""
            **Analysis Types:**
            - **knockout_reaction**: Test single reaction knockouts
            - **knockout_gene**: Test single gene knockouts
            - **overexpression**: Simulate gene overexpression
            - **comprehensive**: All of the above
            """)

            gr.Markdown("### 🎯 Parameters")

            target_reaction_box = gr.Textbox(
                label="Target Reaction (Optional)",
                placeholder="Leave empty to use biomass objective",
                value="",
            )

            knockout_threshold_slider = gr.Slider(
                minimum=0.0,
                maximum=0.5,
                value=0.01,
                step=0.01,
                label="Knockout Threshold",
                info="Minimum change to consider essential (1% = 0.01)",
            )

            oe_fold_changes_box = gr.Textbox(
                label="Overexpression Fold Changes",
                value="2.0, 5.0, 10.0",
                info="Comma-separated values (e.g., 2.0, 5.0, 10.0)",
            )

            run_btn = gr.Button("🚀 Run KO/OE Analysis", variant="primary", size="lg")

            gr.Markdown("### ℹ️ Instructions")
            gr.Markdown("""
            **How to use:**
            1. Select a GEM model (Draft/ecGEM/etcGEM)
            2. Choose analysis type
            3. (Optional) Set target reaction for optimization
            4. Configure parameters
            5. Click "Run KO/OE Analysis"

            **Notes:**
            - Comprehensive analysis may take longer
            - Results include CSV files for download
            - Essential genes/reactions are automatically identified
            """)

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Status")

            task_id_box = gr.Textbox(label="Task ID", interactive=False, visible=False)
            status_box = gr.Textbox(
                label="Status",
                value="Waiting to start...",
                interactive=False,
                lines=2,
            )

            gr.Markdown("### 📈 Results Summary")
            result_summary = gr.Textbox(
                label="Analysis Results",
                value="No results yet",
                interactive=False,
                lines=6,
            )

            gr.Markdown("### 📋 Logs")
            logs_box = gr.Textbox(
                label="Logs",
                lines=15,
                interactive=False,
                max_lines=20,
            )

            gr.Markdown("""
            ### 📥 Download Results

            After analysis completes, go to **"Download Manager"** tab to download:
            - Knockout reaction results (CSV)
            - Knockout gene results (CSV)
            - Essential reactions/genes lists (CSV)
            - Overexpression results (CSV)

            ### 📊 Output Files

            **For Comprehensive Analysis:**
            - `knockout_reaction_results.csv`: All reaction knockout results
            - `knockout_gene_results.csv`: All gene knockout results
            - `essential_reactions.csv`: List of essential reactions
            - `essential_genes.csv`: List of essential genes
            - `overexpression_results.csv`: Overexpression simulation results
            - `cobrapy_reaction_deletion_results.csv`: COBRApy reaction deletion
            - `cobrapy_gene_deletion_results.csv`: COBRApy gene deletion
            """)

    # Event handlers
    run_btn.click(
        fn=start_ko_oe_analysis,
        inputs=[model_dropdown, analysis_type_dropdown, target_reaction_box, knockout_threshold_slider, oe_fold_changes_box, task_id_box],
        outputs=[task_id_box, status_box, logs_box, result_summary],
    )

    # Auto-polling for task status
    timer = gr.Timer(2.0)
    timer.tick(
        fn=poll_ko_oe_analysis,
        inputs=[task_id_box],
        outputs=[logs_box, status_box, result_summary],
    )


# ==================== Tab 6: FBA/FVA/pFBA Analysis ====================


def get_all_models_for_analysis():
    """Get all available models (draft, ecGEM, etcGEM) formatted for dropdown"""
    draft_models = ecgem_service.list_draft_models()
    ecgem_models = ecgem_service.list_ecgem_models()
    etcgem_models = ecgem_service.list_etcgem_models()

    choices = []
    model_info_map = {}

    for m in draft_models:
        display_name = f"[Draft] {m['name']}"
        choices.append(display_name)
        model_info_map[display_name] = {
            "type": "draft",
            "path": m["path"],
            "name": m["name"],
        }

    for m in ecgem_models:
        display_name = f"[ecGEM] {m['name']}"
        choices.append(display_name)
        model_info_map[display_name] = {
            "type": "ecGEM",
            "path": m["path"],
            "name": m["name"],
        }

    for m in etcgem_models:
        temp = m.get("temperature", "N/A")
        display_name = f"[etcGEM] {m['name']} @ {temp}°C"
        choices.append(display_name)
        model_info_map[display_name] = {
            "type": "etcGEM",
            "path": m["path"],
            "name": m["name"],
            "temperature": temp,
        }

    return choices, model_info_map


# Global mapping for model analysis
_analysis_model_map = {}


def refresh_analysis_models():
    """Refresh available models for analysis"""
    global _analysis_model_map
    choices, _analysis_model_map = get_all_models_for_analysis()
    return gr.Dropdown(choices=choices)


def update_algorithm_choices(selected_model):
    """Update available algorithms based on selected model type"""
    global _analysis_model_map

    if not selected_model or selected_model not in _analysis_model_map:
        return gr.Dropdown(choices=[], value=None), gr.update(visible=False), gr.update(visible=False)

    model_info = _analysis_model_map[selected_model]
    model_type = model_info["type"]

    if model_type == "draft":
        # Draft GEM only supports FBA, pFBA, FVA
        algorithms = ["FBA", "pFBA", "FVA"]
        show_ecgem_params = False
    else:
        # ecGEM and etcGEM only support ecGEM algorithm
        algorithms = ["ecGEM"]
        show_ecgem_params = True

    return (
        gr.Dropdown(choices=algorithms, value=algorithms[0]),
        gr.update(visible=show_ecgem_params),
        gr.update(visible=show_ecgem_params),
    )


def start_gem_analysis(selected_model, algorithm, obj, substrate, concentration, _sid):
    """Start GEM analysis task"""
    global _analysis_model_map

    if not selected_model or selected_model not in _analysis_model_map:
        return "", "❌ Please select a model", "", "No results yet"

    if not algorithm:
        return "", "❌ Please select an algorithm", "", "No results yet"

    model_info = _analysis_model_map[selected_model]
    model_path = model_info["path"]

    # Validate algorithm-model type compatibility
    model_type = model_info["type"]
    if model_type == "draft" and algorithm == "ecGEM":
        return "", "❌ Draft GEM does not support ecGEM algorithm. Please use FBA/pFBA/FVA", "", "No results yet"
    if model_type in ["ecGEM", "etcGEM"] and algorithm in ["FBA", "pFBA", "FVA"]:
        return "", f"❌ {model_type} does not support {algorithm} algorithm. Please use ecGEM algorithm", "", "No results yet"

    try:
        task_id = ecgem_service.run_gem_analysis(
            model_file=model_path,
            algorithm=algorithm,
            obj=obj if obj else None,
            substrate=substrate,
            concentration=concentration,
        )

        logs, status, _ = task_manager.poll(task_id)
        return task_id, status, logs, "Analysis running..."
    except Exception as e:
        return "", f"❌ Failed to start: {str(e)}", "", "No results yet"


def poll_gem_analysis(sid: str):
    """Poll GEM analysis task status"""
    if not sid:
        return "", "Waiting to start...", "No results yet"

    logs, status, result = task_manager.poll(sid)

    # If task is complete, extract results
    if "✅" in status or "Complete" in status:
        result_summary = extract_analysis_results(logs, result)
        return logs, status, result_summary

    return logs, status, "Analysis in progress..."


def extract_analysis_results(logs: str, result: str) -> str:
    """Extract key results from analysis logs"""
    summary = "📊 **Analysis Results:**\n\n"

    # Extract optimal value
    optimal_match = re.search(r"Optimal value.*?:\s*([-\d.]+)", logs)
    if optimal_match:
        optimal_value = optimal_match.group(1)
        summary += f"🎯 **Optimal Objective Value:** {optimal_value}\n"

    # Extract result file location
    result_match = re.search(r"Results saved to:\s*(.+?)(?:\n|$)", logs)
    if result_match:
        result_file = result_match.group(1).strip()
        summary += f"📁 **Result File:** `{result_file}`\n"
    elif result:
        summary += f"📁 **Result File:** `{result}`\n"

    # Add download instruction
    summary += "\n💾 **To download:** Go to **'Download Manager'** tab and refresh to see the latest results.\n"

    # Check for errors
    if "error" in logs.lower() or "failed" in logs.lower():
        summary += "\n⚠️ **Warning:** Some errors occurred during analysis. Check logs for details.\n"

    return summary if optimal_match or result_match else "Analysis completed. Check logs for details."


def gem_analysis_tab():
    """Create GEM Analysis tab"""
    with gr.Row():
        with gr.Column(scale=1):
            gr.Markdown("### 📥 Model Selection")

            refresh_btn = gr.Button("🔄 Refresh Models", size="sm")
            model_dropdown = gr.Dropdown(
                label="Select Model",
                choices=[],
                interactive=True,
            )

            refresh_btn.click(
                fn=refresh_analysis_models,
                outputs=[model_dropdown],
            )

            gr.Markdown("### ⚙️ Algorithm Selection")
            algorithm_dropdown = gr.Dropdown(
                label="Analysis Algorithm",
                choices=[],
                interactive=True,
            )

            gr.Markdown("""
            **Algorithm Description:**
            - **FBA**: Flux Balance Analysis (Draft GEM only)
            - **pFBA**: Parsimonious FBA (Draft GEM only)
            - **FVA**: Flux Variability Analysis (Draft GEM only)
            - **ecGEM**: Enzyme-constrained pFBA (ecGEM/etcGEM only)
            """)

            gr.Markdown("### 🎯 Parameters")
            obj_textbox = gr.Textbox(
                label="Target Reaction ID (Optional)",
                placeholder="Leave empty to auto-detect biomass reaction",
                value="",
            )

            # ecGEM specific parameters
            ecgem_params = gr.Column(visible=False)
            with ecgem_params:
                substrate_textbox = gr.Textbox(
                    label="Substrate Reaction ID",
                    value="EX_glc__D_e",
                    info="For ecGEM algorithm only",
                )
                concentration_slider = gr.Slider(
                    minimum=0.1,
                    maximum=50,
                    value=10,
                    step=0.1,
                    label="Substrate Concentration",
                    info="For ecGEM algorithm only",
                )

            model_dropdown.change(
                fn=update_algorithm_choices,
                inputs=[model_dropdown],
                outputs=[algorithm_dropdown, ecgem_params, ecgem_params],
            )

            run_btn = gr.Button("🚀 Run Analysis", variant="primary", size="lg")

        with gr.Column(scale=2):
            gr.Markdown("### 📊 Analysis Status")

            task_id_box = gr.Textbox(label="Task ID", interactive=False, visible=False)
            status_box = gr.Textbox(
                label="Status",
                value="Waiting to start...",
                interactive=False,
                lines=2,
            )

            gr.Markdown("### 📈 Results Summary")
            result_summary = gr.Textbox(
                label="Analysis Results",
                value="No results yet",
                interactive=False,
                lines=4,
            )

            gr.Markdown("""
            ### 📥 Download Results
            After analysis completes, go to **"Download Manager"** tab to download:
            - Model files (Draft GEM, ecGEM, etcGEM)
            - Analysis result CSV files
            """)

            gr.Markdown("### 📋 Logs")
            logs_box = gr.Textbox(
                label="Logs",
                lines=15,
                interactive=False,
                max_lines=20,
            )

            gr.Markdown("""
            ### ℹ️ Instructions

            **Steps:**
            1. 🔄 Click "Refresh Models" to load available models
            2. 📥 Select a model to analyze
            3. ⚙️ System will automatically show compatible algorithms
            4. 🎯 Configure parameters (optional)
            5. 🚀 Click "Run Analysis"
            6. 📥 Download results when complete

            **Notes:**
            - Draft GEM only supports FBA/pFBA/FVA
            - ecGEM/etcGEM only support ecGEM algorithm
            - Results will be saved in model directory's `analysis_result` folder
            """)

    # Event handlers
    run_btn.click(
        fn=start_gem_analysis,
        inputs=[model_dropdown, algorithm_dropdown, obj_textbox, substrate_textbox, concentration_slider, task_id_box],
        outputs=[task_id_box, status_box, logs_box, result_summary],
    )

    # Auto-polling for task status using Timer (consistent with other tabs)
    timer = gr.Timer(2.0)
    timer.tick(
        fn=poll_gem_analysis,
        inputs=[task_id_box],
        outputs=[logs_box, status_box, result_summary],
    )


# ==================== Tab 6: Download Manager ====================


def list_all_downloadable_models():
    """List all GEM models available for download"""
    models = []

    # Draft GEMs
    carveme_dir = Path("src/GEMFactory/data/CarveMe")
    if carveme_dir.exists():
        for file in carveme_dir.glob("*_draft.xml"):
            models.append(
                {
                    "type": "Draft GEM",
                    "name": file.stem.replace("_draft", ""),
                    "filename": file.name,
                    "size": format_size(file.stat().st_size),
                    "modified": format_timestamp(file.stat().st_mtime),
                    "path": str(file),
                }
            )

    # ecGEMs
    ecgem_dir = Path("src/GEMFactory/data/ecGEM")
    if ecgem_dir.exists():
        for folder in ecgem_dir.iterdir():
            if folder.is_dir():
                model_file = folder / "ecModel.json"
                if model_file.exists():
                    models.append(
                        {
                            "type": "ecGEM",
                            "name": folder.name,
                            "filename": "ecModel.json",
                            "size": format_size(model_file.stat().st_size),
                            "modified": format_timestamp(model_file.stat().st_mtime),
                            "path": str(model_file),
                        }
                    )

    # etcGEMs
    etcgem_dir = Path("src/GEMFactory/data/etcGEM")
    if etcgem_dir.exists():
        for folder in etcgem_dir.iterdir():
            if folder.is_dir():
                model_file = folder / "ecModel.json"
                if model_file.exists():
                    # Extract temperature from folder name
                    temp_match = re.search(r"T=([\d.]+)", folder.name)
                    temp = temp_match.group(1) if temp_match else "N/A"
                    models.append(
                        {
                            "type": "etcGEM",
                            "name": f"{folder.name.split('_T=')[0]} @ {temp}°C",
                            "filename": "ecModel.json",
                            "size": format_size(model_file.stat().st_size),
                            "modified": format_timestamp(model_file.stat().st_mtime),
                            "path": str(model_file),
                        }
                    )

    # Sort by modification time (newest first)
    models.sort(key=lambda x: x["modified"], reverse=True)

    return [[m["type"], m["name"], m["filename"], m["size"], m["modified"], m["path"]] for m in models]


def list_analysis_results():
    """List all analysis result files available for download"""
    results = []

    # Look for analysis results in ecGEM folders
    ecgem_dir = Path("src/GEMFactory/data/ecGEM")
    if ecgem_dir.exists():
        for folder in ecgem_dir.iterdir():
            if folder.is_dir():
                # FBA/pFBA/FVA results
                result_dir = folder / "analysis_result"
                if result_dir.exists():
                    for file in result_dir.glob("*.csv"):
                        results.append(
                            {
                                "model": folder.name,
                                "type": "ecGEM Analysis",
                                "filename": file.name,
                                "size": format_size(file.stat().st_size),
                                "modified": format_timestamp(file.stat().st_mtime),
                                "path": str(file),
                            }
                        )

                # KO/OE results
                ko_oe_dir = folder / "ko_oe_analysis"
                if ko_oe_dir.exists():
                    for file in ko_oe_dir.glob("*.csv"):
                        results.append(
                            {
                                "model": folder.name,
                                "type": "KO/OE Analysis",
                                "filename": file.name,
                                "size": format_size(file.stat().st_size),
                                "modified": format_timestamp(file.stat().st_mtime),
                                "path": str(file),
                            }
                        )

    # Look for analysis results in etcGEM folders
    etcgem_dir = Path("src/GEMFactory/data/etcGEM")
    if etcgem_dir.exists():
        for folder in etcgem_dir.iterdir():
            if folder.is_dir():
                # FBA/pFBA/FVA results
                result_dir = folder / "analysis_result"
                if result_dir.exists():
                    for file in result_dir.glob("*.csv"):
                        results.append(
                            {
                                "model": folder.name,
                                "type": "etcGEM Analysis",
                                "filename": file.name,
                                "size": format_size(file.stat().st_size),
                                "modified": format_timestamp(file.stat().st_mtime),
                                "path": str(file),
                            }
                        )

                # KO/OE results
                ko_oe_dir = folder / "ko_oe_analysis"
                if ko_oe_dir.exists():
                    for file in ko_oe_dir.glob("*.csv"):
                        results.append(
                            {
                                "model": folder.name,
                                "type": "KO/OE Analysis",
                                "filename": file.name,
                                "size": format_size(file.stat().st_size),
                                "modified": format_timestamp(file.stat().st_mtime),
                                "path": str(file),
                            }
                        )

    # Look for analysis results in CarveMe folders
    carveme_dir = Path("src/GEMFactory/data/CarveMe")
    if carveme_dir.exists():
        # FBA/pFBA/FVA results
        result_dir = carveme_dir / "analysis_result"
        if result_dir.exists():
            for file in result_dir.glob("*.csv"):
                results.append(
                    {
                        "model": "Draft GEM",
                        "type": "FBA/pFBA/FVA",
                        "filename": file.name,
                        "size": format_size(file.stat().st_size),
                        "modified": format_timestamp(file.stat().st_mtime),
                        "path": str(file),
                    }
                )

        # KO/OE results
        ko_oe_dir = carveme_dir / "ko_oe_analysis"
        if ko_oe_dir.exists():
            for file in ko_oe_dir.glob("*.csv"):
                results.append(
                    {
                        "model": "Draft GEM",
                        "type": "KO/OE Analysis",
                        "filename": file.name,
                        "size": format_size(file.stat().st_size),
                        "modified": format_timestamp(file.stat().st_mtime),
                        "path": str(file),
                    }
                )

    # Sort by modification time (newest first)
    results.sort(key=lambda x: x["modified"], reverse=True)

    return [[r["model"], r["type"], r["filename"], r["size"], r["modified"], r["path"]] for r in results]


def results_manager_tab():
    """Create Download Manager tab for models and analysis results"""
    with gr.Tabs():
        # Tab 6.1: GEM Models Download
        with gr.Tab("📦 GEM Models"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 🧬 Available GEM Models")
                    refresh_models_btn = gr.Button("🔄 Refresh Models", size="sm")

                    models_table = gr.Dataframe(
                        headers=["Type", "Model Name", "Filename", "Size", "Modified", "Path"],
                        label="GEM Models",
                        interactive=False,
                        wrap=True,
                    )

                    refresh_models_btn.click(
                        fn=list_all_downloadable_models,
                        outputs=[models_table],
                    )

                    gr.Markdown("""
                    ### 💡 Instructions
                    1. Click **"Refresh Models"** to load all available models
                    2. Click on any row to select a model
                    3. Click **"Download Selected Model"** to download

                    **Model Types:**
                    - **Draft GEM**: Basic metabolic models (.xml)
                    - **ecGEM**: Enzyme-constrained models (.json)
                    - **etcGEM**: Enzyme-temperature-constrained models (.json)
                    """)

                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Download")

                    selected_model_path = gr.Textbox(
                        label="Selected Model Path",
                        interactive=False,
                        visible=False,
                    )

                    model_info_box = gr.Markdown(
                        "**No model selected**\n\nClick on a model in the table to select it.",
                    )

                    download_model_btn = gr.File(
                        label="📦 Download Selected Model",
                        visible=False,
                    )

                    def on_model_row_select(evt: gr.SelectData):
                        """Handle model row selection"""
                        if evt and evt.value:
                            return evt.value
                        return ""

                    def update_model_download(path):
                        """Update download button and info when model is selected"""
                        if not path or not os.path.exists(path):
                            return (
                                "**No model selected**\n\nClick on a model in the table to select it.",
                                gr.update(visible=False),
                            )

                        # Get file info
                        file_path = Path(path)

                        # Determine model name based on path
                        if "ecGEM" in str(path) or "etcGEM" in str(path):
                            # For ecGEM/etcGEM, use parent folder name as model name
                            model_name = file_path.parent.name
                        else:
                            # For draft GEM, use filename without _draft.xml
                            model_name = file_path.stem.replace("_draft", "")

                        info = f"""
                                **Selected Model:**
                                - **Model Name:** `{model_name}`
                                - **Filename:** `{file_path.name}`
                                - **Size:** {format_size(file_path.stat().st_size)}
                                - **Path:** `{path}`

                                Click the download button below to download this model.
                                """
                        return info, gr.update(visible=True, value=path)

                    models_table.select(
                        fn=on_model_row_select,
                        outputs=[selected_model_path],
                    )

                    selected_model_path.change(
                        fn=update_model_download,
                        inputs=[selected_model_path],
                        outputs=[model_info_box, download_model_btn],
                    )

        # Tab 6.2: Analysis Results Download
        with gr.Tab("📊 Analysis Results"):
            with gr.Row():
                with gr.Column(scale=2):
                    gr.Markdown("### 📈 Analysis Result Files")
                    refresh_results_btn = gr.Button("🔄 Refresh Results", size="sm")

                    results_table = gr.Dataframe(
                        headers=["Model", "Analysis Type", "Filename", "Size", "Modified", "Path"],
                        label="Analysis Results",
                        interactive=False,
                        wrap=True,
                    )

                    refresh_results_btn.click(
                        fn=list_analysis_results,
                        outputs=[results_table],
                    )

                    gr.Markdown("""
                            ### 💡 Instructions
                            1. Click **"Refresh Results"** to load all analysis results
                            2. Click on any row to select a result file
                            3. Click **"Download Selected Result"** to download

                            **Result Types:**
                            - **ecGEM Analysis**: Enzyme-constrained pFBA results
                            - **FBA/pFBA/FVA**: Flux balance analysis results
                            - **KO/OE Analysis**: Knockout and overexpression analysis results
                            - All results are in CSV format
                            """)

                with gr.Column(scale=1):
                    gr.Markdown("### 📥 Download")

                    selected_result_path = gr.Textbox(
                        label="Selected Result Path",
                        interactive=False,
                        visible=False,
                    )

                    result_info_box = gr.Markdown(
                        "**No result selected**\n\nClick on a result file in the table to select it.",
                    )

                    download_result_btn = gr.File(
                        label="📊 Download Selected Result",
                        visible=False,
                    )

                    def on_result_row_select(evt: gr.SelectData):
                        """Handle result row selection"""
                        if evt and evt.value:
                            return evt.value
                        return ""

                    def update_result_download(path):
                        """Update download button and info when result is selected"""
                        if not path or not os.path.exists(path):
                            return (
                                "**No result selected**\n\nClick on a result file in the table to select it.",
                                gr.update(visible=False),
                            )

                        # Get file info
                        file_path = Path(path)
                        info = f"""
                        **Selected Result:**
                        - **Filename:** `{file_path.name}`
                        - **Size:** {format_size(file_path.stat().st_size)}
                        - **Path:** `{path}`

                        Click the download button below to download this result file.
                        """
                        return info, gr.update(visible=True, value=path)

                    results_table.select(
                        fn=on_result_row_select,
                        outputs=[selected_result_path],
                    )

                    selected_result_path.change(
                        fn=update_result_download,
                        inputs=[selected_result_path],
                        outputs=[result_info_box, download_result_btn],
                    )


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

            with gr.Tab("🧬 KO/OE Analysis"):
                gr.Markdown("""
                ### Knockout & Overexpression Analysis
                Analyze gene and reaction knockouts and overexpression:
                - **Knockout Analysis**: Identify essential genes/reactions
                - **Overexpression Analysis**: Simulate gene overexpression
                - **Comprehensive**: All analysis types
                - Compatible with Draft GEM, ecGEM, and etcGEM
                """)
                ko_oe_analysis_tab()

            with gr.Tab("🔬 FBA/FVA Analysis"):
                gr.Markdown("""
                ### Run Flux Balance Analysis
                Perform flux analysis on constructed models:
                - **FBA/pFBA/FVA**: For Draft GEM
                - **ecGEM**: For ecGEM/etcGEM (enzyme-constrained analysis)
                """)
                gem_analysis_tab()

            with gr.Tab("📥 Download Manager"):
                gr.Markdown("""
                ### Download GEM Models & Analysis Results
                Browse and download all available models and analysis results:
                - **GEM Models**: Draft GEM, ecGEM, etcGEM model files
                - **Analysis Results**: FBA/pFBA/FVA, ecGEM, and KO/OE analysis CSV files
                """)
                results_manager_tab()

        gr.Markdown("""
        ---
        **Note:** All tasks run asynchronously in the background. Check the Tasks Monitor tab for overall progress.
        """)

    return interface


# For backward compatibility
def gemfactory_tab_legacy():
    """Legacy single-tab interface"""
    return gemfactory_tab()
