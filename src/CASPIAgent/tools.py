# src/CASPIAgent/tools.py

"""
CASPIAgent Tools
================

This module contains all tools available to the CASPIAgent.

Tools are organized into two categories:
1. Async Tools: Submit long-running tasks, return immediately with task ID
2. Sync Tools: Execute quickly and return results directly in the conversation

"""

from langchain.tools import tool
from typing import Dict, Any
from src.GEMFactory.src.utils.GeneMarkS import GeneMarkSRunner
from src.GEMFactory.src.build_GEM import clean_faa, run_carveme
from pathlib import Path
import os
import subprocess
from functools import partial
from langchain_core.tools import StructuredTool
from src.utils import get_task_manager

# Get global task manager instance
_task_manager = get_task_manager()


# ========================================
# ASYNC TOOLS - Task Submission
# ========================================
# These tools submit tasks to background workers and return immediately.
# Users should check the "Tasks Monitor" tab for progress.


def _gene_annotation_task(logger, genome_path: str) -> str:
    """
    Internal task implementation: Run GeneMarkS gene annotation.
    
    Args:
        logger: Logger instance for task output
        genome_path: Path to genome FASTA file
        
    Returns:
        Path to protein FAA file
    """
    logger.info(f"Starting GeneMarkS annotation for: {genome_path}")
    
    output_dir = os.path.join(os.path.dirname(genome_path), "genemarks_output")
    runner = GeneMarkSRunner(gms_script_path="/home/shenmaa/gms2_linux_64/gms2.pl")
    
    results = runner.run(
        input_fasta=genome_path,
        output_dir=output_dir
    )
    
    protein_faa = results.get("faa", "")
    logger.info(f"✅ Protein sequences saved to: {protein_faa}")
    
    return protein_faa


def _gem_build_task(logger, genome_path: str, gapfill: str = "None") -> str:
    """
    Internal task implementation: Full GEM building pipeline (GeneMarkS + CarveMe).
    
    Args:
        logger: Logger instance for task output
        genome_path: Path to genome FASTA file
        gapfill: Gap-filling medium (e.g., "M9", "LB")
        
    Returns:
        Path to generated GEM XML file
    """
    logger.info(f"🚀 Starting GEM build pipeline for: {genome_path}")
    logger.info(f"Gap-fill medium: {gapfill}")
    
    # Step 1: GeneMarkS
    logger.info("🔬 Step 1/3: Running GeneMarkS annotation...")
    gms_runner = GeneMarkSRunner(gms_script_path="/home/shenmaa/gms2_linux_64/gms2.pl")
    gms_outputs = gms_runner.run(
        input_fasta=genome_path,
        output_dir="src/GEMFactory/data/GeneMarkS",
        genome_type="bacteria",
        gcode="11",
    )
    logger.info("✅ GeneMarkS annotation completed")
    
    # Step 2: Clean FASTA
    logger.info("🧹 Step 2/3: Cleaning protein FASTA headers...")
    clean_faa_path = clean_faa(gms_outputs["faa"])
    logger.info(f"✅ Clean FASTA saved: {clean_faa_path}")
    
    # Step 3: CarveMe
    prefix = Path(genome_path).stem
    gem_output = f"src/GEMFactory/data/CarveMe/{prefix}_draft.xml"
    logger.info("🛠️ Step 3/3: Running CarveMe reconstruction...")
    
    gapfill_arg = None if gapfill == "None" else gapfill
    run_carveme(clean_faa_path, gem_output, gapfill=gapfill_arg, tmpdir="src/GEMFactory/data/temp")
    
    logger.info(f"✅ GEM built successfully: {gem_output}")
    return gem_output


@tool
def submit_gene_annotation(genome_file_path: str) -> Dict[str, str]:
    """
    Submit a gene annotation task using GeneMarkS.
    This tool submits the task and returns immediately with a task ID.
    Use 'check_task_status' to monitor progress.
    
    Args:
        genome_file_path: Path to the genome FASTA file
        
    Returns:
        Dict containing task_id and status message
    """
    if not os.path.exists(genome_file_path):
        return {
            "success": False,
            "message": f"Genome file not found: {genome_file_path}"
        }
    
    task_id = _task_manager.start(
        _gene_annotation_task,
        genome_file_path,
        prefix="gene-",
        task_name=f"Gene Annotation: {Path(genome_file_path).name}",
        task_type="gene_annotation"
    )
    
    return {
        "success": True,
        "task_id": task_id,
        "message": f"✅ Gene annotation task submitted successfully.\n"
                  f"Task ID: {task_id}\n"
                  f"Please check progress in the 'Tasks Monitor' tab or use check_task_status tool."
    }


@tool
def submit_gem_build(genome_file_path: str, gapfill_medium: str = "None") -> Dict[str, str]:
    """
    Submit a complete GEM building pipeline (GeneMarkS + CarveMe).
    This is a long-running task. The tool returns immediately with a task ID.
    
    Args:
        genome_file_path: Path to the genome FASTA file
        gapfill_medium: Gap-filling medium, options: "None", "M9", "LB", "M9,LB"
        
    Returns:
        Dict containing task_id and status message
    """
    if not os.path.exists(genome_file_path):
        return {
            "success": False,
            "message": f"Genome file not found: {genome_file_path}"
        }
    
    task_id = _task_manager.start(
        _gem_build_task,
        genome_file_path,
        gapfill_medium,
        prefix="gem-",
        task_name=f"GEM Build: {Path(genome_file_path).name}",
        task_type="gem_build"
    )
    
    return {
        "success": True,
        "task_id": task_id,
        "message": f"✅ GEM building task submitted successfully.\n"
                  f"Task ID: {task_id}\n"
                  f"This may take 10-30 minutes depending on genome size.\n"
                  f"Please check progress in the 'Tasks Monitor' tab or use check_task_status tool."
    }


@tool
def check_task_status(task_id: str) -> Dict[str, Any]:
    """
    Check the status of a submitted task.
    
    Args:
        task_id: The task ID returned when submitting a task
        
    Returns:
        Dict containing task status and details
    """
    task_info = _task_manager.get_task_info(task_id)
    
    if not task_info:
        return {
            "success": False,
            "message": f"Task ID '{task_id}' not found"
        }
    
    status_emoji = "🚧" if not task_info["done"] else "✅" if task_info["success"] else "❌"
    status_text = "Running" if not task_info["done"] else "Completed" if task_info["success"] else "Failed"
    
    return {
        "success": True,
        "task_id": task_id,
        "task_name": task_info["name"],
        "status": f"{status_emoji} {status_text}",
        "result": task_info.get("result", ""),
        "start_time": str(task_info.get("start_time", "N/A")),
        "message": f"Task '{task_info['name']}' is currently {status_text.lower()}.\n"
                  f"For detailed logs, please visit the 'Tasks Monitor' tab."
    }


# ========================================
# SYNC TOOLS - Direct Execution
# ========================================
# These tools execute quickly and return results immediately.


@tool
def predict_kcat(smiles: str, protein_sequence: str, log_transform: bool = True) -> Dict[str, Any]:
    """
    Predict enzyme catalytic constant (kcat) for a given substrate-enzyme pair.
    This is a fast prediction tool that returns results directly.
    
    Args:
        smiles: SMILES string of the substrate molecule
        protein_sequence: Amino acid sequence of the enzyme
        log_transform: Whether to apply log transformation (default: True)
        
    Returns:
        Dict containing prediction results or error information
    """
    # Get project root directory (2 levels up from this file)
    current_file = os.path.abspath(__file__)
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_file)))
    
    # Build correct paths relative to project root
    caspred_dir = os.path.join(project_root, 'src', 'CASPred')
    model_path = os.path.join(caspred_dir, 'model', 'kcat_models', 'model_1.pth')
    config_path = os.path.join(caspred_dir, 'config.json')
    predict_script = os.path.join(caspred_dir, 'src', 'predict.py')
    
    # Validate inputs
    if not smiles or not protein_sequence:
        return {
            'success': False, 
            'error': 'Both smiles and protein_sequence are required parameters'
        }
    
    # Check if required files exist
    missing_files = []
    for path, name in [(model_path, 'Model file'), (config_path, 'Config file'), (predict_script, 'Prediction script')]:
        if not os.path.exists(path):
            missing_files.append(f"{name}: {path}")
    
    if missing_files:
        return {
            'success': False, 
            'error': f'Missing required files:\n' + '\n'.join(missing_files)
        }
    
    # Build command - use module execution to support relative imports
    cmd = [
        'python', '-m', 'src.CASPred.src.predict',
        '--model', model_path, 
        '--config', config_path,
        '--smiles', smiles, 
        '--sequence', protein_sequence
    ]
    if log_transform:
        cmd.append('--log_transform')
    
    # Execute prediction - must run from project root for module import
    original_cwd = os.getcwd()
    
    try:
        # Change to project root to enable module imports
        os.chdir(project_root)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        if result.returncode != 0:
            return {
                'success': False, 
                'error': f'Prediction script failed (return code: {result.returncode})',
                'raw_output': result.stdout.strip(), 
                'raw_error': result.stderr.strip()
            }
        
        # Parse prediction result
        predicted_kcat = None
        for line in result.stdout.strip().split('\n'):
            if 'Predicted kcat value:' in line:
                try:
                    predicted_kcat = float(line.split(':')[1].strip().split()[0])
                    break
                except:
                    continue
        
        if predicted_kcat is None:
            return {
                'success': False, 
                'error': 'Failed to parse prediction result from output',
                'raw_output': result.stdout.strip(), 
                'raw_error': result.stderr.strip()
            }
        
        return {
            'success': True,
            'predicted_kcat': predicted_kcat,
            'unit': 's^-1',
            'description': f'Predicted kcat value is {predicted_kcat:.4f} s^-1',
            'raw_output': result.stdout.strip()
        }
    
    except subprocess.TimeoutExpired:
        return {
            'success': False, 
            'error': 'Prediction timeout (exceeded 5 minutes)'
        }
    except Exception as e:
        return {
            'success': False, 
            'error': f'Error during prediction: {str(e)}'
        }
    finally:
        os.chdir(original_cwd)


@tool
def multiply(x: int, y: int) -> int:
    """
    Multiply two numbers (demo tool for testing).
    
    Args:
        x: First number
        y: Second number
        
    Returns:
        Product of x and y
    """
    return x * y


# ========================================
# LEGACY SUPPORT
# ========================================
# Support for backward compatibility with existing code


def _run_gene_prediction_implementation(genome_file) -> Dict[str, str]:
    """
    Legacy wrapper: Submit gene annotation task from Gradio file object.
    This maintains backward compatibility with existing code.
    
    Args:
        genome_file: Gradio file object
        
    Returns:
        Dict with message and task_id
    """
    if genome_file is None:
        return {"message": "Error: No genome file provided.", "protein_faa_path": ""}
    
    genome_path = genome_file.name
    result = submit_gene_annotation.invoke({"genome_file_path": genome_path})
    
    if result["success"]:
        return {
            "message": result["message"],
            "task_id": result["task_id"]
        }
    else:
        return {
            "message": result["message"],
            "protein_faa_path": ""
        }


def make_file_prediction_tool(uploaded_file):
    """
    Create a tool for gene prediction from uploaded Gradio file.
    This is for backward compatibility with the file upload workflow.
    
    Args:
        uploaded_file: Gradio file object
        
    Returns:
        StructuredTool instance
    """
    run_prediction_partial = partial(_run_gene_prediction_implementation, genome_file=uploaded_file)

    def wrapper():
        """Submit gene annotation task for the uploaded genome file. No parameters needed."""
        return run_prediction_partial()

    return StructuredTool.from_function(func=wrapper, name="run_gene_prediction_real")
