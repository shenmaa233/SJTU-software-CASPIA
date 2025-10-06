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
from typing import Dict, Any, List, Optional
from src.GEMFactory.src.utils.GeneMarkS import GeneMarkSRunner
from src.GEMFactory.src.build_GEM import clean_faa, run_carveme
from src.GEMFactory.src.ecGEM.ecgem_service import ECGEMService
from pathlib import Path
import os
import subprocess
from functools import partial
from langchain_core.tools import StructuredTool
from src.utils import get_task_manager

# Get global service instances
_task_manager = get_task_manager()
_ecgem_service = ECGEMService()


# ========================================
# ASYNC TOOLS - Task Submission
# ========================================
# These tools submit tasks to background workers and return immediately.
# Users should check the "Tasks Monitor" tab for progress.

@tool
def submit_gem_analysis(
    model_file_path: str,
    algorithm: str,
    obj: Optional[str] = None,
    substrate: str = "EX_glc__D_e",
    concentration: float = 10.0,
    result_folder: Optional[str] = None
) -> Dict[str, Any]:
    """
    Submit GEM analysis task (FBA/pFBA/FVA for draft GEM, ecGEM for ec/etcGEM).
    This is a long-running task. The tool returns immediately with a task ID.
    
    Args:
        model_file_path: Path to model file (draft GEM XML or ecGEM JSON)
        algorithm: Algorithm to use, options: "FBA", "pFBA", "FVA", "ecGEM"
        obj: Target reaction ID (default: auto-detect biomass reaction)
        substrate: Substrate reaction ID (for ecGEM only, default: "EX_glc__D_e")
        concentration: Substrate concentration in mmol/gDW/h (for ecGEM only, default: 10.0)
        result_folder: Optional custom output folder (default: model_dir/analysis_result)
        
    Returns:
        Dict containing task_id and status message
    """
    if not os.path.exists(model_file_path):
        return {
            "success": False,
            "message": f"Model file not found: {model_file_path}"
        }
    
    # Validate algorithm
    valid_algorithms = ["FBA", "pFBA", "FVA", "ecGEM"]
    if algorithm not in valid_algorithms:
        return {
            "success": False,
            "message": f"Invalid algorithm: {algorithm}. Must be one of {valid_algorithms}"
        }
    
    # Validate file type matches algorithm
    if algorithm in ["FBA", "pFBA", "FVA"] and not model_file_path.endswith(".xml"):
        return {
            "success": False,
            "message": f"Algorithm {algorithm} requires a draft GEM XML file, but got: {model_file_path}"
        }
    
    if algorithm == "ecGEM" and not model_file_path.endswith(".json"):
        return {
            "success": False,
            "message": f"Algorithm ecGEM requires an ecGEM JSON file, but got: {model_file_path}"
        }
    
    try:
        task_id = _ecgem_service.run_gem_analysis(
            model_file=model_file_path,
            algorithm=algorithm,
            obj=obj,
            substrate=substrate,
            concentration=concentration,
            result_folder=result_folder
        )
        
        model_name = Path(model_file_path).stem
        
        result_msg = f"✅ GEM analysis task submitted successfully.\n"
        result_msg += f"Task ID: {task_id}\n"
        result_msg += f"Model: {model_name}\n"
        result_msg += f"Algorithm: {algorithm}\n"
        
        if algorithm == "ecGEM":
            result_msg += f"Parameters: substrate={substrate}, concentration={concentration}\n"
        
        if algorithm == "FVA":
            result_msg += "Estimated time: 5-20 minutes (FVA takes longer than FBA/pFBA)\n"
        else:
            result_msg += "Estimated time: 1-10 minutes\n"
        
        result_msg += "Please check progress in the 'Tasks Monitor' tab or use check_task_status tool."
        
        return {
            "success": True,
            "task_id": task_id,
            "message": result_msg
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to submit GEM analysis task: {str(e)}"
        }


@tool
def submit_ko_oe_analysis(
    model_file_path: str,
    analysis_type: str,
    target_ids: Optional[List[str]] = None,
    production_target: Optional[str] = None,
    knockout_threshold: float = 0.01,
    oe_fold_changes: Optional[List[float]] = None,
    result_folder: Optional[str] = None
) -> Dict[str, Any]:
    """
    Submit KO (Knockout) or OE (Overexpression) analysis task for metabolic engineering.
    Identifies essential genes/reactions and optimal overexpression targets.
    This is a long-running task. The tool returns immediately with a task ID.
    
    Args:
        model_file_path: Path to model file (draft GEM XML or ecGEM JSON)
        analysis_type: Type of analysis, options: "knockout_reaction", "knockout_gene", "overexpression", "comprehensive"
        target_ids: List of specific target IDs to analyze (if None, analyze all)
        production_target: Target reaction ID for production optimization (optional)
        knockout_threshold: Threshold for essential identification (default: 0.01, means <1% of original growth)
        oe_fold_changes: List of fold changes for overexpression (default: [2.0, 5.0, 10.0])
        result_folder: Optional custom output folder (default: model_dir/ko_oe_analysis)
        
    Returns:
        Dict containing task_id and status message
    """
    if not os.path.exists(model_file_path):
        return {
            "success": False,
            "message": f"Model file not found: {model_file_path}"
        }
    
    # Validate analysis type
    valid_types = ["knockout_reaction", "knockout_gene", "overexpression", "comprehensive"]
    if analysis_type not in valid_types:
        return {
            "success": False,
            "message": f"Invalid analysis type: {analysis_type}. Must be one of {valid_types}"
        }
    
    # Validate knockout threshold
    if knockout_threshold < 0 or knockout_threshold > 1:
        return {
            "success": False,
            "message": f"Invalid knockout threshold: {knockout_threshold}. Must be between 0 and 1"
        }
    
    try:
        task_id = _ecgem_service.run_ko_oe_analysis(
            model_file=model_file_path,
            analysis_type=analysis_type,
            target_ids=target_ids,
            production_target=production_target,
            knockout_threshold=knockout_threshold,
            oe_fold_changes=oe_fold_changes,
            result_folder=result_folder
        )
        
        model_name = Path(model_file_path).stem
        
        result_msg = f"✅ KO/OE analysis task submitted successfully.\n"
        result_msg += f"Task ID: {task_id}\n"
        result_msg += f"Model: {model_name}\n"
        result_msg += f"Analysis Type: {analysis_type}\n"
        
        if production_target:
            result_msg += f"Production Target: {production_target}\n"
        
        if target_ids:
            result_msg += f"Target Count: {len(target_ids)} specific targets\n"
        else:
            result_msg += "Target Count: All targets (genome-wide analysis)\n"
        
        if analysis_type == "overexpression":
            fold_changes = oe_fold_changes if oe_fold_changes else [2.0, 5.0, 10.0]
            result_msg += f"Fold Changes: {fold_changes}\n"
        
        if analysis_type == "comprehensive":
            result_msg += "Estimated time: 10-60 minutes (comprehensive analysis)\n"
        elif analysis_type == "overexpression":
            result_msg += "Estimated time: 5-30 minutes (depending on targets)\n"
        else:
            result_msg += "Estimated time: 5-20 minutes (depending on targets)\n"
        
        result_msg += "Please check progress in the 'Tasks Monitor' tab or use check_task_status tool."
        
        return {
            "success": True,
            "task_id": task_id,
            "message": result_msg
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to submit KO/OE analysis task: {str(e)}"
        }

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
    runner = GeneMarkSRunner(gms_script_path=os.environ.get("GMS_SCRIPT_PATH"))
    
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
    gms_runner = GeneMarkSRunner(gms_script_path=os.environ.get("GMS_SCRIPT_PATH"))
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
def submit_ecgem_build(
    model_file_path: str,
    f: float = 0.405,
    ptot: float = 0.56,
    sigma: float = 1.0,
    lowerbound: float = 0.0
) -> Dict[str, Any]:
    """
    Submit an enzyme-constrained GEM (ecGEM) construction task.
    Adds enzyme kinetic constraints to improve flux predictions.
    This is a long-running task (30-120 minutes).
    
    Args:
        model_file_path: Path to draft GEM model file (.xml)
        f: Fraction of enzymes with available kcat values (0.1-1.0, default: 0.405)
        ptot: Total protein fraction of cell dry weight in g/gDW (0.1-1.0, default: 0.56)
        sigma: Average enzyme saturation factor (0.1-2.0, default: 1.0)
        lowerbound: Lower bound for enzyme constraints (0.0-0.1, default: 0.0)
        
    Returns:
        Dict containing task_id and status message
    """
    if not os.path.exists(model_file_path):
        return {
            "success": False,
            "message": f"Model file not found: {model_file_path}"
        }
    
    # Check model suitability first
    is_suitable, messages = _ecgem_service.check_model_suitability(model_file_path)
    if not is_suitable:
        return {
            "success": False,
            "message": f"Model is not suitable for ecGEM construction:\n" + "\n".join(messages)
        }
    
    try:
        task_id = _ecgem_service.build_ecgem(
            model_file=model_file_path,
            f=f,
            ptot=ptot,
            sigma=sigma,
            lowerbound=lowerbound
        )
        
        model_name = Path(model_file_path).stem.replace("_draft", "")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": f"✅ ecGEM building task submitted successfully.\n"
                      f"Task ID: {task_id}\n"
                      f"Model: {model_name}\n"
                      f"Parameters: f={f}, ptot={ptot}, sigma={sigma}\n"
                      f"Estimated time: 30-120 minutes\n"
                      f"Please check progress in the 'Tasks Monitor' tab or use check_task_status tool."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to submit ecGEM task: {str(e)}"
        }


@tool
def submit_etcgem_build(
    model_file_path: str,
    temperature: float,
    f: float = 0.405,
    ptot: float = 0.56,
    sigma: float = 1.0,
    lowerbound: float = 0.0
) -> Dict[str, Any]:
    """
    Submit an enzyme-temperature-constrained GEM (etcGEM) construction task.
    Adds both enzyme kinetics AND temperature-dependent constraints.
    This is a long-running task (40-150 minutes).
    
    Args:
        model_file_path: Path to draft GEM model file (.xml)
        temperature: Optimal growth temperature in Celsius (0-100, e.g., 37 for E. coli)
        f: Fraction of enzymes with available kcat values (0.1-1.0, default: 0.405)
        ptot: Total protein fraction of cell dry weight in g/gDW (0.1-1.0, default: 0.56)
        sigma: Average enzyme saturation factor (0.1-2.0, default: 1.0)
        lowerbound: Lower bound for enzyme constraints (0.0-0.1, default: 0.0)
        
    Returns:
        Dict containing task_id and status message
    """
    if not os.path.exists(model_file_path):
        return {
            "success": False,
            "message": f"Model file not found: {model_file_path}"
        }
    
    if temperature < 0 or temperature > 100:
        return {
            "success": False,
            "message": f"Invalid temperature: {temperature}. Must be between 0-100°C"
        }
    
    # Check model suitability first
    is_suitable, messages = _ecgem_service.check_model_suitability(model_file_path)
    if not is_suitable:
        return {
            "success": False,
            "message": f"Model is not suitable for etcGEM construction:\n" + "\n".join(messages)
        }
    
    try:
        task_id = _ecgem_service.build_etcgem(
            model_file=model_file_path,
            temperature=temperature,
            f=f,
            ptot=ptot,
            sigma=sigma,
            lowerbound=lowerbound
        )
        
        model_name = Path(model_file_path).stem.replace("_draft", "")
        
        return {
            "success": True,
            "task_id": task_id,
            "message": f"✅ etcGEM building task submitted successfully.\n"
                      f"Task ID: {task_id}\n"
                      f"Model: {model_name}\n"
                      f"Temperature: {temperature}°C\n"
                      f"Parameters: f={f}, ptot={ptot}, sigma={sigma}\n"
                      f"Estimated time: 40-150 minutes\n"
                      f"Please check progress in the 'Tasks Monitor' tab or use check_task_status tool."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Failed to submit etcGEM task: {str(e)}"
        }


@tool
def check_model_suitability(model_file_path: str) -> Dict[str, Any]:
    """
    Check if a draft GEM model is suitable for enzyme-constrained model construction.
    Validates metabolite coverage and reaction annotation quality.
    
    Args:
        model_file_path: Path to draft GEM model file (.xml)
        
    Returns:
        Dict containing suitability status and detailed messages
    """
    if not os.path.exists(model_file_path):
        return {
            "success": False,
            "is_suitable": False,
            "message": f"Model file not found: {model_file_path}"
        }
    
    try:
        is_suitable, messages = _ecgem_service.check_model_suitability(model_file_path)
        
        return {
            "success": True,
            "is_suitable": is_suitable,
            "model_file": model_file_path,
            "details": messages,
            "message": "✅ Model is suitable for ecGEM/etcGEM construction" if is_suitable 
                      else "❌ Model is NOT suitable for ecGEM/etcGEM construction",
            "recommendation": "You can proceed with ecGEM or etcGEM construction." if is_suitable
                            else "Please use a model with better metabolite and reaction coverage (>25%)."
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error checking suitability: {str(e)}"
        }


@tool
def list_draft_gem_models() -> Dict[str, Any]:
    """
    List all available draft GEM models that can be used for ecGEM/etcGEM construction.
    
    Returns:
        Dict containing list of draft GEM models with metadata
    """
    try:
        models = _ecgem_service.list_draft_models()
        
        if not models:
            return {
                "success": True,
                "count": 0,
                "models": [],
                "message": "No draft GEM models found. Please build a draft GEM first using submit_gem_build tool."
            }
        
        model_list = []
        for model in models:
            model_list.append({
                "name": model["name"],
                "path": model["path"],
                "size_mb": round(model["size"] / 1024 / 1024, 2),
                "modified": model["modified"]
            })
        
        return {
            "success": True,
            "count": len(models),
            "models": model_list,
            "message": f"Found {len(models)} draft GEM model(s)"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error listing models: {str(e)}"
        }


@tool
def list_ecgem_models() -> Dict[str, Any]:
    """
    List all built enzyme-constrained GEM (ecGEM) models.
    
    Returns:
        Dict containing list of ecGEM models with metadata
    """
    try:
        models = _ecgem_service.list_ecgem_models()
        
        if not models:
            return {
                "success": True,
                "count": 0,
                "models": [],
                "message": "No ecGEM models found. Build one using submit_ecgem_build tool."
            }
        
        model_list = []
        for model in models:
            model_list.append({
                "name": model["name"],
                "path": model["path"],
                "folder": model["folder"],
                "size_mb": round(model["size"] / 1024 / 1024, 2),
                "modified": model["modified"]
            })
        
        return {
            "success": True,
            "count": len(models),
            "models": model_list,
            "message": f"Found {len(models)} ecGEM model(s)"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error listing ecGEM models: {str(e)}"
        }


@tool
def list_etcgem_models() -> Dict[str, Any]:
    """
    List all built enzyme-temperature-constrained GEM (etcGEM) models.
    
    Returns:
        Dict containing list of etcGEM models with metadata
    """
    try:
        models = _ecgem_service.list_etcgem_models()
        
        if not models:
            return {
                "success": True,
                "count": 0,
                "models": [],
                "message": "No etcGEM models found. Build one using submit_etcgem_build tool."
            }
        
        model_list = []
        for model in models:
            model_list.append({
                "name": model["name"],
                "path": model["path"],
                "folder": model["folder"],
                "temperature": model.get("temperature"),
                "size_mb": round(model["size"] / 1024 / 1024, 2),
                "modified": model["modified"]
            })
        
        return {
            "success": True,
            "count": len(models),
            "models": model_list,
            "message": f"Found {len(models)} etcGEM model(s)"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error listing etcGEM models: {str(e)}"
        }


@tool
def get_model_statistics(model_folder_path: str) -> Dict[str, Any]:
    """
    Get detailed statistics for a built ecGEM or etcGEM model.
    Shows information about intermediate files and model construction results.
    
    Args:
        model_folder_path: Path to the model result folder
        
    Returns:
        Dict containing model statistics and file information
    """
    if not os.path.exists(model_folder_path):
        return {
            "success": False,
            "message": f"Model folder not found: {model_folder_path}"
        }
    
    try:
        stats = _ecgem_service.get_model_stats(model_folder_path)
        
        if not stats:
            return {
                "success": False,
                "message": f"Unable to retrieve statistics for: {model_folder_path}"
            }
        
        # Format file information
        files_info = []
        for filename, info in stats.get("files", {}).items():
            if info.get("exists"):
                file_info = {
                    "filename": filename,
                    "size_kb": round(info["size"] / 1024, 2),
                    "path": info["path"]
                }
                if "rows" in info:
                    file_info["rows"] = info["rows"]
                if "columns" in info:
                    file_info["columns"] = info["columns"]
                files_info.append(file_info)
        
        return {
            "success": True,
            "folder": stats["folder"],
            "files": files_info,
            "message": f"Retrieved statistics for {len(files_info)} file(s) in model folder"
        }
    except Exception as e:
        return {
            "success": False,
            "message": f"Error getting model statistics: {str(e)}"
        }


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
