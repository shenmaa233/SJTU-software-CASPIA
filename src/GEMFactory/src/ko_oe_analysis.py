"""
KO/OE Analysis Module
=====================

This module provides knockout (KO) and overexpression (OE) analysis functions
for GEM models (Draft, ecGEM, etcGEM).

Functions:
- Single gene/reaction knockout
- Batch knockout analysis
- Essential gene/reaction identification
- Gene overexpression simulation
"""

import pandas as pd
from cobra import Configuration
from cobra.io import load_json_model, read_sbml_model
from cobra.flux_analysis import single_gene_deletion, single_reaction_deletion
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import os


def load_model(model_path: str):
    """
    Load a GEM model (supports both XML and JSON formats)
    
    Args:
        model_path: Path to model file (.xml or .json)
        
    Returns:
        cobra.Model object
    """
    model_path = Path(model_path)
    
    if not model_path.exists():
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    if model_path.suffix == '.json':
        model = load_json_model(str(model_path))
    elif model_path.suffix in ['.xml', '.sbml']:
        model = read_sbml_model(str(model_path))
    else:
        raise ValueError(f"Unsupported file format: {model_path.suffix}")
    
    return model


def knock_out_single_reaction(model, reaction_id: str, optimize: bool = True) -> Dict:
    """
    Knock out a single reaction
    
    Args:
        model: cobra.Model
        reaction_id: Reaction ID to knock out
        optimize: Whether to optimize after knockout
        
    Returns:
        Dictionary with knockout result
    """
    result = {
        'reaction_id': reaction_id,
        'status': None,
        'objective_value': None,
        'error': None
    }
    
    try:
        with model:
            reaction = model.reactions.get_by_id(reaction_id)
            reaction.knock_out()
            
            if optimize:
                solution = model.optimize()
                result['status'] = solution.status
                result['objective_value'] = solution.objective_value if solution.status == 'optimal' else 0.0
            else:
                result['status'] = 'not_optimized'
                
    except Exception as e:
        result['error'] = str(e)
        result['status'] = 'error'
    
    return result


def knock_out_single_gene(model, gene_id: str, optimize: bool = True) -> Dict:
    """
    Knock out a single gene
    
    Args:
        model: cobra.Model
        gene_id: Gene ID to knock out
        optimize: Whether to optimize after knockout
        
    Returns:
        Dictionary with knockout result
    """
    result = {
        'gene_id': gene_id,
        'status': None,
        'objective_value': None,
        'error': None
    }
    
    try:
        with model:
            gene = model.genes.get_by_id(gene_id)
            gene.knock_out()
            
            if optimize:
                solution = model.optimize()
                result['status'] = solution.status
                result['objective_value'] = solution.objective_value if solution.status == 'optimal' else 0.0
            else:
                result['status'] = 'not_optimized'
                
    except Exception as e:
        result['error'] = str(e)
        result['status'] = 'error'
    
    return result


def batch_knockout_reactions(model, reaction_ids: Optional[List[str]] = None, 
                             result_folder: str = None) -> pd.DataFrame:
    """
    Perform knockout analysis on multiple reactions
    
    Args:
        model: cobra.Model
        reaction_ids: List of reaction IDs (if None, test all reactions)
        result_folder: Folder to save results
        
    Returns:
        DataFrame with knockout results
    """
    if reaction_ids is None:
        reaction_ids = [r.id for r in model.reactions]
    
    results = []
    
    # Get original objective value
    original_solution = model.optimize()
    original_objective = original_solution.objective_value if original_solution.status == 'optimal' else 0.0
    
    for reaction_id in reaction_ids:
        result = knock_out_single_reaction(model, reaction_id, optimize=True)
        result['original_objective'] = original_objective
        
        if result['objective_value'] is not None:
            result['objective_change'] = result['objective_value'] - original_objective
            result['objective_change_percent'] = (result['objective_change'] / original_objective * 100) if original_objective != 0 else 0
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Save results if folder is specified
    if result_folder:
        os.makedirs(result_folder, exist_ok=True)
        output_file = os.path.join(result_folder, 'knockout_reaction_results.csv')
        df.to_csv(output_file, index=False)
    
    return df


def batch_knockout_genes(model, gene_ids: Optional[List[str]] = None,
                         result_folder: str = None) -> pd.DataFrame:
    """
    Perform knockout analysis on multiple genes
    
    Args:
        model: cobra.Model
        gene_ids: List of gene IDs (if None, test all genes)
        result_folder: Folder to save results
        
    Returns:
        DataFrame with knockout results
    """
    if gene_ids is None:
        gene_ids = [g.id for g in model.genes]
    
    results = []
    
    # Get original objective value
    original_solution = model.optimize()
    original_objective = original_solution.objective_value if original_solution.status == 'optimal' else 0.0
    
    for gene_id in gene_ids:
        result = knock_out_single_gene(model, gene_id, optimize=True)
        result['original_objective'] = original_objective
        
        if result['objective_value'] is not None:
            result['objective_change'] = result['objective_value'] - original_objective
            result['objective_change_percent'] = (result['objective_change'] / original_objective * 100) if original_objective != 0 else 0
        
        results.append(result)
    
    df = pd.DataFrame(results)
    
    # Save results if folder is specified
    if result_folder:
        os.makedirs(result_folder, exist_ok=True)
        output_file = os.path.join(result_folder, 'knockout_gene_results.csv')
        df.to_csv(output_file, index=False)
    
    return df


def cobrapy_batch_knockout(model, result_folder: str = None) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Use COBRApy's built-in batch knockout functions
    
    Args:
        model: cobra.Model
        result_folder: Folder to save results
        
    Returns:
        Tuple of (reaction_deletion_results, gene_deletion_results)
    """
    # Single reaction deletion
    reaction_results = single_reaction_deletion(model)
    
    # Single gene deletion
    gene_results = single_gene_deletion(model)
    
    # Save results if folder is specified
    if result_folder:
        os.makedirs(result_folder, exist_ok=True)
        reaction_results.to_csv(os.path.join(result_folder, 'cobrapy_reaction_deletion_results.csv'))
        gene_results.to_csv(os.path.join(result_folder, 'cobrapy_gene_deletion_results.csv'))
    
    return reaction_results, gene_results


def find_essential_reactions(model, knockout_results: Optional[pd.DataFrame] = None,
                            production_threshold: float = 0.01) -> List[str]:
    """
    Find essential reactions based on knockout results
    
    Args:
        model: cobra.Model
        knockout_results: Pre-computed knockout results (if None, will compute)
        production_threshold: Minimum change threshold to consider essential
        
    Returns:
        List of essential reaction IDs
    """
    if knockout_results is None:
        knockout_results = batch_knockout_reactions(model)
    
    essential_reactions = []
    
    for _, row in knockout_results.iterrows():
        if row['status'] != 'optimal':
            # Model cannot grow - essential
            essential_reactions.append(row['reaction_id'])
        elif 'objective_change_percent' in row and abs(row['objective_change_percent']) >= production_threshold * 100:
            # Significant impact on objective - essential
            essential_reactions.append(row['reaction_id'])
    
    return essential_reactions


def find_essential_genes(model, knockout_results: Optional[pd.DataFrame] = None,
                        production_threshold: float = 0.01) -> List[str]:
    """
    Find essential genes based on knockout results
    
    Args:
        model: cobra.Model
        knockout_results: Pre-computed knockout results (if None, will compute)
        production_threshold: Minimum change threshold to consider essential
        
    Returns:
        List of essential gene IDs
    """
    if knockout_results is None:
        knockout_results = batch_knockout_genes(model)
    
    essential_genes = []
    
    for _, row in knockout_results.iterrows():
        if row['status'] != 'optimal':
            # Model cannot grow - essential
            essential_genes.append(row['gene_id'])
        elif 'objective_change_percent' in row and abs(row['objective_change_percent']) >= production_threshold * 100:
            # Significant impact on objective - essential
            essential_genes.append(row['gene_id'])
    
    return essential_genes


def simulate_overexpression(model, reaction_id: str, fold_change: float = 2.0) -> Dict:
    """
    Simulate gene overexpression by increasing reaction bounds
    
    Args:
        model: cobra.Model
        reaction_id: Reaction ID to overexpress
        fold_change: Fold change for upper bound (default: 2x)
        
    Returns:
        Dictionary with overexpression result
    """
    result = {
        'reaction_id': reaction_id,
        'fold_change': fold_change,
        'original_bounds': None,
        'new_bounds': None,
        'original_objective': None,
        'new_objective': None,
        'objective_change': None,
        'status': None,
        'error': None
    }
    
    try:
        # Get original objective value
        original_solution = model.optimize()
        result['original_objective'] = original_solution.objective_value if original_solution.status == 'optimal' else 0.0
        
        with model:
            reaction = model.reactions.get_by_id(reaction_id)
            result['original_bounds'] = (reaction.lower_bound, reaction.upper_bound)
            
            # Increase upper bound by fold_change
            new_upper = reaction.upper_bound * fold_change if reaction.upper_bound > 0 else 1000 * fold_change
            reaction.upper_bound = new_upper
            
            # If reaction is reversible, also increase lower bound
            if reaction.lower_bound < 0:
                new_lower = reaction.lower_bound * fold_change
                reaction.lower_bound = new_lower
            
            result['new_bounds'] = (reaction.lower_bound, reaction.upper_bound)
            
            # Optimize with new bounds
            solution = model.optimize()
            result['status'] = solution.status
            result['new_objective'] = solution.objective_value if solution.status == 'optimal' else 0.0
            result['objective_change'] = result['new_objective'] - result['original_objective']
            
    except Exception as e:
        result['error'] = str(e)
        result['status'] = 'error'
    
    return result


def batch_overexpression_analysis(model, reaction_ids: Optional[List[str]] = None,
                                  fold_changes: List[float] = [2.0, 5.0, 10.0],
                                  result_folder: str = None) -> pd.DataFrame:
    """
    Perform overexpression analysis on multiple reactions with different fold changes
    
    Args:
        model: cobra.Model
        reaction_ids: List of reaction IDs (if None, test all reactions)
        fold_changes: List of fold changes to test
        result_folder: Folder to save results
        
    Returns:
        DataFrame with overexpression results
    """
    if reaction_ids is None:
        reaction_ids = [r.id for r in model.reactions]
    
    results = []
    
    for reaction_id in reaction_ids:
        for fold_change in fold_changes:
            result = simulate_overexpression(model, reaction_id, fold_change)
            results.append(result)
    
    df = pd.DataFrame(results)
    
    # Save results if folder is specified
    if result_folder:
        os.makedirs(result_folder, exist_ok=True)
        output_file = os.path.join(result_folder, 'overexpression_results.csv')
        df.to_csv(output_file, index=False)
    
    return df


def analyze_ko_oe_targets(model, result_folder: str, 
                          production_target: Optional[str] = None,
                          knockout_threshold: float = 0.01,
                          oe_fold_changes: List[float] = [2.0, 5.0, 10.0]) -> Dict[str, pd.DataFrame]:
    """
    Comprehensive KO/OE analysis
    
    Args:
        model: cobra.Model
        result_folder: Folder to save all results
        production_target: Target reaction ID for production optimization
        knockout_threshold: Threshold for essential gene/reaction identification
        oe_fold_changes: Fold changes to test for overexpression
        
    Returns:
        Dictionary containing all analysis results
    """
    os.makedirs(result_folder, exist_ok=True)
    
    results = {}
    
    # Set objective if production target is specified
    if production_target:
        model.objective = production_target
    
    # 1. Batch knockout analysis for reactions
    print("Performing reaction knockout analysis...")
    reaction_ko_results = batch_knockout_reactions(model, result_folder=result_folder)
    results['reaction_knockout'] = reaction_ko_results
    
    # 2. Batch knockout analysis for genes
    print("Performing gene knockout analysis...")
    gene_ko_results = batch_knockout_genes(model, result_folder=result_folder)
    results['gene_knockout'] = gene_ko_results
    
    # 3. Find essential reactions and genes
    print("Identifying essential reactions and genes...")
    essential_reactions = find_essential_reactions(model, reaction_ko_results, knockout_threshold)
    essential_genes = find_essential_genes(model, gene_ko_results, knockout_threshold)
    
    # Save essential lists
    pd.DataFrame({'essential_reactions': essential_reactions}).to_csv(
        os.path.join(result_folder, 'essential_reactions.csv'), index=False
    )
    pd.DataFrame({'essential_genes': essential_genes}).to_csv(
        os.path.join(result_folder, 'essential_genes.csv'), index=False
    )
    
    results['essential_reactions'] = essential_reactions
    results['essential_genes'] = essential_genes
    
    # 4. Overexpression analysis
    print("Performing overexpression analysis...")
    oe_results = batch_overexpression_analysis(model, fold_changes=oe_fold_changes, result_folder=result_folder)
    results['overexpression'] = oe_results
    
    # 5. Use COBRApy's built-in functions for comparison
    print("Running COBRApy batch knockout functions...")
    cobrapy_reaction_results, cobrapy_gene_results = cobrapy_batch_knockout(model, result_folder)
    results['cobrapy_reaction_knockout'] = cobrapy_reaction_results
    results['cobrapy_gene_knockout'] = cobrapy_gene_results
    
    print(f"Analysis complete! Results saved to: {result_folder}")
    
    return results

