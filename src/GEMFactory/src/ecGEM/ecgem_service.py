"""
ecGEM Service
=============

Service layer for ecGEM and etcGEM construction.
Provides clean API for Web UI integration.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple, List
from src.utils import get_task_manager
from .utils import *


class ECGEMService:
    """Service for building enzyme-constrained GEMs"""
    
    def __init__(self):
        self.task_manager = get_task_manager()
        self.bigg_met_file = "src/GEMFactory/src/ecGEM/bigg_models_metabolites.txt"
    
    def check_model_suitability(self, model_file: str) -> Tuple[bool, list]:
        """
        Check if a model is suitable for ecGEM construction.
        
        Args:
            model_file: Path to model file
            
        Returns:
            Tuple of (is_suitable, messages)
        """
        try:
            suitability, messages = Determine_suitable_ecGEM(model_file, self.bigg_met_file)
            return (suitability == "Yes", messages)
        except Exception as e:
            return (False, [f"Error checking suitability: {str(e)}"])
    
    def list_draft_models(self) -> list:
        """List all available draft GEM models"""
        carveme_dir = Path("src/GEMFactory/data/CarveMe")
        if not carveme_dir.exists():
            return []
        
        models = []
        for file in carveme_dir.glob("*_draft.xml"):
            models.append({
                "name": file.stem.replace("_draft", ""),
                "path": str(file),
                "size": file.stat().st_size,
                "modified": file.stat().st_mtime
            })
        return sorted(models, key=lambda x: x["modified"], reverse=True)
    
    def list_ecgem_models(self) -> list:
        """List all built ecGEM models"""
        ecgem_dir = Path("src/GEMFactory/data/ecGEM")
        if not ecgem_dir.exists():
            return []
        
        models = []
        for folder in ecgem_dir.iterdir():
            if folder.is_dir():
                model_file = folder / "ecModel.json"
                if model_file.exists():
                    models.append({
                        "name": folder.name,
                        "path": str(model_file),
                        "folder": str(folder),
                        "size": model_file.stat().st_size,
                        "modified": model_file.stat().st_mtime
                    })
        return sorted(models, key=lambda x: x["modified"], reverse=True)
    
    def list_etcgem_models(self) -> list:
        """List all built etcGEM models"""
        etcgem_dir = Path("src/GEMFactory/data/etcGEM")
        if not etcgem_dir.exists():
            return []
        
        models = []
        for folder in etcgem_dir.iterdir():
            if folder.is_dir():
                model_file = folder / "ecModel.json"
                if model_file.exists():
                    models.append({
                        "name": folder.name,
                        "path": str(model_file),
                        "folder": str(folder),
                        "size": model_file.stat().st_size,
                        "modified": model_file.stat().st_mtime,
                        "temperature": self._extract_temperature(folder.name)
                    })
        return sorted(models, key=lambda x: x["modified"], reverse=True)
    
    def _extract_temperature(self, folder_name: str) -> Optional[float]:
        """Extract temperature from folder name like 'model_T=37.0'"""
        import re
        match = re.search(r'T=([0-9.]+)', folder_name)
        return float(match.group(1)) if match else None
    
    def build_ecgem(
        self,
        model_file: str,
        f: float = 0.405,
        ptot: float = 0.56,
        sigma: float = 1.0,
        lowerbound: float = 0.0,
        result_folder: Optional[str] = None
    ) -> str:
        """
        Submit ecGEM building task.
        
        Args:
            model_file: Path to draft GEM model
            f: Fraction of enzymes with available kcat values
            ptot: Total protein fraction (g/gDW)
            sigma: Average enzyme saturation factor
            lowerbound: Lower bound for enzyme constraints
            result_folder: Optional custom output folder
            
        Returns:
            Task ID
        """
        model_name = os.path.basename(model_file).replace("_draft.xml", "")
        
        if result_folder is None:
            result_folder = f"src/GEMFactory/data/ecGEM/{model_name}"
        
        protein_file = f"src/GEMFactory/data/GeneMarkS/{model_name}/{model_name}_protein_clean.fasta"
        
        task_id = self.task_manager.start(
            self._ecgem_task,
            model_file,
            protein_file,
            result_folder,
            f,
            ptot,
            sigma,
            lowerbound,
            prefix="ecgem-",
            task_name=f"ecGEM Build: {model_name}",
            task_type="ecgem_build"
        )
        
        return task_id
    
    def build_etcgem(
        self,
        model_file: str,
        temperature: float,
        f: float = 0.405,
        ptot: float = 0.56,
        sigma: float = 1.0,
        lowerbound: float = 0.0,
        result_folder: Optional[str] = None
    ) -> str:
        """
        Submit etcGEM building task.
        
        Args:
            model_file: Path to draft GEM model
            temperature: Optimal temperature (°C)
            f: Fraction of enzymes with available kcat values
            ptot: Total protein fraction (g/gDW)
            sigma: Average enzyme saturation factor
            lowerbound: Lower bound for enzyme constraints
            result_folder: Optional custom output folder
            
        Returns:
            Task ID
        """
        model_name = os.path.basename(model_file).replace("_draft.xml", "")
        
        if result_folder is None:
            result_folder = f"src/GEMFactory/data/etcGEM/{model_name}_T={temperature}"
        
        protein_file = f"src/GEMFactory/data/GeneMarkS/{model_name}/{model_name}_protein_clean.fasta"
        
        task_id = self.task_manager.start(
            self._etcgem_task,
            model_file,
            protein_file,
            result_folder,
            temperature,
            f,
            ptot,
            sigma,
            lowerbound,
            prefix="etcgem-",
            task_name=f"etcGEM Build: {model_name} @ {temperature}°C",
            task_type="etcgem_build"
        )
        
        return task_id
    
    def _ecgem_task(
        self,
        logger,
        model_file: str,
        protein_file: str,
        result_folder: str,
        f: float,
        ptot: float,
        sigma: float,
        lowerbound: float
    ) -> str:
        """Background task for ecGEM construction"""
        os.makedirs(result_folder, exist_ok=True)
        
        logger.info("="*60)
        logger.info("🏗️  Starting ecGEM Construction Pipeline")
        logger.info("="*60)
        logger.info(f"Model: {model_file}")
        logger.info(f"Output: {result_folder}")
        logger.info(f"Parameters: f={f}, ptot={ptot}, sigma={sigma}")
        
        # Step 1: Check suitability
        logger.info("\n📋 Step 1: Checking model suitability...")
        suitability, messages = Determine_suitable_ecGEM(model_file, self.bigg_met_file)
        for msg in messages:
            logger.info(f"  {msg}")
        
        if suitability == "No":
            raise ValueError("Model is not suitable for ecGEM construction")
        
        # Step 2: Split and pair substrates (with checkpoint recovery)
        logger.info("\n🔬 Step 2: Processing metabolite-reaction-protein pairs...")
        metabolites_gpr_file = f"{result_folder}/metabolites_reactions_gpr.csv"
        
        if os.path.exists(metabolites_gpr_file):
            logger.info("  ♻️  Found existing metabolites_reactions_gpr.csv, loading from checkpoint...")
            gprdf = pd.read_csv(metabolites_gpr_file)
            logger.info(f"  Loaded {len(gprdf)} metabolite-reaction pairs from checkpoint")
        else:
            from .build_ecGEM import split_and_pair_substrate_with_protein
            gprdf = split_and_pair_substrate_with_protein(model_file, result_folder)
            logger.info(f"  Generated {len(gprdf)} metabolite-reaction pairs")
        
        # Step 3: Predict kcat (parameter_predict has internal checkpoint recovery)
        logger.info("\n🧬 Step 3: Predicting kcat parameters...")
        full_metabolites_file = f"{result_folder}/full_metabolites_reactions.csv"
        
        if os.path.exists(full_metabolites_file):
            logger.info("  ♻️  Found existing full_metabolites_reactions.csv, loading from checkpoint...")
            gprdf_with_kcat = pd.read_csv(full_metabolites_file)
            logger.info(f"  Loaded {len(gprdf_with_kcat)} entries from checkpoint")
        else:
            gprdf_with_kcat = parameter_predict(
                gprdf, protein_file, model_file, result_folder,
                is_etc=False, T=None
            )
            logger.info(f"  Predicted kcat for {len(gprdf_with_kcat)} entries")
        
        # Step 4: Get kcat/MW (with checkpoint recovery)
        logger.info("\n⚖️  Step 4: Computing kcat/MW ratios...")
        reaction_kcat_mw_file = f"{result_folder}/reaction_kcat_mw.csv"
        
        if os.path.exists(reaction_kcat_mw_file):
            logger.info("  ♻️  Found existing reaction_kcat_mw.csv, loading from checkpoint...")
            reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file)
            logger.info(f"  Loaded {len(reaction_kcat_mw)} kcat/MW values from checkpoint")
        else:
            reaction_kcat_mw = get_kcat_mw(gprdf_with_kcat, result_folder)
            logger.info(f"  Computed {len(reaction_kcat_mw)} kcat/MW values")
        
        # Step 5: Build ecGEM (with checkpoint recovery)
        logger.info("\n🏗️  Step 5: Building enzyme-constrained model...")
        ecModel_file = f"{result_folder}/ecModel.json"
        
        if os.path.exists(ecModel_file):
            logger.info("  ♻️  Found existing ecModel.json, skipping model construction...")
            logger.info(f"  ✅ ecGEM already exists: {ecModel_file}")
        else:
            from .build_ecGEM import build_ecGEM
            ecModel_file = build_ecGEM(
                model_file, result_folder,
                f=f, ptot=ptot, sigma=sigma, lowerbound=lowerbound
            )
            logger.info(f"  ✅ ecGEM saved: {ecModel_file}")
        
        logger.info("\n"+"="*60)
        logger.info("✅ ecGEM Construction Complete!")
        logger.info("="*60)
        
        return ecModel_file
    
    def _etcgem_task(
        self,
        logger,
        model_file: str,
        protein_file: str,
        result_folder: str,
        temperature: float,
        f: float,
        ptot: float,
        sigma: float,
        lowerbound: float
    ) -> str:
        """Background task for etcGEM construction"""
        os.makedirs(result_folder, exist_ok=True)
        
        logger.info("="*60)
        logger.info("🌡️  Starting etcGEM Construction Pipeline")
        logger.info("="*60)
        logger.info(f"Model: {model_file}")
        logger.info(f"Temperature: {temperature}°C")
        logger.info(f"Output: {result_folder}")
        logger.info(f"Parameters: f={f}, ptot={ptot}, sigma={sigma}")
        
        # Step 1: Check suitability
        logger.info("\n📋 Step 1: Checking model suitability...")
        suitability, messages = Determine_suitable_ecGEM(model_file, self.bigg_met_file)
        for msg in messages:
            logger.info(f"  {msg}")
        
        if suitability == "No":
            raise ValueError("Model is not suitable for etcGEM construction")
        
        # Step 2: Split and pair substrates (with checkpoint recovery)
        logger.info("\n🔬 Step 2: Processing metabolite-reaction-protein pairs...")
        metabolites_gpr_file = f"{result_folder}/metabolites_reactions_gpr.csv"
        
        if os.path.exists(metabolites_gpr_file):
            logger.info("  ♻️  Found existing metabolites_reactions_gpr.csv, loading from checkpoint...")
            gprdf = pd.read_csv(metabolites_gpr_file)
            logger.info(f"  Loaded {len(gprdf)} metabolite-reaction pairs from checkpoint")
        else:
            from .build_ecGEM import split_and_pair_substrate_with_protein
            gprdf = split_and_pair_substrate_with_protein(model_file, result_folder)
            logger.info(f"  Generated {len(gprdf)} metabolite-reaction pairs")
        
        # Step 3: Predict kcat and Topt (parameter_predict has internal checkpoint recovery)
        logger.info(f"\n🧬 Step 3: Predicting kcat and Topt @ {temperature}°C...")
        full_metabolites_file = f"{result_folder}/full_metabolites_reactions.csv"
        
        if os.path.exists(full_metabolites_file):
            logger.info("  ♻️  Found existing full_metabolites_reactions.csv, loading from checkpoint...")
            gprdf_with_kcat = pd.read_csv(full_metabolites_file)
            logger.info(f"  Loaded {len(gprdf_with_kcat)} entries from checkpoint")
        else:
            gprdf_with_kcat = parameter_predict(
                gprdf, protein_file, model_file, result_folder,
                is_etc=True, T=temperature
            )
            logger.info(f"  Predicted parameters for {len(gprdf_with_kcat)} entries")
        
        # Step 4: Get kcat/MW (with checkpoint recovery)
        logger.info("\n⚖️  Step 4: Computing kcat/MW ratios...")
        reaction_kcat_mw_file = f"{result_folder}/reaction_kcat_mw.csv"
        
        if os.path.exists(reaction_kcat_mw_file):
            logger.info("  ♻️  Found existing reaction_kcat_mw.csv, loading from checkpoint...")
            reaction_kcat_mw = pd.read_csv(reaction_kcat_mw_file)
            logger.info(f"  Loaded {len(reaction_kcat_mw)} kcat/MW values from checkpoint")
        else:
            reaction_kcat_mw = get_kcat_mw(gprdf_with_kcat, result_folder)
            logger.info(f"  Computed {len(reaction_kcat_mw)} kcat/MW values")
        
        # Step 5: Build etcGEM (with checkpoint recovery)
        logger.info("\n🏗️  Step 5: Building enzyme-temperature-constrained model...")
        ecModel_file = f"{result_folder}/ecModel.json"
        
        if os.path.exists(ecModel_file):
            logger.info("  ♻️  Found existing ecModel.json, skipping model construction...")
            logger.info(f"  ✅ etcGEM already exists: {ecModel_file}")
        else:
            from .build_ecGEM import build_ecGEM
            ecModel_file = build_ecGEM(
                model_file, result_folder,
                f=f, ptot=ptot, sigma=sigma, lowerbound=lowerbound
            )
            logger.info(f"  ✅ etcGEM saved: {ecModel_file}")
        
        logger.info("\n"+"="*60)
        logger.info(f"✅ etcGEM Construction Complete @ {temperature}°C!")
        logger.info("="*60)
        
        return ecModel_file
    
    def get_model_stats(self, model_folder: str) -> Dict:
        """Get statistics for a built ecGEM/etcGEM model"""
        folder_path = Path(model_folder)
        
        if not folder_path.exists():
            return {}
        
        stats = {
            "folder": str(folder_path),
            "files": {}
        }
        
        # Check for key files
        key_files = [
            "metabolites_reactions_gpr.csv",
            "full_metabolites_reactions.csv",
            "reaction_kcat_mw.csv",
            "ecModel.json"
        ]
        
        for filename in key_files:
            file_path = folder_path / filename
            if file_path.exists():
                stats["files"][filename] = {
                    "exists": True,
                    "size": file_path.stat().st_size,
                    "path": str(file_path)
                }
                
                # Get row counts for CSV files
                if filename.endswith(".csv"):
                    try:
                        df = pd.read_csv(file_path)
                        stats["files"][filename]["rows"] = len(df)
                        stats["files"][filename]["columns"] = len(df.columns)
                    except:
                        pass
            else:
                stats["files"][filename] = {"exists": False}
        
        return stats
    
    def run_gem_analysis(
        self,
        model_file: str,
        algorithm: str,
        obj: Optional[str] = None,
        substrate: str = "EX_glc__D_e",
        concentration: float = 10.0,
        result_folder: Optional[str] = None
    ) -> str:
        """
        Submit GEM analysis task (FBA/pFBA/FVA for draft GEM, ecGEM for ec/etcGEM).
        
        Args:
            model_file: Path to model file (draft GEM XML or ecGEM JSON)
            algorithm: Algorithm to use ("FBA", "pFBA", "FVA", "ecGEM")
            obj: Target reaction ID (default: auto-detect biomass)
            substrate: Substrate reaction ID (for ecGEM only)
            concentration: Substrate concentration (for ecGEM only)
            result_folder: Optional custom output folder
            
        Returns:
            Task ID
        """
        model_name = os.path.basename(model_file).replace("_draft.xml", "").replace(".xml", "").replace(".json", "")
        
        if result_folder is None:
            model_dir = os.path.dirname(model_file)
            result_folder = f"{model_dir}/analysis_result"
        
        task_id = self.task_manager.start(
            self._run_gem_analysis_task,
            model_file,
            algorithm,
            obj,
            substrate,
            concentration,
            result_folder,
            prefix="gem-analysis-",
            task_name=f"GEM Analysis: {model_name} ({algorithm})",
            task_type="gem_analysis"
        )
        
        return task_id
    
    def _run_gem_analysis_task(
        self,
        logger,
        model_file: str,
        algorithm: str,
        obj: Optional[str],
        substrate: str,
        concentration: float,
        result_folder: str
    ) -> str:
        """Background task for running GEM analysis"""
        import sys
        import jpype
        
        os.makedirs(result_folder, exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"🔬 Running GEM Analysis: {algorithm}")
        logger.info("="*60)
        logger.info(f"Model: {model_file}")
        logger.info(f"Output: {result_folder}")
        
        # 确保 JVM 启动（用于 straindesign）
        if algorithm in ["FBA", "pFBA", "FVA"]:
            java_install_path = os.getenv("JAVA_HOME")
            if java_install_path:
                os.environ['JAVA_HOME'] = java_install_path
                logger.info(f"JAVA_HOME set to: {java_install_path}")
            
            if not jpype.isJVMStarted():
                try:
                    jpype.startJVM(
                        jpype.getDefaultJVMPath(),
                        "--enable-native-access=ALL-UNNAMED"
                    )
                    logger.info("JVM started successfully")
                except Exception as e:
                    logger.warning(f"JVM startup warning: {e}")
        
        # 导入必要的模块
        from ..run_GEM import run_ecGEM_fba, run_straindesign
        
        if algorithm == "ecGEM":
            logger.info(f"Parameters: obj={obj}, substrate={substrate}, concentration={concentration}")
            optimal_value = run_ecGEM_fba(
                model_file, result_folder,
                obj=obj, use_substrate=substrate, concentration=concentration
            )
            logger.info(f"✅ Optimal value (ecGEM pFBA): {optimal_value}")
            result_file = f"{result_folder}/ECMpy_solution_{obj or 'biomass'}_pfba.csv"
        else:
            logger.info(f"Parameters: obj={obj}")
            optimal_value = run_straindesign(
                model_file, algorithm=algorithm,
                target_reaction_id=obj, result_folder=result_folder
            )
            if optimal_value is not None:
                logger.info(f"✅ Optimal value ({algorithm}): {optimal_value}")
            else:
                logger.info(f"✅ {algorithm} analysis completed")
            result_file = f"{result_folder}/straindesign_{algorithm}_solution.csv"
        
        logger.info("="*60)
        logger.info(f"✅ Analysis Complete!")
        logger.info(f"Results saved to: {result_file}")
        logger.info("="*60)
        
        return result_file
    
    def run_ko_oe_analysis(
        self,
        model_file: str,
        analysis_type: str,
        target_ids: Optional[List[str]] = None,
        production_target: Optional[str] = None,
        knockout_threshold: float = 0.01,
        oe_fold_changes: List[float] = None,
        result_folder: Optional[str] = None
    ) -> str:
        """
        Submit KO/OE analysis task
        
        Args:
            model_file: Path to model file
            analysis_type: Type of analysis ("knockout_reaction", "knockout_gene", "overexpression", "comprehensive")
            target_ids: Specific target IDs to analyze (if None, analyze all)
            production_target: Target reaction ID for production optimization
            knockout_threshold: Threshold for essential identification
            oe_fold_changes: Fold changes for overexpression (default: [2.0, 5.0, 10.0])
            result_folder: Optional custom output folder
            
        Returns:
            Task ID
        """
        if oe_fold_changes is None:
            oe_fold_changes = [2.0, 5.0, 10.0]
        
        model_name = os.path.basename(model_file).replace("_draft.xml", "").replace(".xml", "").replace(".json", "")
        
        if result_folder is None:
            model_dir = os.path.dirname(model_file)
            result_folder = f"{model_dir}/ko_oe_analysis"
        
        task_id = self.task_manager.start(
            self._run_ko_oe_analysis_task,
            model_file,
            analysis_type,
            target_ids,
            production_target,
            knockout_threshold,
            oe_fold_changes,
            result_folder,
            prefix="ko-oe-",
            task_name=f"KO/OE Analysis: {model_name} ({analysis_type})",
            task_type="ko_oe_analysis"
        )
        
        return task_id
    
    def _run_ko_oe_analysis_task(
        self,
        logger,
        model_file: str,
        analysis_type: str,
        target_ids: Optional[List[str]],
        production_target: Optional[str],
        knockout_threshold: float,
        oe_fold_changes: List[float],
        result_folder: str
    ) -> str:
        """Background task for running KO/OE analysis"""
        from ..ko_oe_analysis import (
            load_model,
            batch_knockout_reactions,
            batch_knockout_genes,
            batch_overexpression_analysis,
            analyze_ko_oe_targets,
            find_essential_reactions,
            find_essential_genes
        )
        
        os.makedirs(result_folder, exist_ok=True)
        
        logger.info("="*60)
        logger.info(f"🧬 Running KO/OE Analysis: {analysis_type}")
        logger.info("="*60)
        logger.info(f"Model: {model_file}")
        logger.info(f"Output: {result_folder}")
        logger.info(f"Knockout Threshold: {knockout_threshold}")
        
        # Load model
        logger.info("Loading model...")
        model = load_model(model_file)
        
        # Set objective if production target is specified
        if production_target:
            logger.info(f"Setting objective to: {production_target}")
            model.objective = production_target
        
        result_files = []
        
        if analysis_type == "knockout_reaction":
            logger.info("Performing reaction knockout analysis...")
            df = batch_knockout_reactions(model, target_ids, result_folder)
            logger.info(f"Analyzed {len(df)} reactions")
            
            # Find essential reactions
            essential = find_essential_reactions(model, df, knockout_threshold)
            logger.info(f"Found {len(essential)} essential reactions")
            
            result_files.append(f"{result_folder}/knockout_reaction_results.csv")
            
        elif analysis_type == "knockout_gene":
            logger.info("Performing gene knockout analysis...")
            df = batch_knockout_genes(model, target_ids, result_folder)
            logger.info(f"Analyzed {len(df)} genes")
            
            # Find essential genes
            essential = find_essential_genes(model, df, knockout_threshold)
            logger.info(f"Found {len(essential)} essential genes")
            
            result_files.append(f"{result_folder}/knockout_gene_results.csv")
            
        elif analysis_type == "overexpression":
            logger.info(f"Performing overexpression analysis with fold changes: {oe_fold_changes}")
            df = batch_overexpression_analysis(model, target_ids, oe_fold_changes, result_folder)
            logger.info(f"Analyzed {len(df)} overexpression scenarios")
            
            result_files.append(f"{result_folder}/overexpression_results.csv")
            
        elif analysis_type == "comprehensive":
            logger.info("Performing comprehensive KO/OE analysis...")
            results = analyze_ko_oe_targets(
                model, result_folder, production_target,
                knockout_threshold, oe_fold_changes
            )
            
            logger.info(f"✅ Reaction knockouts: {len(results['reaction_knockout'])}")
            logger.info(f"✅ Gene knockouts: {len(results['gene_knockout'])}")
            logger.info(f"✅ Essential reactions: {len(results['essential_reactions'])}")
            logger.info(f"✅ Essential genes: {len(results['essential_genes'])}")
            logger.info(f"✅ Overexpression scenarios: {len(results['overexpression'])}")
            
            result_files = [
                f"{result_folder}/knockout_reaction_results.csv",
                f"{result_folder}/knockout_gene_results.csv",
                f"{result_folder}/essential_reactions.csv",
                f"{result_folder}/essential_genes.csv",
                f"{result_folder}/overexpression_results.csv"
            ]
        else:
            raise ValueError(f"Unknown analysis type: {analysis_type}")
        
        logger.info("="*60)
        logger.info("✅ KO/OE Analysis Complete!")
        logger.info(f"Results saved to: {result_folder}")
        for f in result_files:
            logger.info(f"  - {f}")
        logger.info("="*60)
        
        return result_folder

