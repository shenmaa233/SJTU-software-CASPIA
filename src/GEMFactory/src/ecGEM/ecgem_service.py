"""
ecGEM Service
=============

Service layer for ecGEM and etcGEM construction.
Provides clean API for Web UI integration.
"""

import os
import pandas as pd
from pathlib import Path
from typing import Dict, Optional, Tuple
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

