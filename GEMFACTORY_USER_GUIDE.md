# GEMFactory User Guide

## 📚 Table of Contents

1. [Overview](#overview)
2. [Features](#features)
3. [Quick Start](#quick-start)
4. [Detailed Workflows](#detailed-workflows)
5. [Parameters Reference](#parameters-reference)
6. [File Structure](#file-structure)
7. [Troubleshooting](#troubleshooting)
8. [API Documentation](#api-documentation)

---

## Overview

**GEMFactory** is a comprehensive platform for building and enhancing genome-scale metabolic models (GEMs). It integrates multiple state-of-the-art tools to create:

- **Draft GEMs** from genome sequences
- **ecGEMs** (enzyme-constrained) with kinetic parameters
- **etcGEMs** (enzyme-temperature-constrained) with thermal adaptation

### Key Technologies

- **GeneMarkS**: Gene annotation
- **CarveMe**: GEM reconstruction
- **DLKcat**: Deep learning-based kcat prediction
- **ThermoKinetics**: Temperature-dependent kinetics modeling

---

## Features

### 🧬 Draft GEM Builder

Build draft metabolic models from genome sequences.

**Input:**
- Genome FASTA file (.fna, .fa, .fasta)

**Output:**
- Draft GEM in SBML format (.xml)
- Gene annotations (GFF, FNN, FAA)

**Features:**
- Automatic gene annotation
- Gap-filling options (M9, LB, M9+LB, or None)
- Clean protein FASTA generation

---

### ⚗️ ecGEM Builder

Add enzyme kinetic constraints to draft GEMs.

**Input:**
- Draft GEM model
- Protein sequences (automatically retrieved)

**Output:**
- Enzyme-constrained GEM (JSON format)
- Kcat predictions
- Kcat/MW ratios

**Features:**
- Deep learning-based kcat prediction
- Protein allocation constraints
- Improved flux predictions
- Better growth rate accuracy

---

### 🌡️ etcGEM Builder

Add both enzyme AND temperature constraints.

**Input:**
- Draft GEM model
- Protein sequences
- Optimal growth temperature

**Output:**
- Enzyme-temperature-constrained GEM
- Temperature-dependent kcat values
- Topt (optimal temperature) predictions

**Features:**
- All ecGEM capabilities
- Temperature-dependent kinetics
- Thermal adaptation modeling
- Multi-temperature simulations

---

### 🔍 Model Viewer

Browse and analyze all constructed models.

**Features:**
- View all Draft/ecGEM/etcGEM models
- Model statistics and metadata
- File size and modification dates
- Quick access to model files

---

### 📊 Results Manager

Manage intermediate and final files.

**Features:**
- Browse result folders
- View all generated files
- File size and timestamps
- Organized by model type

---

## Quick Start

### 1. Build a Draft GEM

```
1. Go to "🧬 Draft GEM" tab
2. Upload or select a genome file
3. Choose gap-filling medium (optional)
4. Click "🚀 Build Draft GEM"
5. Monitor progress in logs
```

**Time:** ~10-30 minutes depending on genome size

---

### 2. Build an ecGEM

```
1. Go to "⚗️ ecGEM Builder" tab
2. Click "🔄 Refresh Models"
3. Select a draft GEM
4. Click "🔍 Check Suitability" (recommended)
5. Adjust parameters if needed
6. Click "🏗️ Build ecGEM"
7. Monitor progress
```

**Time:** ~30-120 minutes depending on model size

---

### 3. Build an etcGEM

```
1. Go to "🌡️ etcGEM Builder" tab
2. Select a draft GEM
3. Set optimal temperature (e.g., 37°C for E. coli)
4. Adjust parameters if needed
5. Click "🌡️ Build etcGEM"
6. Monitor progress
```

**Time:** ~40-150 minutes (includes Topt prediction)

---

## Detailed Workflows

### Workflow 1: E. coli Draft GEM

**Scenario:** You have an E. coli genome and want to build a basic metabolic model.

1. **Prepare Genome**
   - Download E. coli genome from NCBI
   - Format: FASTA (.fna)
   - Example: `GCF_000005845.2_ASM584v2_genomic.fna`

2. **Upload to GEMFactory**
   - Navigate to "🧬 Draft GEM" tab
   - Click "Upload Genome" and select file
   - Or place file in `src/GEMFactory/data/Genome/` and refresh

3. **Configure Build**
   - Gap-filling: `M9` (for minimal medium compatibility)
   - Or `None` for no gap-filling

4. **Run Pipeline**
   - Click "🚀 Build Draft GEM"
   - Wait for completion (~15 minutes)

5. **Output Location**
   - GEM: `src/GEMFactory/data/CarveMe/[genome]_draft.xml`
   - Annotations: `src/GEMFactory/data/GeneMarkS/[genome]/`

---

### Workflow 2: ecGEM with Kinetic Constraints

**Scenario:** Improve flux predictions with enzyme kinetics.

1. **Prerequisites**
   - Draft GEM from Workflow 1
   - Protein sequences (auto-retrieved)

2. **Check Suitability**
   - Go to "⚗️ ecGEM Builder"
   - Select draft GEM
   - Click "🔍 Check Suitability"
   - Ensure metabolite & reaction coverage > 25%

3. **Configure Parameters**
   - **f** (0.405): Fraction of enzymes with kcat data
   - **Ptot** (0.56 g/gDW): Total protein content
   - **σ** (1.0): Average saturation
   - **Lower Bound** (0.0): Minimum enzyme usage

4. **Run Build**
   - Click "🏗️ Build ecGEM"
   - Monitor progress (~60 minutes)

5. **Outputs**
   - Folder: `src/GEMFactory/data/ecGEM/[model]/`
   - Files:
     - `metabolites_reactions_gpr.csv` - Reactions & genes
     - `full_metabolites_reactions.csv` - With predicted kcat
     - `reaction_kcat_mw.csv` - Final kcat/MW values
     - `ecModel.json` - Final ecGEM

---

### Workflow 3: Temperature-Dependent etcGEM

**Scenario:** Model organism at specific growth temperature.

1. **Prerequisites**
   - Draft GEM
   - Know optimal growth temperature

2. **Set Temperature**
   - Go to "🌡️ etcGEM Builder"
   - Select draft GEM
   - Set temperature slider:
     - E. coli: 37°C
     - Thermophiles: 55-80°C
     - Psychrophiles: 10-20°C

3. **Configure Parameters**
   - Same as ecGEM (f, Ptot, σ, lower bound)
   - Temperature affects kcat calculations

4. **Run Build**
   - Click "🌡️ Build etcGEM"
   - Wait for completion (~90 minutes)

5. **Outputs**
   - Folder: `src/GEMFactory/data/etcGEM/[model]_T=[temp]/`
   - Includes temperature-adjusted kcat values
   - Topt predictions for each enzyme

---

## Parameters Reference

### Draft GEM Parameters

#### Gap-filling Medium

| Option | Description | Use Case |
|--------|-------------|----------|
| None | No gap-filling | Pure reconstruction |
| M9 | Minimal medium | Minimal growth conditions |
| LB | Rich medium | Complex growth conditions |
| M9,LB | Both media | Maximum functionality |

---

### ecGEM/etcGEM Parameters

#### f (Enzyme Fraction)

- **Range:** 0.1 - 1.0
- **Default:** 0.405 (40.5%)
- **Description:** Fraction of enzymes with available kcat values
- **Effect:** Higher = tighter constraints

**Recommended Values:**
- Well-studied organisms (E. coli): 0.4-0.5
- Less-studied organisms: 0.2-0.3

---

#### Ptot (Total Protein)

- **Range:** 0.1 - 1.0 g/gDW
- **Default:** 0.56 g/gDW
- **Description:** Total protein fraction of cell dry weight
- **Effect:** Limits total enzyme allocation

**Typical Values:**
- E. coli: 0.55-0.60 g/gDW
- Yeast: 0.45-0.50 g/gDW
- Bacteria: 0.50-0.60 g/gDW

---

#### σ (Saturation Factor)

- **Range:** 0.1 - 2.0
- **Default:** 1.0
- **Description:** Average enzyme saturation level
- **Effect:** Adjusts effective enzyme capacity

**Interpretation:**
- σ < 1: Enzymes operating below saturation
- σ = 1: Average saturation
- σ > 1: High substrate availability

---

#### Lower Bound

- **Range:** 0.0 - 0.1
- **Default:** 0.0
- **Description:** Minimum allowed enzyme usage
- **Effect:** Prevents zero enzyme allocations

**Use Cases:**
- 0.0: Standard (recommended)
- > 0: Force minimum enzyme expression

---

### Temperature (etcGEM only)

- **Range:** 0 - 100°C
- **Default:** 37°C
- **Description:** Optimal growth temperature
- **Effect:** Adjusts kcat via thermal kinetics

**Examples:**
- E. coli: 37°C
- B. subtilis: 37°C
- Thermus thermophilus: 65°C
- Psychrobacter: 15°C

---

## File Structure

### Input Files

```
src/GEMFactory/data/
├── Genome/                          # Input genomes
│   ├── GCF_000005845.2_ASM584v2_genomic.fna
│   └── ...
└── temp/                            # Temporary files
```

### Output Structure

```
src/GEMFactory/data/
├── GeneMarkS/                       # Gene annotations
│   └── [genome]/
│       ├── [genome].gff             # Gene features
│       ├── [genome]_gene.fasta      # Gene nucleotides
│       ├── [genome]_protein.fasta   # Protein sequences
│       └── [genome]_protein_clean.fasta  # Cleaned for CarveMe
│
├── CarveMe/                         # Draft GEMs
│   └── [genome]_draft.xml           # SBML model
│
├── ecGEM/                           # Enzyme-constrained GEMs
│   └── [genome]/
│       ├── metabolites_reactions_gpr.csv
│       ├── full_metabolites_reactions.csv
│       ├── reaction_kcat_mw.csv
│       └── ecModel.json             # Final ecGEM
│
└── etcGEM/                          # Enzyme-temperature GEMs
    └── [genome]_T=[temp]/
        ├── metabolites_reactions_gpr.csv
        ├── full_metabolites_reactions.csv
        ├── reaction_kcat_mw.csv
        └── ecModel.json             # Final etcGEM
```

---

## Troubleshooting

### Issue: Model Not Suitable for ecGEM

**Symptoms:**
```
❌ The coverage of metabolites is too low (20.0%)
❌ The coverage of reactions is too low (15.0%)
```

**Solutions:**
1. Check if genome annotation was successful
2. Verify CarveMe reconstruction quality
3. Try different gap-filling options
4. Use a well-annotated reference genome

---

### Issue: Kcat Prediction Fails

**Symptoms:**
```
Error in parameter prediction
```

**Solutions:**
1. Check protein FASTA file exists
2. Verify GPU/CPU availability for deep learning
3. Check SMILES generation for metabolites
4. Review logs for specific errors

---

### Issue: Task Stuck/Not Progressing

**Symptoms:**
- Status shows "🚧 Running..." for > 3 hours
- No log updates

**Solutions:**
1. Check Tasks Monitor tab for detailed status
2. Review log file in `logs/` directory
3. Check system resources (CPU, memory)
4. Restart task if necessary

---

### Issue: Out of Memory

**Symptoms:**
```
MemoryError or system slowdown
```

**Solutions:**
1. Close other applications
2. Process smaller models first
3. Reduce batch size in predictions
4. Use machine with more RAM

---

### Issue: Temperature Not Applied

**Symptoms:**
- etcGEM results same as ecGEM
- No Topt predictions

**Solutions:**
1. Verify temperature was set (not 0 or None)
2. Check `is_etc` flag in parameters
3. Ensure etcGEM pipeline was used (not ecGEM)
4. Review logs for temperature mentions

---

## API Documentation

### Using GEMFactory Programmatically

```python
from src.GEMFactory.src.ecGEM.ecgem_service import ECGEMService
from src.utils import get_task_manager

# Initialize services
ecgem_service = ECGEMService()
task_manager = get_task_manager()

# Check model suitability
is_suitable, messages = ecgem_service.check_model_suitability(
    "src/GEMFactory/data/CarveMe/genome_draft.xml"
)

# Build ecGEM
task_id = ecgem_service.build_ecgem(
    model_file="src/GEMFactory/data/CarveMe/genome_draft.xml",
    f=0.405,
    ptot=0.56,
    sigma=1.0,
    lowerbound=0.0
)

# Monitor progress
logs, status, result = task_manager.poll(task_id)
print(f"Status: {status}")

# Build etcGEM
task_id = ecgem_service.build_etcgem(
    model_file="src/GEMFactory/data/CarveMe/genome_draft.xml",
    temperature=37.0,
    f=0.405,
    ptot=0.56,
    sigma=1.0,
    lowerbound=0.0
)
```

---

### ECGEMService Methods

#### `check_model_suitability(model_file: str) -> Tuple[bool, list]`

Check if a model is suitable for ecGEM construction.

**Returns:** (is_suitable, messages)

---

#### `list_draft_models() -> list`

List all available draft GEM models.

**Returns:** List of dicts with model metadata

---

#### `list_ecgem_models() -> list`

List all built ecGEM models.

---

#### `list_etcgem_models() -> list`

List all built etcGEM models.

---

#### `build_ecgem(...) -> str`

Submit ecGEM building task.

**Returns:** Task ID

---

#### `build_etcgem(...) -> str`

Submit etcGEM building task.

**Returns:** Task ID

---

#### `get_model_stats(model_folder: str) -> Dict`

Get statistics for a built model.

**Returns:** Dict with file information and statistics

---

## Best Practices

### 1. Model Quality

✅ **DO:**
- Use well-annotated genomes from NCBI
- Check model suitability before ecGEM/etcGEM
- Verify protein sequences are clean
- Review logs for warnings

❌ **DON'T:**
- Skip suitability checks
- Use poorly assembled genomes
- Ignore low coverage warnings

---

### 2. Parameter Selection

✅ **DO:**
- Start with default parameters
- Adjust based on organism type
- Document parameter choices
- Test sensitivity

❌ **DON'T:**
- Use extreme parameter values without justification
- Copy parameters across very different organisms
- Ignore organism-specific physiology

---

### 3. Result Validation

✅ **DO:**
- Compare with experimental data
- Check growth rates
- Validate flux distributions
- Review kcat distributions

❌ **DON'T:**
- Trust model predictions blindly
- Skip biological validation
- Ignore unrealistic fluxes

---

### 4. Workflow Organization

✅ **DO:**
- Name files clearly
- Keep organized folder structure
- Document build parameters
- Save intermediate results

❌ **DON'T:**
- Overwrite previous builds
- Delete intermediate files prematurely
- Mix results from different versions

---

## Performance Expectations

### Build Times (Approximate)

| Task | Small Genome | Medium Genome | Large Genome |
|------|--------------|---------------|--------------|
| Draft GEM | 10-15 min | 20-30 min | 40-60 min |
| ecGEM | 30-45 min | 60-90 min | 120-180 min |
| etcGEM | 40-60 min | 90-120 min | 150-240 min |

*Times assume: 8-core CPU, 16GB RAM, SSD storage*

---

### Disk Space Requirements

| Model Type | Small | Medium | Large |
|------------|-------|--------|-------|
| Draft GEM | 1-5 MB | 5-10 MB | 10-50 MB |
| ecGEM | 50-100 MB | 100-200 MB | 200-500 MB |
| etcGEM | 60-120 MB | 120-250 MB | 250-600 MB |

---

## References

1. **CarveMe**: Machado et al. (2018). Fast automated reconstruction of genome-scale metabolic models for microbial species and communities. *Nucleic Acids Research*.

2. **GeneMarkS**: Besemer et al. (2001). GeneMarkS: a self-training method for prediction of gene starts in microbial genomes. *Bioinformatics*.

3. **DLKcat**: Li et al. (2022). Deep learning-based kcat prediction enables improved enzyme-constrained model reconstruction. *Nature Catalysis*.

4. **ecGEM**: Sánchez et al. (2017). Improving the phenotype predictions of a yeast genome-scale metabolic model by incorporating enzymatic constraints. *Molecular Systems Biology*.

---

## Support

For questions, issues, or contributions:
- GitHub Issues: [SJTU-software-CASPIA](https://github.com/...)
- Documentation: See `TASK_LOGGING_SYSTEM_GUIDE.md` for background task details
- Team Contact: SJTU-Software Team

---

## Version History

- **v1.0** (2025-10): Initial multi-tab interface release
  - Draft GEM builder
  - ecGEM builder
  - etcGEM builder
  - Model viewer
  - Results manager

---

## License

Part of SJTU-software-CASPIA project.

