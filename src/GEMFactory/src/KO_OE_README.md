# KO/OE Analysis Module

## Overview

The KO (Knockout) and OE (Overexpression) analysis module provides comprehensive tools for analyzing gene and reaction knockouts and overexpression scenarios in GEM models (Draft GEM, ecGEM, etcGEM).

## Features

### 1. Knockout Analysis
- **Single Reaction Knockout**: Test the effect of knocking out individual reactions
- **Single Gene Knockout**: Test the effect of knocking out individual genes
- **Batch Knockout**: Analyze all reactions/genes at once
- **Essential Identification**: Automatically identify essential genes and reactions

### 2. Overexpression Analysis
- **Reaction Overexpression**: Simulate gene overexpression by increasing reaction bounds
- **Multiple Fold Changes**: Test different overexpression levels (e.g., 2x, 5x, 10x)
- **Batch Analysis**: Test overexpression of all reactions

### 3. Comprehensive Analysis
- Combines all knockout and overexpression analyses
- Generates multiple output files for easy analysis
- Compatible with all GEM model types

## Usage

### Through Web Interface (Gradio)

1. Navigate to the **"🧬 KO/OE Analysis"** tab in GEMFactory
2. Select a model (Draft GEM, ecGEM, or etcGEM)
3. Choose analysis type:
   - `knockout_reaction`: Test single reaction knockouts
   - `knockout_gene`: Test single gene knockouts
   - `overexpression`: Simulate gene overexpression
   - `comprehensive`: All of the above
4. Configure parameters:
   - Target reaction (optional)
   - Knockout threshold
   - Overexpression fold changes
5. Click "Run KO/OE Analysis"
6. Download results from "Download Manager" tab

### Through Python API

```python
from src.GEMFactory.src.ko_oe_analysis import (
    load_model,
    batch_knockout_reactions,
    batch_knockout_genes,
    batch_overexpression_analysis,
    find_essential_reactions,
    find_essential_genes,
    analyze_ko_oe_targets
)

# Load model
model = load_model("path/to/model.xml")

# Comprehensive analysis
results = analyze_ko_oe_targets(
    model=model,
    result_folder="results/ko_oe",
    production_target=None,  # or specify a reaction ID
    knockout_threshold=0.01,
    oe_fold_changes=[2.0, 5.0, 10.0]
)

# Access results
reaction_ko = results['reaction_knockout']
gene_ko = results['gene_knockout']
essential_rxns = results['essential_reactions']
essential_genes = results['essential_genes']
oe_results = results['overexpression']
```

### Individual Analysis Functions

```python
# Knockout single reaction
result = knock_out_single_reaction(model, "PGI", optimize=True)

# Knockout single gene
result = knock_out_single_gene(model, "b0755", optimize=True)

# Batch knockout reactions
df = batch_knockout_reactions(model, result_folder="results")

# Batch knockout genes
df = batch_knockout_genes(model, result_folder="results")

# Overexpression analysis
df = batch_overexpression_analysis(
    model,
    fold_changes=[2.0, 5.0, 10.0],
    result_folder="results"
)

# Find essential reactions
essential = find_essential_reactions(model, production_threshold=0.01)

# Find essential genes
essential = find_essential_genes(model, production_threshold=0.01)
```

## Output Files

### Comprehensive Analysis Generates:

1. **`knockout_reaction_results.csv`**
   - Columns: reaction_id, status, objective_value, original_objective, objective_change, objective_change_percent
   - Contains knockout results for all reactions

2. **`knockout_gene_results.csv`**
   - Columns: gene_id, status, objective_value, original_objective, objective_change, objective_change_percent
   - Contains knockout results for all genes

3. **`essential_reactions.csv`**
   - List of essential reactions identified based on threshold

4. **`essential_genes.csv`**
   - List of essential genes identified based on threshold

5. **`overexpression_results.csv`**
   - Columns: reaction_id, fold_change, original_bounds, new_bounds, original_objective, new_objective, objective_change, status
   - Contains overexpression simulation results

6. **`cobrapy_reaction_deletion_results.csv`**
   - Results from COBRApy's single_reaction_deletion function

7. **`cobrapy_gene_deletion_results.csv`**
   - Results from COBRApy's single_gene_deletion function

## Parameters

### Knockout Threshold
- **Type**: float (0.0 - 1.0)
- **Default**: 0.01 (1%)
- **Description**: Minimum percentage change in objective value to consider a gene/reaction as essential
- **Example**: 0.01 means if knockout causes >1% change in objective, it's considered essential

### Overexpression Fold Changes
- **Type**: List of float
- **Default**: [2.0, 5.0, 10.0]
- **Description**: Multiplication factors for reaction upper bounds to simulate overexpression
- **Example**: [2.0, 5.0, 10.0] tests 2x, 5x, and 10x overexpression levels

### Production Target
- **Type**: str (optional)
- **Default**: None (uses model's default objective)
- **Description**: Reaction ID to optimize for (e.g., target product)
- **Example**: "EX_succ_e" for succinate production

## Model Compatibility

### Draft GEM
✅ **Fully Supported**
- All knockout and overexpression functions work
- Uses standard SBML format

### ecGEM
✅ **Fully Supported**
- All functions work with enzyme constraints
- Accounts for protein allocation constraints

### etcGEM
✅ **Fully Supported**
- All functions work with temperature-dependent constraints
- Includes temperature effects in predictions

## Algorithm Details

### Knockout Analysis
1. For each gene/reaction:
   - Create temporary context with model
   - Knock out the target
   - Optimize the model
   - Record objective value and status
2. Compare with original objective
3. Identify essential targets based on threshold

### Overexpression Analysis
1. For each reaction and fold change:
   - Create temporary context with model
   - Multiply upper bound by fold change
   - For reversible reactions, also adjust lower bound
   - Optimize the model
   - Record changes in objective value

### Essential Identification
- **Non-optimal**: Knockout makes model infeasible → Essential
- **Threshold**: Knockout causes objective change ≥ threshold → Essential

## Performance Notes

- **Batch knockout**: Time scales linearly with number of reactions/genes
- **Comprehensive analysis**: Most time-consuming but most informative
- **Recommended**: Start with smaller models or specific target lists
- **Large models**: May take hours for comprehensive analysis

## Example Use Cases

### 1. Find Essential Genes for Growth
```python
model = load_model("e_coli.xml")
results = batch_knockout_genes(model, result_folder="essential_analysis")
essential = find_essential_genes(model, results, production_threshold=0.01)
print(f"Found {len(essential)} essential genes")
```

### 2. Screen for Production Enhancement Targets
```python
model = load_model("producer_strain.xml")
model.objective = "EX_target_product_e"

# Test overexpression
oe_results = batch_overexpression_analysis(
    model,
    fold_changes=[2.0, 5.0, 10.0],
    result_folder="overexpression"
)

# Find best candidates
best = oe_results[oe_results['objective_change'] > 0].sort_values('objective_change', ascending=False)
print("Top overexpression targets:")
print(best.head(10))
```

### 3. Comprehensive Strain Design Analysis
```python
model = load_model("target_strain.xml")
results = analyze_ko_oe_targets(
    model=model,
    result_folder="strain_design",
    production_target="EX_target_e",
    knockout_threshold=0.05,
    oe_fold_changes=[2.0, 3.0, 5.0]
)

# Analyze results
print(f"Essential reactions: {len(results['essential_reactions'])}")
print(f"Essential genes: {len(results['essential_genes'])}")
print(f"OE scenarios tested: {len(results['overexpression'])}")
```

## References

- **COBRApy**: Ebrahim et al., 2013. COBRApy: COnstraints-Based Reconstruction and Analysis for Python.
- **Flux Balance Analysis**: Orth et al., 2010. What is flux balance analysis?
- **Gene Essentiality**: Joyce et al., 2006. Experimental and computational assessment of conditionally essential genes in Escherichia coli.

## Citation

If you use this module in your research, please cite:
```
SJTU-Software CASPIA Team (2025)
KO/OE Analysis Module for GEMFactory
https://github.com/SJTU-Software/CASPIA
```

