# CASPIAgent Tools Guide - Updated

## 📚 Table of Contents

1. [Overview](#overview)
2. [Tool Categories](#tool-categories)
3. [Quick Reference](#quick-reference)
4. [Detailed Documentation](#detailed-documentation)
5. [Workflow Examples](#workflow-examples)
6. [Best Practices](#best-practices)

---

## Overview

CASPIAgent provides a comprehensive suite of tools for computational biology and metabolic modeling. Tools are organized into two main categories:

- **Async Tools**: Submit long-running tasks that execute in the background
- **Sync Tools**: Execute quickly and return results immediately

**Total Available Tools:** 13

---

## Tool Categories

### 🔄 Async Tools - Task Submission

These tools submit tasks to background workers and return immediately with a task ID.

| Tool | Purpose | Duration | Status Check |
|------|---------|----------|--------------|
| `submit_gene_annotation` | Run GeneMarkS gene annotation | 10-30 min | ✅ Required |
| `submit_gem_build` | Build draft GEM (GeneMarkS + CarveMe) | 10-30 min | ✅ Required |
| `submit_ecgem_build` | Build enzyme-constrained GEM | 30-120 min | ✅ Required |
| `submit_etcgem_build` | Build enzyme-temp-constrained GEM | 40-150 min | ✅ Required |

---

### ⚡ Sync Tools - Direct Execution

These tools execute quickly and return results immediately.

| Tool | Purpose | Duration | Return Type |
|------|---------|----------|-------------|
| `predict_kcat` | Predict enzyme catalytic constant | < 1 min | Prediction value |
| `check_task_status` | Check status of submitted task | Instant | Status dict |
| `check_model_suitability` | Validate model for ecGEM/etcGEM | < 10 sec | Suitability status |
| `list_draft_gem_models` | List available draft GEMs | Instant | Model list |
| `list_ecgem_models` | List built ecGEMs | Instant | Model list |
| `list_etcgem_models` | List built etcGEMs | Instant | Model list |
| `get_model_statistics` | Get model statistics | < 5 sec | Statistics dict |

---

## Quick Reference

### Task Submission Pattern

```python
# 1. Submit task
result = tool_name(parameters)
task_id = result["task_id"]

# 2. Check status
status = check_task_status(task_id)

# 3. Wait and check again
# Repeat step 2 until status shows "Done" or "Failed"
```

---

## Detailed Documentation

### 1. submit_gene_annotation

**Purpose:** Submit a gene annotation task using GeneMarkS.

**Parameters:**
- `genome_file_path` (str, required): Path to genome FASTA file

**Returns:**
```python
{
    "success": True/False,
    "task_id": "gene-abc123",
    "message": "Status message"
}
```

**Example:**
```python
result = submit_gene_annotation(
    genome_file_path="data/genome.fna"
)
task_id = result["task_id"]
```

**Notes:**
- File must be in .fna, .fa, or .fasta format
- Gene annotation is prerequisite for GEM building
- Check status using `check_task_status(task_id)`

---

### 2. submit_gem_build

**Purpose:** Submit a complete draft GEM building pipeline (GeneMarkS + CarveMe).

**Parameters:**
- `genome_file_path` (str, required): Path to genome FASTA file
- `gapfill_medium` (str, optional): Gap-filling medium
  - Options: "None" (default), "M9", "LB", "M9,LB"

**Returns:**
```python
{
    "success": True/False,
    "task_id": "gem-xyz789",
    "message": "Status message"
}
```

**Example:**
```python
result = submit_gem_build(
    genome_file_path="data/genome.fna",
    gapfill_medium="M9"
)
```

**Notes:**
- Automatically runs gene annotation first
- Gap-filling improves model functionality
- M9: minimal medium, LB: rich medium

---

### 3. submit_ecgem_build

**Purpose:** Submit enzyme-constrained GEM (ecGEM) construction task.

**Parameters:**
- `model_file_path` (str, required): Path to draft GEM (.xml)
- `f` (float, optional): Enzyme fraction with kcat data (default: 0.405)
  - Range: 0.1 - 1.0
- `ptot` (float, optional): Total protein fraction g/gDW (default: 0.56)
  - Range: 0.1 - 1.0  
- `sigma` (float, optional): Enzyme saturation factor (default: 1.0)
  - Range: 0.1 - 2.0
- `lowerbound` (float, optional): Lower enzyme bound (default: 0.0)
  - Range: 0.0 - 0.1

**Returns:**
```python
{
    "success": True/False,
    "task_id": "ecgem-def456",
    "message": "Status message with parameters"
}
```

**Example:**
```python
# First check model suitability
suitability = check_model_suitability("data/model_draft.xml")

if suitability["is_suitable"]:
    result = submit_ecgem_build(
        model_file_path="data/model_draft.xml",
        f=0.405,
        ptot=0.56,
        sigma=1.0
    )
```

**Notes:**
- Requires draft GEM from `submit_gem_build`
- Always check model suitability first
- Adds enzyme kinetic constraints for better predictions

---

### 4. submit_etcgem_build

**Purpose:** Submit enzyme-temperature-constrained GEM (etcGEM) construction task.

**Parameters:**
- `model_file_path` (str, required): Path to draft GEM (.xml)
- `temperature` (float, required): Optimal growth temperature in °C
  - Range: 0 - 100
  - Example: 37 for E. coli, 65 for thermophiles
- `f` (float, optional): Same as ecGEM (default: 0.405)
- `ptot` (float, optional): Same as ecGEM (default: 0.56)
- `sigma` (float, optional): Same as ecGEM (default: 1.0)
- `lowerbound` (float, optional): Same as ecGEM (default: 0.0)

**Returns:**
```python
{
    "success": True/False,
    "task_id": "etcgem-ghi789",
    "message": "Status message with temperature and parameters"
}
```

**Example:**
```python
result = submit_etcgem_build(
    model_file_path="data/model_draft.xml",
    temperature=37.0,  # E. coli optimal temp
    f=0.405,
    ptot=0.56
)
```

**Notes:**
- All ecGEM features PLUS temperature effects
- Predicts Topt (optimal temperature) for each enzyme
- Temperature affects kcat values

---

### 5. predict_kcat

**Purpose:** Predict enzyme catalytic constant (kcat) for substrate-enzyme pair.

**Parameters:**
- `smiles` (str, required): SMILES string of substrate molecule
- `protein_sequence` (str, required): Amino acid sequence of enzyme
- `log_transform` (bool, optional): Apply log transformation (default: True)

**Returns:**
```python
{
    "success": True/False,
    "predicted_kcat": 123.45,
    "unit": "s^-1",
    "description": "Predicted kcat value is 123.45 s^-1",
    "raw_output": "Full prediction output..."
}
```

**Example:**
```python
result = predict_kcat(
    smiles="CC(=O)O",  # Acetic acid
    protein_sequence="MKTAYIAKQRQISFVKSHFSRQ...",
    log_transform=True
)
kcat_value = result["predicted_kcat"]
```

**Notes:**
- Fast execution (< 1 minute)
- Uses deep learning model
- SMILES must be valid chemical notation

---

### 6. check_task_status

**Purpose:** Check the status of any submitted async task.

**Parameters:**
- `task_id` (str, required): Task ID from submit_* functions

**Returns:**
```python
{
    "success": True/False,
    "task_id": "task-abc123",
    "task_name": "GEM Build: genome.fna",
    "status": "🚧 Running / ✅ Completed / ❌ Failed",
    "result": "Path to output file (if done)",
    "start_time": "2025-10-05 10:30:15",
    "message": "Detailed status message"
}
```

**Example:**
```python
# After submitting a task
task_id = gem_result["task_id"]

# Check status repeatedly
status = check_task_status(task_id)
print(status["status"])  # 🚧 Running...

# Wait and check again...
status = check_task_status(task_id)
print(status["status"])  # ✅ Completed
```

**Notes:**
- Check every 30-60 seconds for long tasks
- Status updates in real-time
- Logs available in Tasks Monitor tab

---

### 7. check_model_suitability

**Purpose:** Check if a draft GEM is suitable for ecGEM/etcGEM construction.

**Parameters:**
- `model_file_path` (str, required): Path to draft GEM file (.xml)

**Returns:**
```python
{
    "success": True/False,
    "is_suitable": True/False,
    "model_file": "path/to/model.xml",
    "details": [
        "Metabolite coverage: 85%",
        "Reaction coverage: 72%"
    ],
    "message": "✅ Model is suitable / ❌ Model is NOT suitable",
    "recommendation": "Action recommendation"
}
```

**Example:**
```python
suitability = check_model_suitability("data/model_draft.xml")

if suitability["is_suitable"]:
    print("✅ Can proceed with ecGEM/etcGEM")
    print("\n".join(suitability["details"]))
else:
    print("❌ Model not suitable")
    print(suitability["recommendation"])
```

**Notes:**
- Always check before ecGEM/etcGEM construction
- Requires >25% metabolite coverage
- Requires >25% reaction annotation coverage

---

### 8. list_draft_gem_models

**Purpose:** List all available draft GEM models.

**Parameters:** None

**Returns:**
```python
{
    "success": True/False,
    "count": 3,
    "models": [
        {
            "name": "E_coli_K12",
            "path": "data/CarveMe/E_coli_K12_draft.xml",
            "size_mb": 5.2,
            "modified": 1696512000.0
        },
        ...
    ],
    "message": "Found 3 draft GEM model(s)"
}
```

**Example:**
```python
result = list_draft_gem_models()

for model in result["models"]:
    print(f"Model: {model['name']}")
    print(f"  Size: {model['size_mb']} MB")
    print(f"  Path: {model['path']}")
```

**Notes:**
- Lists models in CarveMe output directory
- Sorted by modification time (newest first)
- Empty if no models built yet

---

### 9. list_ecgem_models

**Purpose:** List all built enzyme-constrained GEM (ecGEM) models.

**Parameters:** None

**Returns:**
```python
{
    "success": True/False,
    "count": 2,
    "models": [
        {
            "name": "E_coli_K12",
            "path": "data/ecGEM/E_coli_K12/ecModel.json",
            "folder": "data/ecGEM/E_coli_K12",
            "size_mb": 15.8,
            "modified": 1696512000.0
        },
        ...
    ],
    "message": "Found 2 ecGEM model(s)"
}
```

**Example:**
```python
result = list_ecgem_models()
print(f"Available ecGEMs: {result['count']}")
```

---

### 10. list_etcgem_models

**Purpose:** List all built enzyme-temperature-constrained GEM (etcGEM) models.

**Parameters:** None

**Returns:**
```python
{
    "success": True/False,
    "count": 2,
    "models": [
        {
            "name": "E_coli_K12_T=37.0",
            "path": "data/etcGEM/E_coli_K12_T=37.0/ecModel.json",
            "folder": "data/etcGEM/E_coli_K12_T=37.0",
            "temperature": 37.0,
            "size_mb": 18.3,
            "modified": 1696512000.0
        },
        ...
    ],
    "message": "Found 2 etcGEM model(s)"
}
```

**Example:**
```python
result = list_etcgem_models()

for model in result["models"]:
    print(f"{model['name']} @ {model['temperature']}°C")
```

---

### 11. get_model_statistics

**Purpose:** Get detailed statistics for a built ecGEM or etcGEM model.

**Parameters:**
- `model_folder_path` (str, required): Path to model result folder

**Returns:**
```python
{
    "success": True/False,
    "folder": "data/ecGEM/E_coli_K12",
    "files": [
        {
            "filename": "metabolites_reactions_gpr.csv",
            "size_kb": 1024.5,
            "path": "full/path/to/file.csv",
            "rows": 2500,
            "columns": 15
        },
        ...
    ],
    "message": "Retrieved statistics for 4 file(s)"
}
```

**Example:**
```python
# Get folder from list_ecgem_models
models = list_ecgem_models()
folder = models["models"][0]["folder"]

# Get statistics
stats = get_model_statistics(folder)

for file in stats["files"]:
    print(f"{file['filename']}: {file['rows']} rows")
```

**Notes:**
- Shows all intermediate and final files
- Includes row/column counts for CSV files
- Useful for debugging and validation

---

## Workflow Examples

### Workflow 1: Build Complete Pipeline

```python
# Step 1: Build draft GEM
gem_result = submit_gem_build(
    genome_file_path="data/ecoli_genome.fna",
    gapfill_medium="M9"
)
gem_task_id = gem_result["task_id"]

# Step 2: Monitor until complete
import time
while True:
    status = check_task_status(gem_task_id)
    print(status["status"])
    
    if "Done" in status["status"]:
        draft_model_path = status["result"]
        break
    elif "Failed" in status["status"]:
        print("GEM build failed!")
        break
    
    time.sleep(60)  # Check every minute

# Step 3: Check model suitability
suitability = check_model_suitability(draft_model_path)

if not suitability["is_suitable"]:
    print("Model not suitable for ecGEM")
    exit()

# Step 4: Build ecGEM
ecgem_result = submit_ecgem_build(
    model_file_path=draft_model_path,
    f=0.405,
    ptot=0.56
)
ecgem_task_id = ecgem_result["task_id"]

# Step 5: Monitor ecGEM build
while True:
    status = check_task_status(ecgem_task_id)
    print(status["status"])
    
    if "Done" in status["status"]:
        ecgem_model_path = status["result"]
        print(f"✅ ecGEM built: {ecgem_model_path}")
        break
    
    time.sleep(120)  # Check every 2 minutes
```

---

### Workflow 2: Temperature Analysis

```python
# Build etcGEMs at different temperatures
temperatures = [25, 30, 37, 42, 50]
model_path = "data/CarveMe/thermophile_draft.xml"

task_ids = []

for temp in temperatures:
    result = submit_etcgem_build(
        model_file_path=model_path,
        temperature=temp,
        f=0.4,
        ptot=0.55
    )
    task_ids.append((temp, result["task_id"]))
    print(f"Submitted etcGEM @ {temp}°C: {result['task_id']}")

# Monitor all tasks
import time
while task_ids:
    time.sleep(180)  # Check every 3 minutes
    
    completed = []
    for temp, task_id in task_ids:
        status = check_task_status(task_id)
        
        if "Done" in status["status"]:
            print(f"✅ {temp}°C model complete")
            completed.append((temp, task_id))
        elif "Failed" in status["status"]:
            print(f"❌ {temp}°C model failed")
            completed.append((temp, task_id))
    
    for item in completed:
        task_ids.remove(item)

print("All temperature models complete!")
```

---

### Workflow 3: Quick Kcat Prediction

```python
# Predict kcat for multiple substrate-enzyme pairs
pairs = [
    ("CC(=O)O", "MKTAYIAKQRQ..."),  # Acetic acid
    ("C(C(=O)O)O", "MALQTHYSAQ..."),  # Glycolic acid
    # Add more pairs...
]

predictions = []

for smiles, sequence in pairs:
    result = predict_kcat(
        smiles=smiles,
        protein_sequence=sequence,
        log_transform=True
    )
    
    if result["success"]:
        predictions.append({
            "smiles": smiles,
            "kcat": result["predicted_kcat"],
            "unit": result["unit"]
        })
        print(f"✅ {smiles}: {result['predicted_kcat']} s^-1")
    else:
        print(f"❌ {smiles}: {result['error']}")

# Analyze predictions
avg_kcat = sum(p["kcat"] for p in predictions) / len(predictions)
print(f"Average kcat: {avg_kcat:.2f} s^-1")
```

---

### Workflow 4: Model Management

```python
# List all available models
draft_models = list_draft_gem_models()
ecgem_models = list_ecgem_models()
etcgem_models = list_etcgem_models()

print(f"Draft GEMs: {draft_models['count']}")
print(f"ecGEMs: {ecgem_models['count']}")
print(f"etcGEMs: {etcgem_models['count']}")

# Get statistics for each ecGEM
for model in ecgem_models["models"]:
    print(f"\n=== {model['name']} ===")
    stats = get_model_statistics(model["folder"])
    
    for file in stats["files"]:
        if "rows" in file:
            print(f"{file['filename']}: {file['rows']} reactions/metabolites")
```

---

## Best Practices

### 1. Task Monitoring

✅ **DO:**
- Check task status every 1-3 minutes for long tasks
- Store task IDs for later reference
- Use Tasks Monitor tab for detailed logs
- Set reasonable timeouts

❌ **DON'T:**
- Poll too frequently (< 30 seconds)
- Submit duplicate tasks if first is still running
- Forget to check task status after submission

---

### 2. Model Building

✅ **DO:**
- Always check model suitability before ecGEM/etcGEM
- Use appropriate gap-filling medium for your organism
- Document parameters used
- Save intermediate results

❌ **DON'T:**
- Skip suitability checks
- Use default parameters without understanding them
- Build ecGEM from poor quality draft GEM

---

### 3. Parameter Selection

✅ **DO:**
- Start with default parameters
- Adjust based on organism physiology
- Test parameter sensitivity
- Document choices

❌ **DON'T:**
- Use extreme values without justification
- Copy parameters from very different organisms
- Ignore organism-specific literature

---

### 4. Error Handling

✅ **DO:**
- Check `success` field in all returns
- Read error messages carefully
- Validate file paths before submission
- Review logs for failed tasks

❌ **DON'T:**
- Ignore error messages
- Retry immediately without fixing issues
- Assume success without checking

---

## Tool Combinations

### Common Patterns

1. **Gene → GEM → ecGEM**
   ```
   submit_gene_annotation → submit_gem_build → submit_ecgem_build
   ```

2. **GEM → Validate → ecGEM**
   ```
   submit_gem_build → check_model_suitability → submit_ecgem_build
   ```

3. **List → Select → Build**
   ```
   list_draft_gem_models → check_model_suitability → submit_etcgem_build
   ```

4. **Build → Monitor → Analyze**
   ```
   submit_ecgem_build → check_task_status → get_model_statistics
   ```

---

## Support

For questions or issues:
- See `TASK_LOGGING_SYSTEM_GUIDE.md` for background task details
- See `GEMFACTORY_USER_GUIDE.md` for GEM building details
- Check Tasks Monitor tab for real-time logs
- Contact: SJTU-Software Team

---

## Version History

- **v1.0** (2025-10): Initial tool set (5 tools)
- **v2.0** (2025-10): Added ecGEM/etcGEM tools (13 tools total)
  - Added: submit_ecgem_build
  - Added: submit_etcgem_build
  - Added: check_model_suitability
  - Added: list_draft_gem_models
  - Added: list_ecgem_models
  - Added: list_etcgem_models
  - Added: get_model_statistics

---

## License

Part of SJTU-software-CASPIA project.

