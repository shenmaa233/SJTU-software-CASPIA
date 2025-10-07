# CASPIA: Comprehensive AI System for Protein and Integrated Analysis

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![iGEM 2025](https://img.shields.io/badge/iGEM-2025-green.svg)](https://2025.igem.org/)
[![Team](https://img.shields.io/badge/team-SJTU--Software-red.svg)](https://2025.igem.org/teams)

> **Team SJTU-Software 2025 Official Software Tool**
> 
> This repository contains the complete source code for CASPIA, an AI-powered platform designed to revolutionize synthetic biology research through intelligent automation, knowledge retrieval, and metabolic modeling.

**Team Wiki**: [Visit our iGEM Wiki](https://2025.igem.org/teams) (to be updated)

---

## Table of Contents

- [Overview](#overview)
- [Key Features](#key-features)
- [Architecture](#architecture)
- [Installation](#installation)
  - [Requirements](#requirements)
  - [Setup Instructions](#setup-instructions)
- [Usage](#usage)
  - [Quick Start](#quick-start)
  - [Module-Specific Usage](#module-specific-usage)
- [Modules Description](#modules-description)
  - [CASPIAgent](#caspiagent)
  - [GEMFactory](#gemfactory)
  - [CASPIA-RAG](#caspia-rag)
  - [Tasks Monitor](#tasks-monitor)
- [Examples](#examples)
- [Project Structure](#project-structure)
- [Contributing](#contributing)
- [Testing](#testing)
- [Citation](#citation)
- [Authors and Acknowledgments](#authors-and-acknowledgments)
- [License](#license)
- [Contact](#contact)

---

## Overview

**CASPIA (Comprehensive AI System for Protein and Integrated Analysis)** is an advanced, modular software platform developed by Team SJTU-Software for the iGEM 2025 competition. The platform integrates cutting-edge artificial intelligence technologies with synthetic biology workflows to provide researchers with:

- **Intelligent Conversational Agent**: Natural language interface for complex biological queries
- **Automated Metabolic Modeling**: Genome-scale metabolic model (GEM) construction and analysis
- **Knowledge Retrieval System**: RAG-based document understanding and question-answering
- **Workflow Management**: Real-time monitoring and orchestration of computational tasks

CASPIA addresses key challenges in synthetic biology research by reducing manual effort, improving reproducibility, and enabling researchers to focus on high-level scientific questions rather than technical implementation details.

### Why CASPIA?

Traditional synthetic biology workflows often require:
- Manual literature review and knowledge extraction
- Complex programming skills for metabolic modeling
- Scattered tools across different platforms
- Time-consuming data integration and analysis

CASPIA provides a unified, user-friendly solution that:
- ✅ Automates literature mining and knowledge synthesis
- ✅ Simplifies genome-scale metabolic model construction
- ✅ Offers intuitive natural language interfaces
- ✅ Integrates state-of-the-art AI models (GPT-4, LLaMA, etc.)
- ✅ Supports reproducible computational workflows

---

## Key Features

### 🤖 **CASPIAgent**
- Multi-turn conversational AI powered by large language models
- Tool-augmented reasoning for biological database queries
- Context-aware responses with citation support
- Customizable knowledge base integration

### 🧬 **GEMFactory**
- Automated genome-scale metabolic model (GEM) construction
- Integration with COBRApy, CarveMe, and other modeling tools
- Flux balance analysis (FBA) and optimization
- Model validation and quality assessment
- Export models in SBML and other standard formats

### 🔍 **CASPIA-RAG**
- Retrieval-Augmented Generation for scientific literature
- PDF/DOCX document parsing and indexing
- Multi-modal understanding (text + images)
- Semantic search with ChromaDB vector store
- Automatic translation and summarization

### 📊 **Tasks Monitor**
- Real-time task tracking and visualization
- Job queue management for long-running computations
- Resource usage monitoring
- Results aggregation and export

---

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      CASPIA Platform                         │
├─────────────────────────────────────────────────────────────┤
│                                                               │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐      │
│  │ CASPIAgent   │  │  GEMFactory  │  │ CASPIA-RAG   │      │
│  │              │  │              │  │              │      │
│  │ • Chat UI    │  │ • Model Gen  │  │ • Doc Parser │      │
│  │ • Tools      │  │ • FBA        │  │ • VectorDB   │      │
│  │ • Memory     │  │ • Optimize   │  │ • Retrieval  │      │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘      │
│         │                 │                  │               │
│         └─────────────────┴──────────────────┘               │
│                           │                                   │
│                  ┌────────▼────────┐                         │
│                  │ Tasks Monitor   │                         │
│                  │ • Queue Mgmt    │                         │
│                  │ • Progress Track│                         │
│                  └─────────────────┘                         │
│                                                               │
├─────────────────────────────────────────────────────────────┤
│                    Core Infrastructure                        │
│  • LLM Backend (vLLM, OpenAI, etc.)                          │
│  • Vector Database (ChromaDB)                                │
│  • Web Framework (Gradio)                                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Installation

### Requirements

**System Requirements:**
- Operating System: Linux (Ubuntu 20.04+), macOS (10.15+), or WSL2 on Windows
- RAM: 16GB minimum (32GB recommended for large models)
- GPU: CUDA-compatible GPU with 24GB+ VRAM recommended (for local LLM inference)
- Storage: 20GB+ free space

**Software Dependencies:**
- CUDA Toolkit 12.8 (for GPU acceleration)
- Git LFS (for large model files)

### Setup Instructions

1. **Clone the Repository**

```bash
git clone https://github.com/shenmaa233/SJTU-software-CASPIA.git
cd SJTU-software-CASPIA
```

2. **Create Virtual Environment**

```bash
# Using conda (recommended)
conda create -n caspia python=3.10
conda activate caspia

# Or using venv
python3 -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. **Install Dependencies**

```bash
# Install PyTorch with CUDA support (adjust CUDA version as needed)

pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
pip install torch-scatter -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.7.1+cu128.html
pip install torch-geometric

# diamond
conda install -c bioconda -c conda-forge diamond=2.1.13

# Install project dependencies
pip install -r requirements.txt
```

4. **Configure Environment Variables**

Create a `.env` file in the project root:

```bash
# OpenAI API (if using GPT models)
OPENAI_API_KEY=your_openai_api_key_here

# Optional: Other LLM API keys
DASHSCOPE_API_KEY=your_dashscope_key  # For Qwen models

# Database paths
CHROMA_DB_PATH=./src/CASPIA_RAG/db

# Model cache directory
HF_HOME=./models
TRANSFORMERS_CACHE=./models
```

5. **Download Required Models** (Optional, for local inference)

```bash
# Example: Download embedding model
huggingface-cli download sentence-transformers/all-MiniLM-L6-v2

# Example: Download LLM model (requires significant storage)
# huggingface-cli download meta-llama/Llama-2-7b-chat-hf
```

6. **Verify Installation**

```bash
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA Available: {torch.cuda.is_available()}')"
```

---

## Usage

### Quick Start

Launch the CASPIA web interface:

```bash
python webui.py
```

The application will start on `http://localhost:7860` (or `http://0.0.0.0:7860` for network access). Open this URL in your web browser to access the interface.

### Module-Specific Usage

#### Using CASPIAgent

1. Navigate to the **🤖 CASPIAgent** tab
2. Type your biological question in natural language
3. The agent will process your query using available tools and knowledge bases
4. Receive answers with citations and relevant information

**Example queries:**
- "What is the function of the lacZ gene in E. coli?"
- "Design a plasmid for expressing GFP in yeast"
- "Compare the metabolic pathways of glycolysis in prokaryotes and eukaryotes"

#### Using GEMFactory

1. Navigate to the **🧬 GEMFactory** tab
2. Upload a genome file (FASTA or GenBank format)
3. Configure model parameters (organism type, biomass function, etc.)
4. Click "Generate Model" to start automated model construction
5. Download the resulting SBML model or view analysis results

#### Using CASPIA-RAG

1. Navigate to the **🔍 CASPIA-RAG** tab
2. Upload scientific papers (PDF, DOCX, TXT)
3. Wait for documents to be processed and indexed
4. Ask questions about the uploaded documents
5. Receive context-aware answers with source citations

#### Monitoring Tasks

1. Navigate to the **📊 Tasks Monitor** tab
2. View all running and completed tasks
3. Check progress, logs, and resource usage
4. Download results when tasks complete

---

## Modules Description

### CASPIAgent

**Purpose**: Intelligent conversational assistant for synthetic biology research

**Components**:
- `conversation.py`: Multi-turn dialogue management
- `service.py`: LLM inference service integration
- `tools.py`: Tool definitions (database queries, calculations, etc.)
- `utils.py`: Helper functions

**Supported LLMs**:
- OpenAI GPT-4/GPT-3.5
- Alibaba Qwen
- Meta LLaMA
- Custom vLLM deployments

**Key Capabilities**:
- Natural language understanding of biological concepts
- Tool-augmented reasoning (database queries, calculations)
- Multi-step problem solving
- Context retention across conversation turns

---

### GEMFactory

**Purpose**: Automated construction and analysis of genome-scale metabolic models

**Workflow**:
1. Genome annotation (if needed)
2. Draft model reconstruction
3. Gap-filling and curation
4. Biomass function definition
5. Model validation
6. Flux analysis and optimization

**Supported Formats**:
- Input: FASTA, GenBank, SBML
- Output: SBML, JSON, MAT

**Integration with**:
- COBRApy
- CarveMe
- ModelSEED
- Reframed

---

### CASPIA-RAG

**Purpose**: Retrieval-Augmented Generation for scientific literature understanding

**Pipeline**:
1. **Document Parsing**: Extract text and images from PDFs/DOCX
2. **Chunking**: Split documents into semantic units
3. **Embedding**: Generate vector embeddings
4. **Indexing**: Store in ChromaDB vector database
5. **Retrieval**: Semantic search for relevant passages
6. **Generation**: LLM-based answer synthesis with citations

**Features**:
- Multi-modal document understanding
- Automatic image captioning
- Cross-lingual support (translation)
- Citation tracking

---

### Tasks Monitor

**Purpose**: Centralized task management and monitoring

**Features**:
- Task queue with priority scheduling
- Real-time progress tracking
- Resource usage monitoring (CPU, GPU, memory)
- Error logging and recovery
- Result aggregation and download

---

## Examples

### Example 1: Metabolic Model Construction

```python
# Example script for programmatic access (advanced users)
from src.GEMFactory.script.build_model import build_gem

# Build a GEM from genome sequence
model = build_gem(
    genome_file="data/ecoli_k12.fasta",
    organism_name="Escherichia coli K-12",
    gram="negative",
    output_format="sbml"
)

# Perform flux balance analysis
from cobra.flux_analysis import flux_variability_analysis

fva_result = flux_variability_analysis(model)
print(fva_result)
```

### Example 2: RAG-based Literature Query

```python
from src.CASPIA_RAG.agent import RAGAgent

# Initialize RAG agent
agent = RAGAgent(db_path="./src/CASPIA_RAG/db")

# Index documents
agent.index_documents(["paper1.pdf", "paper2.pdf"])

# Query
response = agent.query(
    "What are the latest advances in CRISPR base editing?",
    top_k=5
)
print(response)
```

### Example 3: Conversational Agent

```python
from src.CASPIAgent.service import CASPIAgentService

# Initialize agent
agent = CASPIAgentService(model="gpt-4")

# Interactive conversation
response = agent.chat("How can I optimize the production of lycopene in E. coli?")
print(response)
```

---

## Project Structure

```
SJTU-software-CASPIA/
│
├── webui.py                 # Main application entry point
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── LICENSE                 # License information
│
├── src/                    # Source code modules
│   ├── CASPIAgent/        # Conversational AI agent
│   │   ├── conversation.py
│   │   ├── service.py
│   │   ├── tools.py
│   │   └── utils.py
│   │
│   ├── GEMFactory/        # Metabolic model construction
│   │   ├── data/
│   │   ├── script/
│   │   └── src/
│   │
│   ├── CASPIA_RAG/        # Retrieval-Augmented Generation
│   │   ├── agent.py
│   │   ├── bochaAI.py
│   │   ├── db/
│   │   ├── document/
│   │   ├── image_captioning.py
│   │   ├── load_split_store.py
│   │   ├── prompt.py
│   │   ├── translate.py
│   │   └── util.py
│   │
│   ├── CASPred/           # Prediction modules (if applicable)
│   │
│   └── utils/             # Shared utilities
│
├── tabs/                  # Gradio UI tab definitions
│   ├── agent_tab.py
│   ├── gemfactory_tab.py
│   ├── rag_tab.py
│   └── tasks_monitor_tab.py
│
├── static/                # Static assets (images, CSS, etc.)
│   └── logo.png
│
├── uploads/               # User uploaded files
├── logs/                  # Application logs
│
└── tests/                # Unit and integration tests (to be added)
```

---

## Contributing

We welcome contributions from the community! Whether you're fixing bugs, adding new features, or improving documentation, your help is appreciated.

### How to Contribute

1. **Fork the Repository**
   ```bash
   git clone https://github.com/your-username/SJTU-software-CASPIA.git
   ```

2. **Create a Feature Branch**
   ```bash
   git checkout -b feature/your-feature-name
   ```

3. **Make Your Changes**
   - Follow PEP 8 style guidelines for Python code
   - Add docstrings to all functions and classes
   - Include type hints where appropriate
   - Write unit tests for new features

4. **Run Linting and Tests**
   ```bash
   # Format code
   black src/ tabs/
   
   # Run linter
   ruff check src/ tabs/
   
   # Run tests (when available)
   pytest tests/
   ```

5. **Commit Your Changes**
   ```bash
   git add .
   git commit -m "Add feature: description of your changes"
   ```

6. **Push to Your Fork**
   ```bash
   git push origin feature/your-feature-name
   ```

7. **Open a Pull Request**
   - Go to the original repository on GitHub
   - Click "New Pull Request"
   - Provide a clear description of your changes
   - Reference any related issues

### Contribution Guidelines

- **Code Quality**: Maintain high code quality with proper documentation and testing
- **Modularity**: Keep components modular and reusable
- **Performance**: Consider computational efficiency, especially for AI/ML operations
- **Security**: Never commit API keys or sensitive credentials
- **Documentation**: Update documentation for any user-facing changes

### Reporting Issues

If you encounter bugs or have feature requests:
1. Check existing issues to avoid duplicates
2. Use our issue templates
3. Provide detailed descriptions and reproduction steps
4. Include system information (OS, Python version, GPU, etc.)

---

## Testing

### Running Tests

```bash
# Install testing dependencies
pip install pytest pytest-cov

# Run all tests
pytest tests/

# Run with coverage report
pytest --cov=src tests/

# Run specific test modules
pytest tests/test_caspiagent.py
```

### Test Structure

```
tests/
├── test_caspiagent.py       # CASPIAgent unit tests
├── test_gemfactory.py       # GEMFactory unit tests
├── test_rag.py              # CASPIA-RAG unit tests
└── integration/             # Integration tests
    └── test_workflow.py
```

---

## Citation

If you use CASPIA in your research, please cite:

```bibtex
@software{caspia2025,
  author = {{Team SJTU-Software}},
  title = {CASPIA: Comprehensive AI System for Protein and Integrated Analysis},
  year = {2025},
  publisher = {iGEM},
  url = {https://github.com/shenmaa233/SJTU-software-CASPIA},
  note = {iGEM 2025 Software Tool}
}
```

---

## Authors and Acknowledgments

### Development Team

**Team SJTU-Software 2025**
- Principal Investigators: [To be updated]
- Lead Developers: [To be updated]
- Contributors: See [CONTRIBUTORS.md](CONTRIBUTORS.md)

### Acknowledgments

We would like to express our gratitude to:

- **iGEM Foundation** for organizing the International Genetically Engineered Machine competition
- **Shanghai Jiao Tong University** for institutional support
- **Open Source Community** for the amazing tools and libraries that made this project possible:
  - [Gradio](https://gradio.app/) for the web interface framework
  - [Hugging Face](https://huggingface.co/) for transformer models and hosting
  - [LangChain](https://www.langchain.com/) for LLM orchestration
  - [COBRApy](https://opencobra.github.io/cobrapy/) for metabolic modeling
  - [ChromaDB](https://www.trychroma.com/) for vector database capabilities
  - [PyTorch](https://pytorch.org/) for deep learning infrastructure

### Special Thanks

- Our mentors and advisors for their guidance
- Beta testers and early users for valuable feedback
- All contributors who helped improve CASPIA

---

## License

This project is licensed under the **MIT License** - see the [LICENSE](LICENSE) file for details.

### Third-Party Licenses

This project uses various open-source libraries, each with their own licenses. See [LICENSES_THIRD_PARTY.md](LICENSES_THIRD_PARTY.md) for details.

---

## Contact

- **Team Email**: sjtu.software@gmail.com (to be updated)
- **GitHub Issues**: [Report bugs or request features](https://github.com/shenmaa233/SJTU-software-CASPIA/issues)
- **iGEM Wiki**: [Visit our team wiki](https://2025.igem.org/teams) (to be updated)
- **Twitter/X**: [@SJTU_Software](https://twitter.com/SJTU_Software) (to be updated)

---

## Project Status

**Current Version**: 1.0.0-beta  
**Development Status**: Active Development  
**Last Updated**: October 2025

### Roadmap

- [x] Core platform architecture
- [x] CASPIAgent module
- [x] GEMFactory module
- [x] CASPIA-RAG module
- [x] Tasks Monitor module
- [ ] Comprehensive unit tests
- [ ] API documentation
- [ ] Docker containerization
- [ ] Cloud deployment support
- [ ] Multi-language UI support

---

<p align="center">
  <b>Built with ❤️ by Team SJTU-Software for iGEM 2025</b>
</p>

<p align="center">
  <a href="https://2025.igem.org/">iGEM 2025</a> •
  <a href="https://github.com/shenmaa233/SJTU-software-CASPIA">GitHub</a> •
  <a href="https://github.com/shenmaa233/SJTU-software-CASPIA/wiki">Wiki</a> •
  <a href="https://github.com/shenmaa233/SJTU-software-CASPIA/issues">Issues</a>
</p>

