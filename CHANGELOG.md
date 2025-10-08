# CASPIA Version Update Timeline

> **CASPIA (Cell Automated Synthetic Pathway Intelligent Architecture)**  

---

## 📋 Version History Overview

This document records the complete development history of the CASPIA platform from its initial construction to the current version, showcasing the project's evolution and milestones.

---

## 🚀 Latest Version (v1.0.0-beta)

### [5147b72] - Latest Feature Update
**✨ feat: always use model ID to download BiGG model**
- Optimized BiGG model download mechanism
- Always use model ID for downloads to improve stability and consistency
- Improved model retrieval process in the GEMFactory module

### [76190db] - Documentation Update
**📝 Update README.md**
- Updated project README documentation
- Improved project description and user guide

### [490daca] - Documentation Fix
**🐛 fix readme.md**
- Fixed issues in the README documentation
- Optimized documentation format and content

---

## 📊 v0.9.0 - Main Interface & RAG Enhancement

### [d3ede87] - Index Page & Documentation Update
**✨ feat: add index tab & update README file**
- Added homepage tab to optimize user navigation and project introduction
- Updated README file with more detailed project information
- Improved overall user experience

### [3f379d4] - Agent Analysis Tools Enhancement
**✨ feat: add analysis tools in agent**
- Added new analysis tools to CASPIAgent
- Enhanced agent functionality and usability
- Expanded toolset to support more biological analysis scenarios

### [047b11d] - Metabolic Analysis Features
**✨ feat: add FBA/pFBA/FVA analysis and KO/OE analysis**
- Implemented Flux Balance Analysis (FBA)
- Implemented Parsimonious FBA (pFBA)
- Implemented Flux Variability Analysis (FVA)
- Added gene Knockout (KO) and Overexpression (OE) analysis features
- Provided a complete toolchain for metabolic model analysis

### [d1c8aae] - RAG System Completion
**✨ feat: complete the CASPIA-RAG**
- Completed the CASPIA-RAG Retrieval-Augmented Generation system
- Implemented document indexing and semantic search
- Integrated vector database support
- Provided intelligent literature Q&A functionality

---

## 🔧 v0.8.0 - GEMFactory Optimization & Agent Integration for ecGEM/ etcGEM

### [6e7b18b] - GEMFactory Update & Bug Fixes
**🐛 fix: update GEMFactory & solved some bugs in it**
- Updated GEMFactory module
- Fixed several known bugs
- Improved module stability and performance

### [05b316c] - Enzyme-Constrained Model Tools
**✨ feat: add ecGEM and etcGEM agent tools**
- Added agent tools for enzyme-constrained metabolic models (ecGEM)
- Added agent tools for thermodynamic-constrained metabolic models (etcGEM)
- Enhanced diversity in metabolic model construction

### [f35f371] - GEMFactory Refactor
**✨ feat: refactor GEMFactory tab**
- Refactored GEMFactory tab
- Optimized user interface layout
- Improved workflow and interaction logic

---

## 🤖 v0.7.0 - CASPIAgent Refactor

### [bca04bc] - Agent Architecture Refactor
**✨ feat: CASPIAgent refactor**
- Refactored CASPIAgent module
- Optimized conversation management mechanism
- Improved tool invocation architecture
- Enhanced agent response quality
- Removed WebSocket architecture
- Added Task Monitor asynchronous processing architecture

### [9309358, 07fc71f] - Community Contribution Merge
**🔀 Merge pull request #1 from chengjilai/main**
- Merged community contributions
- **✨ feat: add genome file upload and persistent dropdown selection to gemfactory tab**
  - Added genome file upload feature
  - Implemented persistent dropdown selection
  - Improved user experience in GEMFactory tab

---

## 🌐 v0.6.0 - WebSocket & Frontend-Backend Refactor

### [9cc40b6] - Architecture Upgrade
**✨ feat: add WebSocket support and agent service with frontend-backend refactoring**
- Added WebSocket support for real-time communication
- Refactored agent service architecture
- Optimized frontend-backend separation
- Improved system response speed and concurrency

---

## 🧬 v0.5.0 - ecGEM Feature Expansion

### [7ee8d06] - Kcat Prediction Integration
**✨ feat: add ensemble in kcat predict & refactor the parameter utils**
- Added ensemble methods in Kcat prediction
- Refactored parameter utility module
- Improved accuracy of enzyme kinetic parameter prediction

### [9d898c5] - Optimal Temperature Prediction
**✨ add topt prediction**
- Added optimal temperature (Topt) prediction feature
- Supported modeling for temperature-constrained metabolic models

### [9c6127e] - ecGEM Running Feature
**✨ feat: add run_ecGEM**
- Added ecGEM model running feature
- Enabled simulation and analysis of enzyme-constrained metabolic models

### [2d0bb54] - Cache Cleanup
**🐛 fix: delete some cache files**
- Deleted unnecessary cache files
- Optimized project structure and storage

---

## 🏗️ v0.4.0 - ecGEM Core Features

### [0a2feae] - ecGEM Module Addition
**✨ feat: add ecGEM & refactor code and import**
- Added core features for enzyme-constrained metabolic models (ecGEM)
- Refactored code structure
- Optimized import mechanism

### [85efbd6] - Source Code Refactor
**✨ feat: refactor the source code structure**
- Refactored overall code structure
- Optimized module organization
- Improved code maintainability

---

## 🔬 v0.3.0 - CASPred Integration

### [ab10d5f] - Prediction Module Update
**✨ feat: update CASPred**
- Updated CASPred prediction module
- Enhanced enzyme parameter prediction capability

---

## 🎯 v0.2.0 - GEMFactory Basic Features

### [f757baf] - Project Refactor & GEMFactory
**✨ feat: reconstruct the project category & add the primary functions of the GEMFactory**
- Refactored project directory structure
- Added primary functions of GEMFactory
- Implemented basic workflow for metabolic model construction

---

## 🌱 v0.1.0 - Project Initialization

### [54e6651] - Add Git Configuration
**✨ feat: add .gitignore**
- Added .gitignore file
- Configured version control rules

### [9414963] - Project Inception
**🎉 Init commit**
- Initialized CASPIA project
- Established basic project framework
- Started the software development journey of Team SJTU-Software 2025