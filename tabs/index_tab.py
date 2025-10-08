"""
Index Tab - Welcome page for CASPIA platform
Provides project overview, demo videos, and acknowledgments
"""

import gradio as gr


def index_tab():
    """
    Create the index/welcome tab with project introduction,
    demo GIFs, and acknowledgments
    """
    with gr.Column():
        # Project Introduction Section
        gr.Markdown(
            """
            # 🎉 Welcome to CASPIA
            
            ## Cell Automated Synthetic Pathway Intelligent Architecture
            
            **CASPIA** is an advanced, modular AI-powered platform designed to revolutionize synthetic biology research. 
            Developed by Team SJTU-Software for iGEM 2025, CASPIA integrates cutting-edge artificial intelligence 
            technologies with synthetic biology workflows to accelerate scientific discovery.
            
            ---
            
            ## 🌟 What Can CASPIA Do?
            
            CASPIA provides four powerful modules to support your synthetic biology research:
            """
        )
        
        # Features Grid
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### 🤖 CASPIAgent
                    
                    **Intelligent Conversational Assistant**
                    
                    - Natural language interface for complex biological queries
                    - Tool-augmented reasoning with database integration
                    - Multi-turn dialogue with context awareness
                    - Powered by state-of-the-art LLMs (GPT-4, LLaMA, Qwen)
                    
                    **Use Cases:**
                    - Literature review and knowledge synthesis
                    - Experimental design assistance
                    - Biological concept explanation
                    - Hypothesis generation and validation
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### 🧬 GEMFactory
                    
                    **Automated Metabolic Model Construction**
                    
                    - Genome-scale metabolic model (GEM) generation
                    - Flux balance analysis (FBA) and optimization
                    - Integration with COBRApy, CarveMe, and ModelSEED
                    - Model validation and quality assessment
                    
                    **Use Cases:**
                    - Metabolic engineering design
                    - Strain optimization
                    - Pathway analysis and prediction
                    - Biomass production simulation
                    """
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### 🔍 CASPIA-RAG
                    
                    **Knowledge Retrieval and Understanding**
                    
                    - Retrieval-Augmented Generation for scientific literature
                    - PDF/DOCX document parsing and indexing
                    - Multi-modal understanding (text + images)
                    - Semantic search with citation tracking
                    
                    **Use Cases:**
                    - Automated literature review
                    - Document-based question answering
                    - Knowledge extraction from papers
                    - Cross-document information synthesis
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown(
                    """
                    ### 📊 Tasks Monitor
                    
                    **Workflow Management and Monitoring**
                    
                    - Real-time task tracking and visualization
                    - Job queue management for long computations
                    - Resource usage monitoring (CPU, GPU, memory)
                    - Results aggregation and export
                    
                    **Use Cases:**
                    - Batch job management
                    - Computational workflow orchestration
                    - Performance monitoring
                    - Result tracking and organization
                    """
                )
        
        gr.Markdown("---")
        
        # Demo Videos Section
        gr.Markdown(
            """
            ## 🎬 Platform Demonstrations
            
            Explore how CASPIA can enhance your research workflow through these interactive demonstrations.
            """
        )
        
        # Placeholder for demo GIFs - organized in a grid
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### CASPIAgent Demo")
                # Placeholder for GIF
                demo_agent_placeholder = gr.Image(
                    value=None,
                    label="CASPIAgent in Action",
                    show_label=True,
                    interactive=False,
                    show_download_button=False,
                    height=300,
                    container=True
                )
                gr.Markdown(
                    """
                    *Demonstration of conversational AI assistant answering complex biological questions*
                    
                    **Demo Coming Soon!**
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### GEMFactory Demo")
                # Placeholder for GIF
                demo_gem_placeholder = gr.Image(
                    value=None,
                    label="GEMFactory Workflow",
                    show_label=True,
                    interactive=False,
                    show_download_button=False,
                    height=300,
                    container=True
                )
                gr.Markdown(
                    """
                    *Automated metabolic model construction from genome sequence to analysis*
                    
                    **Demo Coming Soon!**
                    """
                )
        
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### CASPIA-RAG Demo")
                # Placeholder for GIF
                demo_rag_placeholder = gr.Image(
                    value=None,
                    label="CASPIA-RAG Document Analysis",
                    show_label=True,
                    interactive=False,
                    show_download_button=False,
                    height=300,
                    container=True
                )
                gr.Markdown(
                    """
                    *Document-based question answering with intelligent retrieval*
                    
                    **Demo Coming Soon!**
                    """
                )
            
            with gr.Column(scale=1):
                gr.Markdown("### Tasks Monitor Demo")
                # Placeholder for GIF
                demo_monitor_placeholder = gr.Image(
                    value=None,
                    label="Tasks Monitor Dashboard",
                    show_label=True,
                    interactive=False,
                    show_download_button=False,
                    height=300,
                    container=True
                )
                gr.Markdown(
                    """
                    *Real-time monitoring of computational workflows and resource usage*
                    
                    **Demo Coming Soon!**
                    """
                )
        
        gr.Markdown("---")
        
        # Quick Start Guide
        gr.Markdown(
            """
            ## 🚀 Quick Start Guide
            
            ### Getting Started in 3 Steps:
            
            1. **Choose Your Module**: Navigate to the appropriate tab based on your research needs
               - Use **CASPIAgent** for conversational assistance and knowledge queries
               - Use **GEMFactory** for metabolic model construction and analysis
               - Use **CASPIA-RAG** for document-based research and literature mining
               - Use **Tasks Monitor** to track your computational workflows
            
            2. **Prepare Your Input**: 
               - For CASPIAgent: Simply type your question in natural language
               - For GEMFactory: Upload genome files (FASTA/GenBank format)
               - For CASPIA-RAG: Upload your scientific papers (PDF/DOCX)
               - For Tasks Monitor: Check status of running jobs
            
            3. **Get Results**: 
               - Receive AI-powered insights and analysis
               - Download generated models, reports, or visualizations
               - Export results for further analysis
            
            ### Need Help?
            
            - 📖 **Documentation**: Check our [GitHub README](https://github.com/shenmaa233/SJTU-software-CASPIA) for detailed guides
            - 🐛 **Report Issues**: Found a bug? [Open an issue](https://github.com/shenmaa233/SJTU-software-CASPIA/issues)
            - 💬 **Community**: Join our discussions and share your experience
            - 📧 **Contact**: Reach out to our team for support
            """
        )
        
        gr.Markdown("---")
        
        # Acknowledgments Section
        gr.Markdown(
            """
            ## 🙏 Acknowledgments
            
            CASPIA would not be possible without the support and contributions from many individuals and organizations.
            """
        )
        
        # Acknowledgments organized in expandable sections
        with gr.Accordion("🏆 Competition and Institutional Support", open=True):
            gr.Markdown(
                """
                ### iGEM Foundation
                
                We are grateful to the [International Genetically Engineered Machine (iGEM) Foundation](https://igem.org/) 
                for organizing this incredible competition and fostering innovation in synthetic biology worldwide.
                
                ### Shanghai Jiao Tong University
                
                Special thanks to Shanghai Jiao Tong University for providing institutional support, resources, 
                and guidance throughout the development of CASPIA.
                
                **[Additional acknowledgments to be added by the team]**
                
                ---
                
                *Placeholder for institutional logos and additional acknowledgments*
                """
            )
        
        with gr.Accordion("💻 Open Source Technologies", open=False):
            gr.Markdown(
                """
                CASPIA is built on the shoulders of giants. We acknowledge the following open-source projects:
                
                ### Core Technologies
                - **[Gradio](https://gradio.app/)** - Beautiful and intuitive web interface framework
                - **[PyTorch](https://pytorch.org/)** - Deep learning infrastructure powering our AI models
                - **[Hugging Face](https://huggingface.co/)** - Transformers library and model hosting
                - **[LangChain](https://www.langchain.com/)** - LLM orchestration and agent framework
                
                ### Specialized Tools
                - **[COBRApy](https://opencobra.github.io/cobrapy/)** - Constraint-based metabolic modeling
                - **[ChromaDB](https://www.trychroma.com/)** - Vector database for semantic search
                - **[vLLM](https://vllm.ai/)** - High-performance LLM inference engine
                - **[BioPython](https://biopython.org/)** - Computational biology tools
                
                ### Development Tools
                - **[FastAPI](https://fastapi.tiangolo.com/)** - Modern API framework
                - **[NumPy](https://numpy.org/)** & **[SciPy](https://scipy.org/)** - Scientific computing
                - **[Pandas](https://pandas.pydata.org/)** - Data manipulation and analysis
                - **[Matplotlib](https://matplotlib.org/)** & **[Plotly](https://plotly.com/)** - Data visualization
                
                *And many more amazing open-source libraries listed in our requirements.txt*
                """
            )
        
        with gr.Accordion("👥 Team and Contributors", open=False):
            gr.Markdown(
                """
                ### Team SJTU-Software 2025
                
                **[Team member names and roles to be added]**
                
                - **Principal Investigators**: [Names]
                - **Lead Developers**: [Names]
                - **Bioinformatics Team**: [Names]
                - **AI/ML Team**: [Names]
                - **UI/UX Design**: [Names]
                - **Documentation**: [Names]
                
                ### Mentors and Advisors
                
                **[Mentor names and affiliations to be added]**
                
                We are deeply grateful to our mentors for their invaluable guidance, expertise, and support.
                
                ### Special Thanks
                
                **[Additional acknowledgments to be added]**
                
                - Beta testers and early users who provided crucial feedback
                - Research collaborators who contributed domain expertise
                - Everyone who contributed code, documentation, or ideas
                
                ---
                
                ### 🤝 Want to Contribute?
                
                CASPIA is open-source and welcomes contributions! Check out our 
                [Contributing Guidelines](https://github.com/shenmaa233/SJTU-software-CASPIA#contributing) 
                to get started.
                """
            )
        
        gr.Markdown("---")
        
        # Footer with important links
        gr.Markdown(
            """
            ## 🔗 Important Links
            
            <div style="text-align: center; padding: 20px;">
                <a href="https://github.com/shenmaa233/SJTU-software-CASPIA" target="_blank" style="margin: 0 15px; text-decoration: none; font-size: 16px;">
                    📦 GitHub Repository
                </a>
                <a href="https://github.com/shenmaa233/SJTU-software-CASPIA#installation" target="_blank" style="margin: 0 15px; text-decoration: none; font-size: 16px;">
                    📚 Documentation
                </a>
                <a href="https://2025.igem.org/teams" target="_blank" style="margin: 0 15px; text-decoration: none; font-size: 16px;">
                    🌐 Team Wiki
                </a>
                <a href="https://github.com/shenmaa233/SJTU-software-CASPIA/issues" target="_blank" style="margin: 0 15px; text-decoration: none; font-size: 16px;">
                    🐛 Report Issues
                </a>
            </div>
            
            ---
            
            <div style="text-align: center; padding: 10px; color: #666;">
                <p><strong>CASPIA v1.0.0-beta</strong></p>
                <p>Developed with ❤️ by Team SJTU-Software for iGEM 2025</p>
                <p>Licensed under MIT License | © 2025 Team SJTU-Software</p>
            </div>
            """
        )

    return None

