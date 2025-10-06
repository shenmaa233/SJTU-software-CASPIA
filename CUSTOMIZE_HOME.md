# Customizing the Home Page (Index Tab)

This guide explains how to customize the Home page of CASPIA with your demo GIFs and acknowledgments.

## 📁 File Location

The home page is defined in: `tabs/index_tab.py`

## 🎬 Adding Demo GIFs

### Step 1: Prepare Your Demo Files

Create a `demos` folder in the `static` directory:

```bash
mkdir -p static/demos
```

### Step 2: Add Your GIF/Video Files

Place your demo files in the `static/demos` folder:

```
static/demos/
├── caspiagent_demo.gif
├── gemfactory_demo.gif
├── rag_demo.gif
└── monitor_demo.gif
```

**Recommended specifications:**
- Format: GIF or MP4
- Resolution: 800x600 or 1280x720
- File size: < 10MB for optimal loading
- Duration: 10-30 seconds

### Step 3: Update the Code

Edit `tabs/index_tab.py`, find the placeholder sections and update them:

**For CASPIAgent Demo (around line 120):**

```python
demo_agent_placeholder = gr.Image(
    value="static/demos/caspiagent_demo.gif",  # Update this line
    label="CASPIAgent in Action",
    show_label=True,
    interactive=False,
    show_download_button=False,
    height=300,
    container=True
)
```

**For GEMFactory Demo (around line 135):**

```python
demo_gem_placeholder = gr.Image(
    value="static/demos/gemfactory_demo.gif",  # Update this line
    label="GEMFactory Workflow",
    show_label=True,
    interactive=False,
    show_download_button=False,
    height=300,
    container=True
)
```

**For CASPIA-RAG Demo (around line 150):**

```python
demo_rag_placeholder = gr.Image(
    value="static/demos/rag_demo.gif",  # Update this line
    label="CASPIA-RAG Document Analysis",
    show_label=True,
    interactive=False,
    show_download_button=False,
    height=300,
    container=True
)
```

**For Tasks Monitor Demo (around line 165):**

```python
demo_monitor_placeholder = gr.Image(
    value="static/demos/monitor_demo.gif",  # Update this line
    label="Tasks Monitor Dashboard",
    show_label=True,
    interactive=False,
    show_download_button=False,
    height=300,
    container=True
)
```

### Step 4: Update Demo Descriptions (Optional)

You can also update the markdown descriptions below each demo image to provide more context.

---

## 🙏 Updating Acknowledgments

### Competition and Institutional Support

Find this section in `tabs/index_tab.py` (around line 200):

```python
with gr.Accordion("🏆 Competition and Institutional Support", open=True):
    gr.Markdown(
        """
        ### iGEM Foundation
        
        We are grateful to the [International Genetically Engineered Machine (iGEM) Foundation](https://igem.org/) 
        for organizing this incredible competition and fostering innovation in synthetic biology worldwide.
        
        ### Shanghai Jiao Tong University
        
        Special thanks to Shanghai Jiao Tong University for providing institutional support, resources, 
        and guidance throughout the development of CASPIA.
        
        **Add your additional acknowledgments here:**
        
        - Funding organizations
        - Laboratory facilities
        - Research centers
        - Corporate sponsors
        
        ---
        
        *You can also add institutional logos as images*
        """
    )
```

### Team and Contributors

Find this section (around line 240):

```python
with gr.Accordion("👥 Team and Contributors", open=False):
    gr.Markdown(
        """
        ### Team SJTU-Software 2025
        
        **Replace the placeholders with your team information:**
        
        - **Principal Investigators**: 
          - Dr. [Name], [Affiliation]
          - Prof. [Name], [Affiliation]
        
        - **Lead Developers**: 
          - [Name] - Project Lead
          - [Name] - Backend Development
          - [Name] - Frontend Development
        
        - **Bioinformatics Team**: 
          - [Name] - GEMFactory Module
          - [Name] - CASPred Module
        
        - **AI/ML Team**: 
          - [Name] - CASPIAgent Module
          - [Name] - RAG System
        
        - **UI/UX Design**: 
          - [Name] - Interface Design
        
        - **Documentation**: 
          - [Name] - Technical Writing
        
        ### Mentors and Advisors
        
        - **[Mentor Name]**, [Title], [Institution]
          - Contribution: [Brief description]
        
        - **[Mentor Name]**, [Title], [Institution]
          - Contribution: [Brief description]
        
        ### Special Thanks
        
        - Beta testers: [Names]
        - Research collaborators: [Names/Institutions]
        - Technical advisors: [Names]
        - Anyone else who contributed to the project
        """
    )
```

---

## 🖼️ Adding Institutional Logos

### Step 1: Prepare Logo Files

Add logos to `static/logos/` directory:

```bash
mkdir -p static/logos
```

Place your logo files:
```
static/logos/
├── sjtu_logo.png
├── igem_logo.png
├── sponsor1_logo.png
└── sponsor2_logo.png
```

### Step 2: Add Logos to the Page

You can add logos anywhere in the markdown sections. Example:

```python
gr.Markdown(
    """
    ### Our Supporters
    
    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin: 20px 0;">
        <img src="file/static/logos/sjtu_logo.png" alt="SJTU" style="height: 80px; margin: 10px;">
        <img src="file/static/logos/igem_logo.png" alt="iGEM" style="height: 80px; margin: 10px;">
        <img src="file/static/logos/sponsor1_logo.png" alt="Sponsor 1" style="height: 80px; margin: 10px;">
    </div>
    """
)
```

**Note**: In Gradio, you need to use `file/` prefix for local static files.

---

## 📝 Quick Customization Checklist

- [ ] Record demo GIFs for all four modules
- [ ] Place GIF files in `static/demos/` directory
- [ ] Update image paths in `tabs/index_tab.py`
- [ ] List all team members with their roles
- [ ] Add mentor names and affiliations
- [ ] Include special acknowledgments
- [ ] Add institutional logos (optional)
- [ ] Update team contact information
- [ ] Review and test the home page
- [ ] Update descriptions and captions as needed

---

## 🧪 Testing Your Changes

After making changes, restart the web server:

```bash
python webui.py
```

Navigate to the **🏠 Home** tab to see your updates.

---

## 💡 Tips

1. **GIF Optimization**: Use tools like [ezgif.com](https://ezgif.com/) to optimize GIF file sizes
2. **Video Alternative**: If GIFs are too large, consider using HTML5 video:
   ```python
   gr.Video(value="static/demos/demo.mp4", label="Demo Video")
   ```
3. **Responsive Images**: Keep images reasonably sized for different screen resolutions
4. **Attribution**: Make sure to properly credit all contributors and supporters
5. **Regular Updates**: Keep the acknowledgments section up-to-date as new contributors join

---

## 📧 Need Help?

If you encounter any issues while customizing:
1. Check the Gradio documentation: https://gradio.app/docs/
2. Review the file `tabs/index_tab.py` for the structure
3. Test changes incrementally to identify issues quickly

---

**Last Updated**: October 2025

