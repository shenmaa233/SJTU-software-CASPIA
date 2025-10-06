# Logos Directory

This directory is for storing institutional and sponsor logos to be displayed in the acknowledgments section.

## Suggested Logo Files

Add logos for:

1. **sjtu_logo.png** - Shanghai Jiao Tong University logo
2. **igem_logo.png** - iGEM Foundation logo
3. **sponsor_*.png** - Any sponsor or partner organization logos

## File Specifications

- **Format**: PNG (with transparent background preferred) or SVG
- **Resolution**: 
  - Width: 200-400px for standard display
  - Height: 80-150px for standard display
  - High-DPI versions (2x) recommended
- **File Size**: < 200KB per logo
- **Background**: Transparent PNG recommended for better integration

## Logo Usage Guidelines

### For displaying in the Home page:

```python
gr.Markdown(
    """
    <div style="display: flex; justify-content: space-around; align-items: center; flex-wrap: wrap; margin: 20px 0;">
        <img src="file/static/logos/sjtu_logo.png" alt="SJTU" style="height: 80px; margin: 10px;">
        <img src="file/static/logos/igem_logo.png" alt="iGEM" style="height: 80px; margin: 10px;">
    </div>
    """
)
```

### Important Notes:

1. **Attribution**: Ensure you have permission to use all logos
2. **Trademark Compliance**: Respect trademark and branding guidelines
3. **Color Modes**: Consider providing both light and dark versions if needed
4. **Aspect Ratio**: Maintain original aspect ratios when resizing

## Optimization

Use these tools to optimize logo files:
- **PNG Optimization**: TinyPNG (https://tinypng.com/)
- **SVG Optimization**: SVGOMG (https://jakearchibald.github.io/svgomg/)
- **Background Removal**: Remove.bg (https://www.remove.bg/)

## After Adding Logos

1. Update the acknowledgments section in `tabs/index_tab.py`
2. Test the display in the web interface
3. Ensure logos are properly aligned and sized

---

**Current Status**: Waiting for logo files to be added

