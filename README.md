# Multi-Scan Highlight Removal with User Mask

A Python tool that removes specular reflections (highlights) from scanned documents and artwork by registering multiple scans and blending them using a user-defined mask with automatic exposure correction.

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Input Folder
Create an `input/` folder with:
- `straightened.tif` - Your reference/primary scan **(REQUIRED)**
- `mask.png` - Grayscale PNG marking problem areas **(REQUIRED)**
- Additional `.tif` files - Extra scans taken at different angles (2+ recommended)

Example structure:
```
input/
├── straightened.tif    # Reference image
├── scan_angle_1.tif    # Additional scans
├── scan_angle_2.tif
├── scan_angle_3.tif
└── mask.png            # User-created mask
```

### 3. Create Your Mask
In Photoshop or similar:
1. Open your reference image at the same dimensions
2. Create a new grayscale image
3. Paint **white** over all highlight/reflection areas you want fixed
4. Paint **white** over shadow areas you want to remove
5. Keep everything else **black**
6. Save as `mask.png` in the `input/` folder

### 4. Run the Script
```bash
python remove_reflections.py
```

## Output Structure

```
output/
├── 01_registered/                        # Aligned scans
│   ├── registered_00_straightened.tif
│   ├── registered_01_scan_angle_1.tif
│   ├── registered_02_scan_angle_2.tif
│   └── ...
├── FINAL_masked_exposure_blended.tif     # MAIN OUTPUT
└── mask_used.tif                         # Reference mask used
```

## How It Works

### Step 1: Image Registration
- Loads your reference image (`straightened.tif`)
- Registers all other scans to match the reference
- Uses ORB feature detection and homography alignment
- Creates perfectly aligned image stack

### Step 2: Mask-Based Replacement
- Loads your `mask.png` (white = problem areas)
- Extracts the **darkest pixels** from the entire image stack at masked locations
- This removes highlights while preserving original colors
- Feathers mask edges for smooth blending

### Step 3: Automatic Exposure Blending
- Analyzes the border between corrected and original areas
- Measures brightness difference using Lab color space
- Calculates exposure adjustment factor automatically
- Applies adjustment only to corrected regions for seamless integration

### Result
A final image with:
- ✓ Highlights removed from masked areas
- ✓ Original detail preserved in non-masked areas
- ✓ Automatic color matching to surrounding pixels
- ✓ Photoshop-compatible RGB TIFF format

## Mask Best Practices

### Marking Highlights
- Paint white over shiny spots, glare, and reflections
- Include the full extent of each highlight
- Extend 5-10 pixels beyond visible highlight edge
- Soft brush edges = smoother blending

### Marking Shadows
- Paint white over dark shadow areas you want to lighten
- Don't over-mask - be selective
- Larger white areas = more aggressive correction

### Example Masks
- **BAD:** Tiny precise mask → Jagged edges, visible blending line
- **GOOD:** Generous soft mask → Smooth, natural-looking result

## Configuration

Edit the top of `remove_reflections.py` to customize:

```python
INPUT_FOLDER = "input"                   # Input folder name
OUTPUT_FOLDER = "output"                 # Output folder name
MAX_FEATURES = 5000                      # Feature detection (higher = more accurate)
GOOD_MATCH_PERCENT = 0.15                # Feature matching strictness
REFERENCE_IMAGE_NAME = "straightened.tif" # Reference image filename
USER_MASK_NAME = "mask.png"              # Mask filename
```

## Tips for Best Results

### Creating a Good Reference Image
- Scan multiple times at slightly different angles
- Choose the best - use the scan with fewest/lightest highlights
- Pre-straighten the image before using as reference
- High resolution (300+ DPI) provides better detail

### Multi-Scan Strategy
- Take 5-7 scans at different angles (rotate 15-30° between scans)
- Vary lighting angle if possible (light from different directions)
- Ensure scans are of the same subject (don't move/tilt document)
- Some scans can have different exposures (script handles this)

### Mask Creation Tips
- Start with a generous mask - it's safe to over-mark
- Use soft brush (20-50% hardness) in Photoshop
- Feather mask edges further for difficult transitions
- Save as grayscale PNG for best compatibility

## Understanding Results

| Issue | Solution |
|-------|----------|
| Corrected areas too dark | Reduce mask coverage or try fewer scans |
| Corrected areas too bright | Use scans with darker exposures |
| Edges look blotchy | Soften mask edges more in Photoshop |
| Alignment is poor | Ensure reference image is straight |

## Troubleshooting

### Error: "Reference image not found"
- Ensure `straightened.tif` exists in `input/` folder
- Check spelling - must be exact
- File must be a TIFF (`.tif` or `.tiff`)

### Error: "Could not load mask"
- Ensure `mask.png` exists in `input/` folder
- File must be PNG format
- Image dimensions should match or be close to scans
- Can be grayscale or RGB (script converts to grayscale)

### Poor Registration Results
- Reference image may be rotated - pre-straighten it
- Additional scans may be too different - use similar angles
- Try with fewer scans first (3-4) to isolate the problem

### Corrected Areas Don't Blend
- Mask edges need to be softer - feather in Photoshop
- Too few scans with good pixel coverage
- Try different `GOOD_MATCH_PERCENT` (increase to 0.2 for stricter matching)

### Highlights Not Fully Removed
- Mask may be too small - extend white areas
- May need more scans with different lighting angles
- Try additional manual passes with refined mask

## Output Files Explained

### `FINAL_masked_exposure_blended.tif`
The main result. Contains:
- Original reference image for non-masked areas
- Darkest pixels from image stack for masked areas
- Automatic exposure correction for seamless blending
- RGB TIFF, fully Photoshop-compatible
- Ready for print or further editing

### `01_registered/` folder
All individually registered scans:
- Can be loaded as layers in Photoshop
- Useful for manual blending if needed
- Allows verification of alignment quality
- Each file aligned to reference geometry

### `mask_used.tif`
The mask as applied to your images:
- Resized to match image dimensions if needed
- Saved as RGB TIFF for consistency
- Useful for verification and documentation

## Using Results in Photoshop

### Direct Use
Simply open `FINAL_masked_exposure_blended.tif` - it's ready to use.

### Fine-Tuning
1. Open both `FINAL_masked_exposure_blended.tif` and `01_registered/registered_00_straightened.tif`
2. Create a layer with the final result
3. Add layer mask using `mask_used.tif`
4. Use Curves/Levels to fine-tune exposure
5. Blend with original using layer opacity

## System Requirements

- Python 3.6+
- 2GB RAM minimum for 6000×4000 images
- Processing time: 2-10 minutes depending on image size and scan count

## Performance

- Script processes full-resolution images
- Registration is the slowest step (most computation here)
- Larger images and more scans = longer processing
- Progress displayed in console output

## Advanced Features

### Adjusting Feathering
In the code, change `feather_size` parameter:

```python
create_user_masked_composite_with_exposure_blend(
    registered_images,
    user_mask,
    reference_idx=0,
    feather_size=5  # Increase for softer edges (0-15 range)
)
```

### Using SIFT for Better Alignment
For difficult alignments, try SIFT instead of ORB:

1. Install: `pip install opencv-contrib-python`
2. Then modify the `register_image` function to use SIFT detector

## Dependencies

See `requirements.txt`:
```
opencv-python>=4.8.0
numpy>=1.24.0
scikit-image>=0.21.0
Pillow>=10.0.0
scipy>=1.10.0
```

## License & Credits

Free to use and modify. Built with OpenCV, NumPy, Pillow, and SciPy.

**Version:** 1.0  
**Last Updated:** January 2026  
**Python:** 3.6+
