# Specular Reflection Removal for Scanned Images

## Quick Start

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Prepare Your Images
- Create a folder named `input_scans`
- Place all your TIFF scans in this folder
- Name them sequentially (e.g., scan_01.tif, scan_02.tif, ...)

### 3. Run the Script
```bash
python remove_reflections.py
```

## Output Structure

```
output/
├── 01_registered/          # All images aligned to first image
│   ├── registered_00_scan_01.tif
│   ├── registered_01_scan_02.tif
│   └── ...
├── 02_normalized/          # Exposure-normalized images
│   ├── normalized_00_scan_01.tif
│   ├── normalized_01_scan_02.tif
│   └── ...
├── final_result_exposure_fusion.tif    # MAIN OUTPUT - Reflections removed
└── alternative_min_composite.tif       # Alternative method for comparison
```

## What Each Step Does

### Step 1: Image Registration
- Detects ORB features (corners, edges) in each image
- Matches corresponding features between images
- Computes transformation matrix (rotation + translation + perspective)
- Warps images to align perfectly with the reference image

### Step 2: Exposure Normalization
- Compensates for scanner auto-exposure variations
- Matches histogram of each image to the reference
- Ensures consistent brightness across all scans

### Step 3: Exposure Fusion
- Analyzes each pixel across all registered images
- Weights pixels by contrast, saturation, and exposure quality
- Specular highlights get low weights (overexposed, low contrast)
- Properly exposed regions get high weights
- Blends images seamlessly using pyramid blending

## Customization

Edit these parameters at the top of `remove_reflections.py`:

```python
INPUT_FOLDER = "input_scans"  # Change to your folder path
MAX_FEATURES = 5000           # More features = better alignment but slower
GOOD_MATCH_PERCENT = 0.15     # Lower = stricter matching
```

## Using Results in Photoshop

The `01_registered/` folder contains all aligned images that you can:
1. Load as layers in Photoshop (File > Scripts > Load Files into Stack)
2. Manually mask areas if you want more control
3. The images are already aligned, so masks will work across layers

## Troubleshooting

**"Not enough matches found"**
- Images may be too different in rotation/position
- Try increasing `MAX_FEATURES` to 10000
- Ensure images have overlapping content

**"Images look misaligned"**
- Check if images have sufficient texture/features
- Flat uniform areas don't provide good feature points
- Consider using SIFT instead of ORB (more accurate but requires opencv-contrib-python)

**"Final result still has reflections"**
- Try the `alternative_min_composite.tif` output
- May need more scans at different angles
- Ensure rotations actually move the specular highlights

## Advanced: Using SIFT Instead of ORB

For higher accuracy with tricky images, replace ORB with SIFT:

1. Install: `pip install opencv-contrib-python`
2. In the script, replace:
```python
orb = cv2.ORB_create(MAX_FEATURES)
```
with:
```python
orb = cv2.SIFT_create(MAX_FEATURES)
matcher = cv2.BFMatcher(cv2.NORM_L2, crossCheck=True)  # Also change matcher
```
