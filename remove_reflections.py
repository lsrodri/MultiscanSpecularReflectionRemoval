#!/usr/bin/env python3
"""
Multi-Scan Specular Reflection Removal with User Mask
------------------------------------------------------
Registers multiple TIFF scans to straightened.tif, then uses a user-provided
mask (mask.png) to selectively replace highlighted areas with darker pixels
from other scans while preserving color.

Usage: python remove_reflections.py

Place your TIFF images, straightened.tif, and mask.png in INPUT_FOLDER.
"""

import cv2
import numpy as np
from pathlib import Path
from PIL import Image

# ==================== CONFIGURATION ====================
INPUT_FOLDER = "input_scans"  # Folder containing your TIFF images
OUTPUT_FOLDER = "output"
MAX_FEATURES = 5000  # Number of features to detect for alignment
GOOD_MATCH_PERCENT = 0.15  # Top % of matches to use
REFERENCE_IMAGE_NAME = "straightened.tif"  # Use this as the reference
USER_MASK_NAME = "mask.png"  # White areas will be fixed

# ==================== HELPER FUNCTIONS ====================

def load_user_mask(folder_path):
    """Load user-provided mask. White areas (255) will be replaced."""
    mask_path = Path(folder_path) / USER_MASK_NAME
    
    if not mask_path.exists():
        print(f"Warning: {USER_MASK_NAME} not found in {folder_path}")
        return None
    
    # Load mask
    mask = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
    if mask is None:
        print(f"Warning: Could not load {USER_MASK_NAME}")
        return None
    
    print(f"Loaded user mask: {mask_path.name} - Shape: {mask.shape}")
    
    # Count white pixels
    white_pixels = np.sum(mask > 128)  # Consider anything > 128 as "white"
    total_pixels = mask.shape[0] * mask.shape[1]
    percent = (white_pixels / total_pixels) * 100
    print(f"  Mask covers {white_pixels:,} pixels ({percent:.2f}% of image)")
    
    return mask


def load_images(folder_path):
    """Load all TIFF images from folder, with straightened.tif as first (reference)"""
    folder = Path(folder_path)
    
    # First, check if straightened.tif exists
    reference_path = folder / REFERENCE_IMAGE_NAME
    if not reference_path.exists():
        raise ValueError(f"Reference image '{REFERENCE_IMAGE_NAME}' not found in {folder_path}")
    
    # Get all TIFF images
    image_paths = sorted(folder.glob("*.tif")) + sorted(folder.glob("*.tiff"))
    
    # Remove reference from list and put it first
    image_paths = [p for p in image_paths if p.name != REFERENCE_IMAGE_NAME]
    image_paths.insert(0, reference_path)
    
    if len(image_paths) < 2:
        raise ValueError(f"Need at least 2 images. Found {len(image_paths)} in {folder_path}")

    images = []
    loaded_filenames = []
    
    for path in image_paths:
        # Try OpenCV first
        img = cv2.imread(str(path), cv2.IMREAD_COLOR)
        
        # If OpenCV fails, try PIL/Pillow as fallback
        if img is None:
            try:
                print(f"OpenCV failed for {path.name}, trying PIL...")
                pil_img = Image.open(path)
                # Convert PIL image to numpy array
                img_array = np.array(pil_img)
                
                # Convert RGB to BGR (OpenCV format)
                if len(img_array.shape) == 3 and img_array.shape[2] == 3:
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR)
                elif len(img_array.shape) == 3 and img_array.shape[2] == 4:
                    # Handle RGBA
                    img = cv2.cvtColor(img_array, cv2.COLOR_RGBA2BGR)
                else:
                    # Grayscale
                    img = cv2.cvtColor(img_array, cv2.COLOR_GRAY2BGR)
                    
                print(f"  Successfully loaded with PIL: {path.name} - Shape: {img.shape}")
            except Exception as e:
                print(f"Warning: Could not load {path} with PIL either: {e}")
                continue
        else:
            print(f"Loaded: {path.name} - Shape: {img.shape}")
            
        images.append(img)
        loaded_filenames.append(path.name)

    if len(images) < 2:
        raise ValueError(f"Need at least 2 images. Successfully loaded {len(images)} images.")

    return images, loaded_filenames


def register_image(img_to_align, reference_img):
    """
    Register img_to_align to match reference_img using feature-based alignment.
    Returns the transformed image and transformation matrix.
    """
    # Convert to grayscale
    img1_gray = cv2.cvtColor(reference_img, cv2.COLOR_BGR2GRAY)
    img2_gray = cv2.cvtColor(img_to_align, cv2.COLOR_BGR2GRAY)

    # Detect ORB features and compute descriptors
    orb = cv2.ORB_create(MAX_FEATURES)
    keypoints1, descriptors1 = orb.detectAndCompute(img1_gray, None)
    keypoints2, descriptors2 = orb.detectAndCompute(img2_gray, None)

    # Match features using BFMatcher
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = matcher.match(descriptors1, descriptors2)

    # Sort matches by score
    matches = sorted(matches, key=lambda x: x.distance)

    # Keep only the best matches
    num_good_matches = int(len(matches) * GOOD_MATCH_PERCENT)
    matches = matches[:num_good_matches]

    print(f"  Found {len(matches)} good matches out of {len(keypoints1)} and {len(keypoints2)} keypoints")

    # Extract location of good matches
    points1 = np.zeros((len(matches), 2), dtype=np.float32)
    points2 = np.zeros((len(matches), 2), dtype=np.float32)

    for i, match in enumerate(matches):
        points1[i, :] = keypoints1[match.queryIdx].pt
        points2[i, :] = keypoints2[match.trainIdx].pt

    # Find homography
    h, mask = cv2.findHomography(points2, points1, cv2.RANSAC)

    # Use homography to warp image
    height, width = reference_img.shape[:2]
    img_aligned = cv2.warpPerspective(img_to_align, h, (width, height))

    return img_aligned, h


def create_user_masked_composite(images, user_mask, reference_idx=0, feather_size=5):
    """
    Use user-provided mask to replace highlighted areas with darker pixels
    from other images while preserving color and detail.
    
    Only processes pixels where mask is white (>128).
    Uses minimum (darkest) values from all images for those pixels.
    """
    print(f"\nCreating user-masked composite...")
    print(f"  Reference image: {reference_idx}")
    
    reference = images[reference_idx].copy()
    h, w = reference.shape[:2]
    
    # Ensure mask matches image size
    if user_mask.shape[0] != h or user_mask.shape[1] != w:
        print(f"  Resizing mask from {user_mask.shape} to {(h, w)}")
        user_mask = cv2.resize(user_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    
    # Create binary mask (white areas = 255, rest = 0)
    binary_mask = (user_mask > 128).astype(np.uint8) * 255
    
    # Optional: feather the mask for smoother blending
    if feather_size > 0:
        feathered_mask = cv2.GaussianBlur(binary_mask, (feather_size, feather_size), 0)
    else:
        feathered_mask = binary_mask
    
    # Convert to float for blending
    feathered_mask_float = feathered_mask.astype(np.float32) / 255.0
    
    # Stack all images
    images_float = [img.astype(np.float32) for img in images]
    stack = np.stack(images_float, axis=0)
    
    # For masked areas, find the minimum (darkest) value across all images
    # This removes highlights while preserving color
    min_composite = np.min(stack, axis=0)
    
    # Blend: use min composite in masked areas, reference everywhere else
    result = (reference.astype(np.float32) * (1 - feathered_mask_float[:, :, np.newaxis]) + 
              min_composite * feathered_mask_float[:, :, np.newaxis])
    
    result_final = np.clip(result, 0, 255).astype(np.uint8)
    
    return result_final


def save_tiff_rgb(image, filepath):
    """
    Save image as RGB TIFF compatible with Photoshop.
    Uses PIL to ensure proper RGB format.
    """
    # Ensure image is uint8
    if image.dtype != np.uint8:
        image = np.clip(image, 0, 255).astype(np.uint8)
    
    # Convert BGR (OpenCV) to RGB (PIL/Photoshop)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Save using PIL for better compatibility
    pil_image = Image.fromarray(image_rgb)
    pil_image.save(str(filepath), format='TIFF', compression='none')


# ==================== MAIN PIPELINE ====================

def main():
    # Create output directory
    output_path = Path(OUTPUT_FOLDER)
    output_path.mkdir(exist_ok=True)

    print("=" * 60)
    print("USER-MASKED HIGHLIGHT REMOVAL")
    print("=" * 60)

    # Step 1: Load user mask
    print("\nStep 1: Loading user mask...")
    user_mask = load_user_mask(INPUT_FOLDER)
    
    if user_mask is None:
        print(f"\nError: Could not load mask. Please ensure '{USER_MASK_NAME}' exists in '{INPUT_FOLDER}'")
        return

    # Step 2: Load images (straightened.tif will be first)
    print("\nStep 2: Loading images...")
    images, filenames = load_images(INPUT_FOLDER)
    print(f"Successfully loaded {len(images)} images")
    print(f"Reference image: {filenames[0]}")

    # Step 3: Register all images to the reference (straightened.tif)
    print("\nStep 3: Registering images to reference...")
    reference_img = images[0]
    registered_images = [reference_img.copy()]

    for i in range(1, len(images)):
        print(f"\nAligning image {i} ({filenames[i]}) to reference...")
        aligned_img, homography = register_image(images[i], reference_img)
        registered_images.append(aligned_img)

    # Save registered images
    print("\nSaving registered images...")
    registered_path = output_path / "01_registered"
    registered_path.mkdir(exist_ok=True)

    for i, (img, name) in enumerate(zip(registered_images, filenames)):
        output_file = registered_path / f"registered_{i:02d}_{name}"
        save_tiff_rgb(img, output_file)
        print(f"  Saved: {output_file.name}")

    # Step 4: Create user-masked composite
    print("\nStep 4: Creating user-masked composite...")
    
    # No feathering for exact mask
    final_result_nofeather = create_user_masked_composite(
        registered_images, 
        user_mask,
        reference_idx=0,
        feather_size=0
    )
    
    final_output_nofeather = output_path / "FINAL_masked_no_feather.tif"
    save_tiff_rgb(final_result_nofeather, final_output_nofeather)
    print(f"\n✓ Saved final result (no feather): {final_output_nofeather}")
    
    # With slight feathering for smoother blend
    final_result_feather = create_user_masked_composite(
        registered_images, 
        user_mask,
        reference_idx=0,
        feather_size=5
    )
    
    final_output_feather = output_path / "FINAL_masked_feathered.tif"
    save_tiff_rgb(final_result_feather, final_output_feather)
    print(f"✓ Saved final result (feathered): {final_output_feather}")
    
    # Save the mask used (resized if needed)
    h, w = reference_img.shape[:2]
    if user_mask.shape[0] != h or user_mask.shape[1] != w:
        user_mask_resized = cv2.resize(user_mask, (w, h), interpolation=cv2.INTER_NEAREST)
    else:
        user_mask_resized = user_mask
    
    mask_output = output_path / "mask_used.tif"
    # Convert grayscale mask to RGB for consistency
    mask_rgb = cv2.cvtColor(user_mask_resized, cv2.COLOR_GRAY2RGB)
    save_tiff_rgb(mask_rgb, mask_output)
    print(f"✓ Saved mask used: {mask_output}")

    print("\n" + "=" * 60)
    print("PROCESSING COMPLETE")
    print("=" * 60)
    print(f"\nOutput files saved to: {output_path.absolute()}")
    print(f"\n*** FINAL RESULTS: ***")
    print(f"  1. {final_output_nofeather.name} (exact mask)")
    print(f"  2. {final_output_feather.name} (smooth blend)")
    print(f"\nAll TIFF files are saved as RGB and should open in Photoshop!")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()