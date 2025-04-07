#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import glob
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageChops
import argparse
import time

def process_images(input_dir, output_dir, mode="standard", seed=None, **params):
    """
    Process images from input directory with selective focus effect.
    
    Parameters:
    input_dir (str): Directory containing input images
    output_dir (str): Directory for saving processed images
    mode (str): Processing mode ('standard', 'explore', 'sample', 'refine')
    seed (int, optional): Random seed for reproducibility
    **params: Additional parameters for the effect
    
    Returns:
    list: List of output image paths
    """
    # Make sure output directory exists
    os.makedirs(output_dir, exist_ok=True)
    
    # Set random seed
    if seed is None:
        seed = random.randint(1, 999999)
    print(f"Using seed: {seed}")
    random.seed(seed)
    np.random.seed(seed)
    
    # Default parameters
    default_params = {
        "focus_ratio": 0.3,      # 0.1-0.9
        "blur_strength": 0.7,    # 0.1-1.0
        "randomness": 0.5,       # 0.0-1.0
        "ghost_threshold": 0.5   # 0.0-1.0
    }
    
    # Merge with custom parameters if provided
    for key, value in params.items():
        if key in default_params:
            default_params[key] = value
    
    # Find all image files
    image_files = []
    for ext in ['.jpg', '.jpeg', '.png', '.tiff', '.JPG', '.JPEG', '.PNG', '.TIFF']:
        image_files.extend(glob.glob(os.path.join(input_dir, f"*{ext}")))
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return []
    
    print(f"Found {len(image_files)} images to process")
    output_files = []
    
    # Process each image
    for img_path in image_files:
        img_name = os.path.basename(img_path)
        base_name, ext = os.path.splitext(img_name)
        print(f"Processing {img_name}...")
        
        # Load the image
        try:
            img = Image.open(img_path)
        except Exception as e:
            print(f"Error loading {img_path}: {e}")
            continue
        
        # Process according to mode
        if mode == "standard" or mode == "refine":
            # Generate a single image with the specified parameters
            result = apply_effect(img, default_params, seed)
            output_path = os.path.join(output_dir, f"{base_name}_effect{ext}")
            result.save(output_path, quality=95)
            output_files.append(output_path)
            print(f"Saved to {output_path}")
            
        elif mode == "explore":
            # Generate 4 variations with different seeds
            for i in range(4):
                current_seed = seed + i
                result = apply_effect(img, default_params, current_seed)
                output_path = os.path.join(output_dir, f"{base_name}_explore_{i+1}{ext}")
                result.save(output_path, quality=95)
                output_files.append(output_path)
                print(f"Saved variation {i+1} to {output_path}")
                
        elif mode == "sample":
            # Generate 6 artistic variations with the same seed
            presets = [
                {"focus_ratio": 0.2, "blur_strength": 0.5, "randomness": 0.3, "ghost_threshold": 0.4},
                {"focus_ratio": 0.3, "blur_strength": 0.8, "randomness": 0.6, "ghost_threshold": 0.5},
                {"focus_ratio": 0.4, "blur_strength": 0.9, "randomness": 0.4, "ghost_threshold": 0.6},
                {"focus_ratio": 0.5, "blur_strength": 0.7, "randomness": 0.7, "ghost_threshold": 0.3},
                {"focus_ratio": 0.6, "blur_strength": 0.6, "randomness": 0.5, "ghost_threshold": 0.4},
                {"focus_ratio": 0.7, "blur_strength": 0.5, "randomness": 0.8, "ghost_threshold": 0.2}
            ]
            
            for i, preset in enumerate(presets):
                # Merge with default parameters
                params_copy = default_params.copy()
                for key, value in preset.items():
                    params_copy[key] = value
                
                result = apply_effect(img, params_copy, seed)
                output_path = os.path.join(output_dir, f"{base_name}_sample_{i+1}{ext}")
                result.save(output_path, quality=95)
                output_files.append(output_path)
                print(f"Saved sample {i+1} to {output_path}")
    
    return output_files

def apply_effect(img, params, seed):
    """Apply the selective focus effect to a single image"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Extract parameters
    focus_ratio = params["focus_ratio"]  # 0.1-0.9
    blur_strength = params["blur_strength"]  # 0.1-1.0
    randomness = params["randomness"]  # 0.0-1.0
    ghost_threshold = params["ghost_threshold"]  # 0.0-1.0
    
    # Create a blurred copy of the image
    blur_radius = int(20 * blur_strength)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Create a mask for selective focus
    width, height = img.size
    mask = Image.new('L', (width, height), 0)
    
    # Center of the image
    center_x, center_y = width // 2, height // 2
    
    # Radius of the focused area, modified with focus_ratio
    focus_size = min(width, height) * (1.0 - focus_ratio * 0.8)
    
    # Add randomness to the focus center if requested
    if randomness > 0:
        center_x += int(random.uniform(-width * 0.2, width * 0.2) * randomness)
        center_y += int(random.uniform(-height * 0.2, height * 0.2) * randomness)
    
    # Create focus mask with gradual transition
    draw = ImageDraw.Draw(mask)
    draw.ellipse((center_x - focus_size/2, center_y - focus_size/2,
                  center_x + focus_size/2, center_y + focus_size/2), fill=255)
    
    # Blur the mask edges for smooth transition
    mask = mask.filter(ImageFilter.GaussianBlur(radius=focus_size/10))
    
    # Apply randomness to the mask if requested
    if randomness > 0:
        mask_array = np.array(mask)
        noise = np.random.normal(0, 30 * randomness, mask_array.shape)
        mask_array = np.clip(mask_array + noise, 0, 255).astype(np.uint8)
        mask = Image.fromarray(mask_array)
    
    # Combine the original image and the blurred one using the mask
    result = Image.composite(img, blurred, mask)
    
    # Apply "ghost" effect if threshold > 0
    if ghost_threshold > 0:
        result = apply_ghost_effect(result, mask, ghost_threshold)
    
    return result

def apply_ghost_effect(img, mask, threshold):
    """Apply a creative 'ghost' effect to out-of-focus areas"""
    img = img.convert('RGBA')
    width, height = img.size
    result = img.copy()
    pixels = result.load()
    mask_pixels = mask.load()
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            mask_value = mask_pixels[x, y]
            
            # If we're outside the focus area based on threshold
            if mask_value < 255 * threshold:
                # Partial desaturation effect
                gray = (r + g + b) // 3
                factor = 0.7  # Effect intensity
                
                # Mix between original color and gray
                r = int(r * (1 - factor) + gray * factor)
                g = int(g * (1 - factor) + gray * factor)
                b = int(b * (1 - factor) + gray * factor)
                
                # Add slight blue/cold tint for ethereal effect
                b = min(255, int(b * 1.1))
                
                pixels[x, y] = (r, g, b, a)
    
    return result

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Apply selective focus effect to images.")
    parser.add_argument("--input_dir", required=True, help="Input directory containing images")
    parser.add_argument("--output_dir", default="output", help="Output directory for processed images")
    parser.add_argument("--mode", default="standard", choices=["standard", "explore", "sample", "refine"],
                        help="Processing mode")
    parser.add_argument("--seed", type=int, help="Random seed for reproducibility")
    parser.add_argument("--focus_ratio", type=float, help="Focus ratio (0.1-0.9)")
    parser.add_argument("--blur_strength", type=float, help="Blur strength (0.1-1.0)")
    parser.add_argument("--randomness", type=float, help="Randomness (0.0-1.0)")
    parser.add_argument("--ghost_threshold", type=float, help="Ghost threshold (0.0-1.0)")
    
    args = parser.parse_args()
    
    # Extract parameters
    params = {}
    if args.focus_ratio is not None:
        params["focus_ratio"] = args.focus_ratio
    if args.blur_strength is not None:
        params["blur_strength"] = args.blur_strength
    if args.randomness is not None:
        params["randomness"] = args.randomness
    if args.ghost_threshold is not None:
        params["ghost_threshold"] = args.ghost_threshold
    
    # Process images
    start_time = time.time()
    output_files = process_images(args.input_dir, args.output_dir, args.mode, args.seed, **params)
    elapsed_time = time.time() - start_time
    
    print(f"Processing completed in {elapsed_time:.2f} seconds.")
    print(f"Generated {len(output_files)} output images in {args.output_dir}")

if __name__ == "__main__":
    main()
