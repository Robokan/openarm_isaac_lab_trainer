#!/usr/bin/env python3
"""
Fix fruit USD materials to use relative paths for textures and materials.
"""

import os
from pxr import Usd, UsdShade, Sdf

FRUITS_DIR = "/home/bizon/sparkpack/openarm_isaac_lab_trainer/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/fruits"
FIXED_DIR = os.path.join(FRUITS_DIR, "fixed")
TEXTURES_DIR = os.path.join(FRUITS_DIR, "textures")
MATERIALS_DIR = os.path.join(FRUITS_DIR, "materials")

def fix_asset_path(old_path, fruit_usd_dir):
    """Convert absolute path to relative path."""
    if not old_path:
        return old_path
    
    # Remove @ delimiters if present
    clean_path = str(old_path).replace('@', '')
    
    if not clean_path:
        return old_path
    
    # Get just the filename
    filename = os.path.basename(clean_path)
    
    # Check if it's a texture
    if filename.endswith('.png') or filename.endswith('.jpg'):
        # Use relative path to textures directory
        rel_path = os.path.relpath(os.path.join(TEXTURES_DIR, filename), fruit_usd_dir)
        return f"@{rel_path}@"
    
    # Check if it's an MDL material
    if filename.endswith('.mdl'):
        rel_path = os.path.relpath(os.path.join(MATERIALS_DIR, filename), fruit_usd_dir)
        return f"@{rel_path}@"
    
    return old_path

def fix_fruit_usd(fruit_name):
    """Fix material paths in a single fruit USD."""
    usd_path = os.path.join(FIXED_DIR, f"{fruit_name}.usd")
    
    if not os.path.exists(usd_path):
        print(f"  SKIP: {usd_path} not found")
        return False
    
    print(f"\n{'='*60}")
    print(f"Fixing: {fruit_name}")
    print(f"{'='*60}")
    
    # Open the stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"  ERROR: Could not open {usd_path}")
        return False
    
    fruit_usd_dir = FIXED_DIR
    modified = False
    
    # Find all shaders and fix their paths
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'Shader':
            print(f"  Processing shader: {prim.GetPath()}")
            
            # Fix MDL source asset path
            mdl_attr = prim.GetAttribute('info:mdl:sourceAsset')
            if mdl_attr and mdl_attr.Get():
                old_val = str(mdl_attr.Get())
                new_val = fix_asset_path(old_val, fruit_usd_dir)
                if new_val != old_val:
                    mdl_attr.Set(Sdf.AssetPath(new_val.replace('@', '')))
                    print(f"    Fixed MDL: {old_val} -> {new_val}")
                    modified = True
            
            # Fix texture inputs
            for attr in prim.GetAttributes():
                attr_name = attr.GetName()
                if attr_name.startswith('inputs:') and 'texture' in attr_name.lower():
                    val = attr.Get()
                    if val:
                        old_val = str(val)
                        if '.png' in old_val.lower() or '.jpg' in old_val.lower():
                            new_val = fix_asset_path(old_val, fruit_usd_dir)
                            if new_val != old_val:
                                attr.Set(Sdf.AssetPath(new_val.replace('@', '')))
                                print(f"    Fixed {attr_name}: {os.path.basename(old_val.replace('@', ''))}")
                                modified = True
    
    if modified:
        stage.GetRootLayer().Save()
        print(f"  Saved: {usd_path}")
    else:
        print(f"  No changes needed")
    
    return True

def main():
    fruits = [
        "orange_02",
        "lemon_02", 
        "lime01",
        "avocado01",
        "pomegranate01",
        "lychee01",
    ]
    
    print(f"Materials dir: {MATERIALS_DIR}")
    print(f"Textures dir: {TEXTURES_DIR}")
    print(f"Fixed USDs dir: {FIXED_DIR}")
    
    for fruit in fruits:
        fix_fruit_usd(fruit)
    
    print(f"\n{'='*60}")
    print("Done! Material paths have been updated to use relative references.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
