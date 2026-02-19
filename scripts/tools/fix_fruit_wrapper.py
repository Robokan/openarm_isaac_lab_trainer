#!/usr/bin/env python3
"""
Create wrapper USDs that reference original fruits and add RigidBodyAPI.
This preserves all original materials without flattening.
"""

import os
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

FRUITS_DIR = "/home/bizon/sparkpack/openarm_isaac_lab_trainer/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/fruits"
OUTPUT_DIR = os.path.join(FRUITS_DIR, "fixed")

# Local paths (relative from fixed/ to parent fruits/ directory)
FRUITS = {
    "orange_02": "../orange_02.usd",
    "lemon_02": "../lemon_02.usd",
    "lime01": "../lime01.usd",
    "avocado01": "../avocado01.usd",
    "pomegranate01": "../pomegranate01.usd",
    "lychee01": "../lychee01.usd",
}

def create_wrapper_usd(fruit_name, source_url):
    """Create a wrapper USD that references the original and adds physics."""
    output_path = os.path.join(OUTPUT_DIR, f"{fruit_name}.usd")
    
    print(f"\n{'='*60}")
    print(f"Creating wrapper: {fruit_name}")
    print(f"{'='*60}")
    
    # Create new stage
    stage = Usd.Stage.CreateNew(output_path)
    
    # Create root prim
    root_path = Sdf.Path("/Root")
    root_prim = stage.DefinePrim(root_path, "Xform")
    stage.SetDefaultPrim(root_prim)
    
    # Add reference to original SimReady asset
    root_prim.GetReferences().AddReference(source_url)
    print(f"  Added reference to: {source_url}")
    
    # Apply RigidBodyAPI to root
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    print(f"  Applied RigidBodyAPI")
    
    # Apply MassAPI
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.GetMassAttr().Set(0.15)
    print(f"  Applied MassAPI (0.15 kg)")
    
    # Apply CollisionAPI with convexHull
    UsdPhysics.CollisionAPI.Apply(root_prim)
    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(root_prim)
    mesh_col.GetApproximationAttr().Set("convexHull")
    print(f"  Applied CollisionAPI (convexHull)")
    
    # Set stage metadata
    UsdGeom.SetStageUpAxis(stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(stage, 1.0)
    
    # Save
    stage.GetRootLayer().Save()
    print(f"  Saved: {output_path}")
    
    return True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    
    # Remove old fixed files first
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.usd'):
            os.remove(os.path.join(OUTPUT_DIR, f))
    
    success = 0
    for name, url in FRUITS.items():
        if create_wrapper_usd(name, url):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(FRUITS)} wrapper USDs created")
    print(f"These reference the original SimReady assets via URL")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
