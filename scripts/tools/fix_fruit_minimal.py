#!/usr/bin/env python3
"""
Minimal fix for SimReady fruit assets - ONLY adds RigidBodyAPI to root.
Does NOT touch materials at all.
"""

import os
from pxr import Usd, UsdGeom, UsdPhysics, Sdf

FRUITS_DIR = "/home/bizon/sparkpack/openarm_isaac_lab_trainer/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/fruits"
OUTPUT_DIR = os.path.join(FRUITS_DIR, "fixed")

FRUITS = [
    "orange_02",
    "lemon_02", 
    "lime01",
    "avocado01",
    "pomegranate01",
    "lychee01",
]

def fix_fruit_usd(fruit_name):
    """Add RigidBodyAPI to root without touching materials."""
    input_path = os.path.join(FRUITS_DIR, f"{fruit_name}.usd")
    output_path = os.path.join(OUTPUT_DIR, f"{fruit_name}.usd")
    
    print(f"\n{'='*60}")
    print(f"Processing: {fruit_name}")
    print(f"{'='*60}")
    
    # Open and flatten the stage
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"  ERROR: Could not open {input_path}")
        return False
    
    # Flatten to resolve all references
    flattened = stage.Flatten()
    
    # Create new stage from flattened
    new_stage = Usd.Stage.CreateNew(output_path)
    Sdf.CopySpec(flattened, Sdf.Path.absoluteRootPath, 
                 new_stage.GetRootLayer(), Sdf.Path.absoluteRootPath)
    
    # Find root prim
    root_prim = new_stage.GetDefaultPrim()
    if not root_prim:
        for prim in new_stage.GetPseudoRoot().GetChildren():
            if prim.IsActive():
                root_prim = prim
                break
    
    if not root_prim:
        print(f"  ERROR: No root prim found")
        return False
    
    print(f"  Root prim: {root_prim.GetPath()}")
    
    # Check if RigidBodyAPI already exists anywhere
    existing_rigid = None
    for prim in new_stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            existing_rigid = prim
            break
    
    if existing_rigid:
        print(f"  Found existing RigidBodyAPI on: {existing_rigid.GetPath()}")
        if existing_rigid.GetPath() != root_prim.GetPath():
            # Remove from child
            existing_rigid.RemoveAPI(UsdPhysics.RigidBodyAPI)
            print(f"  Removed from child")
    
    # Apply RigidBodyAPI to root
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        print(f"  Applied RigidBodyAPI to root")
    
    # Apply MassAPI if not present
    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.GetMassAttr().Set(0.15)  # 150g default
        print(f"  Applied MassAPI (0.15 kg)")
    
    # Ensure collision on meshes
    for prim in new_stage.Traverse():
        if prim.GetTypeName() == 'Mesh':
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
                # Use convex hull for performance
                if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                    mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                    mesh_col.GetApproximationAttr().Set("convexHull")
    
    # Set stage metadata
    new_stage.SetDefaultPrim(root_prim)
    UsdGeom.SetStageUpAxis(new_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(new_stage, 1.0)
    
    # Save
    new_stage.GetRootLayer().Save()
    print(f"  Saved: {output_path}")
    
    # Verify
    verify = Usd.Stage.Open(output_path)
    vroot = verify.GetDefaultPrim()
    has_rigid = vroot.HasAPI(UsdPhysics.RigidBodyAPI) if vroot else False
    print(f"  Verified RigidBodyAPI: {has_rigid}")
    
    return has_rigid

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output: {OUTPUT_DIR}")
    
    success = 0
    for fruit in FRUITS:
        if fix_fruit_usd(fruit):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(FRUITS)} fruits processed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
