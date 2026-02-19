#!/usr/bin/env python3
"""
Fix SimReady fruit assets by moving RigidBodyAPI from child meshes to root prim.
This makes them compatible with Isaac Lab's RigidObjectCfg.
"""

import os
from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf

# Try to import PhysxSchema (only available in Isaac Sim runtime)
try:
    from pxr import PhysxSchema
    HAS_PHYSX = True
except ImportError:
    HAS_PHYSX = False
    print("Note: PhysxSchema not available, using standard USD physics only")

FRUITS_DIR = "/home/bizon/sparkpack/openarm_isaac_lab_trainer/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/fruits"
OUTPUT_DIR = os.path.join(FRUITS_DIR, "fixed")

# Fruit files to process (main USD files that reference _base.usd)
FRUITS = [
    "orange_02",
    "lemon_02",
    "lime01",
    "avocado01",
    "pomegranate01",
    "lychee01",
]

def find_rigid_body_prim(stage):
    """Find the first prim with RigidBodyAPI."""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            return prim
    return None

def find_collision_prim(stage):
    """Find the first prim with CollisionAPI."""
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            return prim
    return None

def get_mass_properties(prim):
    """Get mass properties from a prim if it has MassAPI."""
    if prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(prim)
        return {
            'mass': mass_api.GetMassAttr().Get(),
            'density': mass_api.GetDensityAttr().Get(),
            'center_of_mass': mass_api.GetCenterOfMassAttr().Get(),
        }
    return None

def fix_fruit_usd(fruit_name):
    """Fix a single fruit USD by moving RigidBodyAPI to root."""
    input_path = os.path.join(FRUITS_DIR, f"{fruit_name}.usd")
    output_path = os.path.join(OUTPUT_DIR, f"{fruit_name}.usd")
    
    print(f"\n{'='*60}")
    print(f"Processing: {fruit_name}")
    print(f"{'='*60}")
    
    # Open the stage and flatten it (resolves all references)
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"  ERROR: Could not open {input_path}")
        return False
    
    # Flatten to resolve references
    flattened_layer = stage.Flatten()
    
    # Create a new stage from the flattened layer
    new_stage = Usd.Stage.CreateNew(output_path)
    
    # Copy the flattened content
    Sdf.CopySpec(flattened_layer, Sdf.Path.absoluteRootPath, 
                 new_stage.GetRootLayer(), Sdf.Path.absoluteRootPath)
    
    # Get the root prim (usually named after the fruit)
    root_prim = new_stage.GetDefaultPrim()
    if not root_prim:
        # Try to find it
        for prim in new_stage.GetPseudoRoot().GetChildren():
            if prim.GetTypeName() in ['Xform', 'Scope', '']:
                root_prim = prim
                break
    
    if not root_prim:
        print(f"  ERROR: Could not find root prim")
        return False
    
    print(f"  Root prim: {root_prim.GetPath()}")
    
    # Find existing RigidBodyAPI prim
    rigid_prim = find_rigid_body_prim(new_stage)
    collision_prim = find_collision_prim(new_stage)
    
    print(f"  Existing RigidBodyAPI on: {rigid_prim.GetPath() if rigid_prim else 'None'}")
    print(f"  Existing CollisionAPI on: {collision_prim.GetPath() if collision_prim else 'None'}")
    
    # Get mass properties from existing rigid body
    mass_props = None
    if rigid_prim:
        mass_props = get_mass_properties(rigid_prim)
        print(f"  Mass properties: {mass_props}")
        
        # Remove RigidBodyAPI from child prim
        rigid_prim.RemoveAPI(UsdPhysics.RigidBodyAPI)
        if HAS_PHYSX and rigid_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
            rigid_prim.RemoveAPI(PhysxSchema.PhysxRigidBodyAPI)
        if rigid_prim.HasAPI(UsdPhysics.MassAPI):
            rigid_prim.RemoveAPI(UsdPhysics.MassAPI)
        print(f"  Removed RigidBodyAPI from {rigid_prim.GetPath()}")
    
    # Apply RigidBodyAPI to root prim
    if not root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        print(f"  Applied RigidBodyAPI to root: {root_prim.GetPath()}")
    
    # Apply PhysxRigidBodyAPI for PhysX-specific properties (if available)
    if HAS_PHYSX and not root_prim.HasAPI(PhysxSchema.PhysxRigidBodyAPI):
        PhysxSchema.PhysxRigidBodyAPI.Apply(root_prim)
    
    # Apply MassAPI to root with default mass if we have properties
    if not root_prim.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        if mass_props and mass_props.get('mass'):
            mass_api.GetMassAttr().Set(mass_props['mass'])
        else:
            # Default mass for small fruit (in kg)
            mass_api.GetMassAttr().Set(0.15)
        print(f"  Applied MassAPI to root")
    
    # Ensure collision is set up on mesh children
    for prim in new_stage.Traverse():
        if prim.GetTypeName() == 'Mesh':
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
                print(f"  Applied CollisionAPI to mesh: {prim.GetPath()}")
            # Use mesh collision approximation
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_collision = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_collision.GetApproximationAttr().Set("convexHull")
    
    # Set the default prim
    new_stage.SetDefaultPrim(root_prim)
    
    # Set up/meters metadata
    UsdGeom.SetStageUpAxis(new_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(new_stage, 1.0)
    
    # Save
    new_stage.GetRootLayer().Save()
    print(f"  Saved to: {output_path}")
    
    # Verify
    verify_stage = Usd.Stage.Open(output_path)
    verify_root = verify_stage.GetDefaultPrim()
    has_rigid = verify_root.HasAPI(UsdPhysics.RigidBodyAPI) if verify_root else False
    print(f"  Verification - Root has RigidBodyAPI: {has_rigid}")
    
    return has_rigid

def main():
    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    print(f"Output directory: {OUTPUT_DIR}")
    
    success_count = 0
    for fruit in FRUITS:
        if fix_fruit_usd(fruit):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Done! Fixed {success_count}/{len(FRUITS)} fruits")
    print(f"Fixed USDs are in: {OUTPUT_DIR}")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
