#!/usr/bin/env python3
"""
Script to download SimReady fruit assets and fix their rigid body structure.
SimReady assets have RigidBodyAPI on child meshes, but Isaac Lab RigidObjectCfg
expects RigidBodyAPI at the root prim level.

This script:
1. Downloads the full fruit assets (not just stubs)
2. Moves RigidBodyAPI from child mesh to root prim
3. Saves the fixed USDs locally
"""

import os
import sys

# Import pxr (USD) - this works when running through Isaac Sim's python
try:
    from pxr import Usd, UsdGeom, UsdPhysics, Sdf, Gf, PhysxSchema
except ImportError:
    print("ERROR: pxr module not available. Run this script through Isaac Sim's python.")
    print("Usage: /path/to/IsaacLab/_isaac_sim/python.sh fix_fruits.py")
    sys.exit(1)

# Fruit asset URLs from SimReady
FRUIT_URLS = {
    "orange": "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/orange_02/orange_02.usd",
    "lemon": "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/lemon_02/lemon_02.usd",
    "lime": "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/lime01/lime01.usd",
    "avocado": "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/avocado01/avocado01.usd",
    "pomegranate": "https://omniverse-content-staging.s3.us-west-2.amazonaws.com/Assets/simready_content/common_assets/props/pomegranate01/pomegranate01.usd",
}

OUTPUT_DIR = "/home/bizon/sparkpack/openarm_isaac_lab_trainer/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/fruits"


def analyze_stage(stage, name):
    """Analyze the structure of a USD stage."""
    print(f"\n=== Analyzing {name} ===")
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        prims = list(stage.GetPseudoRoot().GetChildren())
        if prims:
            root_prim = prims[0]
    
    if root_prim:
        print(f"Root prim: {root_prim.GetPath()}")
        print(f"  Type: {root_prim.GetTypeName()}")
        print(f"  Has RigidBodyAPI: {root_prim.HasAPI(UsdPhysics.RigidBodyAPI)}")
        print(f"  Has CollisionAPI: {root_prim.HasAPI(UsdPhysics.CollisionAPI)}")
        print(f"  Has MassAPI: {root_prim.HasAPI(UsdPhysics.MassAPI)}")
        
    print("\nAll prims with RigidBodyAPI:")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI):
            print(f"  {prim.GetPath()} (type: {prim.GetTypeName()})")
            
    print("\nAll prims with CollisionAPI:")
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.CollisionAPI):
            print(f"  {prim.GetPath()} (type: {prim.GetTypeName()})")


def fix_rigid_body(stage, name):
    """
    Fix the rigid body structure:
    1. Find the child prim with RigidBodyAPI
    2. Copy physics properties to root prim
    3. Remove RigidBodyAPI from child
    """
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        prims = list(stage.GetPseudoRoot().GetChildren())
        if prims:
            root_prim = prims[0]
    
    if not root_prim:
        print(f"ERROR: No root prim found in {name}")
        return False
        
    # Check if root already has RigidBodyAPI
    if root_prim.HasAPI(UsdPhysics.RigidBodyAPI):
        print(f"{name}: Root already has RigidBodyAPI, no fix needed")
        return True
        
    # Find child with RigidBodyAPI
    rigid_child = None
    for prim in stage.Traverse():
        if prim.HasAPI(UsdPhysics.RigidBodyAPI) and prim != root_prim:
            rigid_child = prim
            break
            
    if not rigid_child:
        print(f"{name}: No child with RigidBodyAPI found, adding to root")
        # Just add RigidBodyAPI to root
        UsdPhysics.RigidBodyAPI.Apply(root_prim)
        UsdPhysics.CollisionAPI.Apply(root_prim)
        UsdPhysics.MassAPI.Apply(root_prim)
        return True
        
    print(f"{name}: Moving RigidBodyAPI from {rigid_child.GetPath()} to {root_prim.GetPath()}")
    
    # Copy mass properties from child if it has MassAPI
    mass = None
    if rigid_child.HasAPI(UsdPhysics.MassAPI):
        mass_api = UsdPhysics.MassAPI(rigid_child)
        mass = mass_api.GetMassAttr().Get()
        print(f"  Found mass: {mass}")
    
    # Apply APIs to root
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    
    # Add MassAPI if we found mass
    if mass is not None:
        mass_api = UsdPhysics.MassAPI.Apply(root_prim)
        mass_api.GetMassAttr().Set(mass)
    
    # Remove RigidBodyAPI from child (make it just a collision shape)
    # We can't easily remove APIs in USD, so we'll just leave it
    # The root-level RigidBodyAPI should take precedence
    
    # Actually let's try to remove it
    try:
        rigid_child.RemoveAPI(UsdPhysics.RigidBodyAPI)
        print(f"  Removed RigidBodyAPI from child")
    except Exception as e:
        print(f"  Could not remove RigidBodyAPI from child: {e}")
    
    return True


def process_fruit(name, url):
    """Process a single fruit asset."""
    print(f"\n{'='*60}")
    print(f"Processing: {name}")
    print(f"URL: {url}")
    print(f"{'='*60}")
    
    # Open the stage (this will resolve references from the URL)
    try:
        stage = Usd.Stage.Open(url)
        if not stage:
            print(f"ERROR: Could not open {url}")
            return False
    except Exception as e:
        print(f"ERROR opening {url}: {e}")
        return False
    
    # Analyze original structure
    analyze_stage(stage, f"{name} (original)")
    
    # Fix the rigid body structure
    if not fix_rigid_body(stage, name):
        return False
    
    # Flatten the stage to resolve all references
    flattened = stage.Flatten()
    
    # Save to local file
    output_path = os.path.join(OUTPUT_DIR, f"{name}.usd")
    
    # Create a new stage and copy the flattened content
    new_stage = Usd.Stage.CreateNew(output_path)
    
    # Copy all prims from flattened stage
    for prim in flattened.Traverse():
        Sdf.CopySpec(flattened.GetRootLayer(), prim.GetPath(), 
                     new_stage.GetRootLayer(), prim.GetPath())
    
    # Set the default prim
    root_prims = list(new_stage.GetPseudoRoot().GetChildren())
    if root_prims:
        new_stage.SetDefaultPrim(root_prims[0])
    
    new_stage.Save()
    print(f"Saved: {output_path}")
    
    # Verify the result
    verify_stage = Usd.Stage.Open(output_path)
    analyze_stage(verify_stage, f"{name} (fixed)")
    
    return True


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    print("=" * 60)
    print("Fixing SimReady Fruit Assets for Isaac Lab")
    print("=" * 60)
    
    success_count = 0
    for name, url in FRUIT_URLS.items():
        if process_fruit(name, url):
            success_count += 1
    
    print(f"\n{'='*60}")
    print(f"Complete: {success_count}/{len(FRUIT_URLS)} fruits processed successfully")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
