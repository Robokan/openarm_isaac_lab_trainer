#!/usr/bin/env python3
"""
Replace MDL materials with simple USD Preview Surface materials.
These work universally without needing MDL infrastructure.
"""

import os
from pxr import Usd, UsdShade, UsdGeom, Sdf, Gf

FRUITS_DIR = "/home/bizon/sparkpack/openarm_isaac_lab_trainer/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/fruits"
FIXED_DIR = os.path.join(FRUITS_DIR, "fixed")
TEXTURES_DIR = os.path.join(FRUITS_DIR, "textures")

# Fruit info: name -> (diffuse_texture, fallback_color)
FRUIT_INFO = {
    "orange_02": ("orange_fruit_basecolor.png", (1.0, 0.5, 0.0)),  # Orange
    "lemon_02": ("Lemons_BaseColor.png", (1.0, 0.9, 0.0)),  # Yellow
    "lime01": ("Lime01_A.png", (0.2, 0.8, 0.0)),  # Green
    "avocado01": ("Avocado01_A.png", (0.2, 0.4, 0.1)),  # Dark green
    "pomegranate01": ("Pomegranate01_A.png", (0.8, 0.1, 0.2)),  # Red
    "lychee01": ("Lychee01_A.png", (0.9, 0.6, 0.5)),  # Pinkish
}

def create_preview_material(stage, material_path, diffuse_texture_path, fallback_color):
    """Create a UsdPreviewSurface material with texture."""
    
    # Create material
    material = UsdShade.Material.Define(stage, material_path)
    
    # Create shader
    shader_path = material_path.AppendPath("PreviewSurface")
    shader = UsdShade.Shader.Define(stage, shader_path)
    shader.CreateIdAttr("UsdPreviewSurface")
    
    # Set basic properties
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(0.5)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    
    # Check if texture exists
    if os.path.exists(diffuse_texture_path):
        # Create texture reader
        tex_reader_path = material_path.AppendPath("DiffuseTexture")
        tex_reader = UsdShade.Shader.Define(stage, tex_reader_path)
        tex_reader.CreateIdAttr("UsdUVTexture")
        tex_reader.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(diffuse_texture_path)
        tex_reader.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_reader.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex_reader.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        
        # Create primvar reader for UVs
        uv_reader_path = material_path.AppendPath("UVReader")
        uv_reader = UsdShade.Shader.Define(stage, uv_reader_path)
        uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
        uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
        uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
        
        # Connect UV reader to texture
        tex_reader.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(
            uv_reader.ConnectableAPI(), "result"
        )
        
        # Connect texture to shader diffuse
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(
            tex_reader.ConnectableAPI(), "rgb"
        )
    else:
        # Use fallback color
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*fallback_color))
    
    # Create material outputs
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    return material

def fix_fruit_usd(fruit_name):
    """Replace MDL materials with simple preview materials."""
    usd_path = os.path.join(FIXED_DIR, f"{fruit_name}.usd")
    
    if not os.path.exists(usd_path):
        print(f"  SKIP: {usd_path} not found")
        return False
    
    print(f"\n{'='*60}")
    print(f"Fixing: {fruit_name}")
    print(f"{'='*60}")
    
    info = FRUIT_INFO.get(fruit_name)
    if not info:
        print(f"  ERROR: No info for {fruit_name}")
        return False
    
    diffuse_tex, fallback_color = info
    diffuse_path = os.path.join(TEXTURES_DIR, diffuse_tex)
    
    # Open the stage
    stage = Usd.Stage.Open(usd_path)
    if not stage:
        print(f"  ERROR: Could not open {usd_path}")
        return False
    
    # Find all mesh prims and their material bindings
    meshes = []
    for prim in stage.Traverse():
        if prim.GetTypeName() == 'Mesh':
            meshes.append(prim)
            print(f"  Found mesh: {prim.GetPath()}")
    
    if not meshes:
        print(f"  ERROR: No meshes found")
        return False
    
    # Create a new simple material at the root
    root_prim = stage.GetDefaultPrim()
    if not root_prim:
        root_prim = stage.GetPrimAtPath("/RootNode")
    
    # Create Looks scope if it doesn't exist
    looks_path = root_prim.GetPath().AppendPath("SimpleLooks")
    looks_prim = stage.GetPrimAtPath(looks_path)
    if not looks_prim:
        looks_prim = stage.DefinePrim(looks_path, "Scope")
    
    # Create simple material
    material_path = looks_path.AppendPath("SimpleMaterial")
    
    # Use relative path for texture
    rel_tex_path = os.path.relpath(diffuse_path, FIXED_DIR)
    print(f"  Texture: {rel_tex_path} (exists: {os.path.exists(diffuse_path)})")
    
    material = create_preview_material(stage, material_path, rel_tex_path, fallback_color)
    print(f"  Created material: {material_path}")
    
    # Bind material to all meshes
    for mesh_prim in meshes:
        binding = UsdShade.MaterialBindingAPI(mesh_prim)
        binding.Bind(material)
        print(f"  Bound material to: {mesh_prim.GetPath()}")
    
    # Save
    stage.GetRootLayer().Save()
    print(f"  Saved: {usd_path}")
    
    return True

def main():
    print("Replacing MDL materials with simple USD Preview Surface materials")
    print(f"Textures dir: {TEXTURES_DIR}")
    print(f"Fixed USDs dir: {FIXED_DIR}")
    
    for fruit_name in FRUIT_INFO.keys():
        fix_fruit_usd(fruit_name)
    
    print(f"\n{'='*60}")
    print("Done! Materials have been replaced with simple preview surfaces.")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
