#!/usr/bin/env python3
"""
Final fruit fix: flatten local files, add RigidBodyAPI, replace MDL with proper UsdPreviewSurface.
"""

import os
from pxr import Usd, UsdGeom, UsdShade, UsdPhysics, Sdf, Gf

FRUITS_DIR = "/home/bizon/sparkpack/openarm_isaac_lab_trainer/source/openarm/openarm/tasks/manager_based/openarm_manipulation/usds/fruits"
OUTPUT_DIR = os.path.join(FRUITS_DIR, "fixed")
TEXTURES_DIR = os.path.join(FRUITS_DIR, "textures")

# Fruit config: name -> (diffuse_tex, normal_tex, roughness_tex, color_fallback, roughness_value)
FRUIT_CONFIG = {
    "orange_02": ("orange_fruit_basecolor.png", "orange_fruit_normal.png", "orange_fruit_roughness.png", (1.0, 0.5, 0.0), 0.3),
    "lemon_02": ("Lemons_BaseColor.png", "Lemons_Normal.png", None, (1.0, 0.9, 0.0), 0.2),
    "lime01": ("Lime01_A.png", "Lime01_N.png", "Lime01_R.png", (0.2, 0.8, 0.0), 0.2),
    "avocado01": ("Avocado01_A.png", "Avocado01_N.png", "Avocado01_R.png", (0.3, 0.4, 0.1), 0.5),
    "pomegranate01": ("Pomegranate01_A.png", "Pomegranate01_N.png", "Pomegranate01_R.png", (0.7, 0.1, 0.2), 0.3),
    "lychee01": ("Lychee01_A.png", "Lychee01_N.png", "Lychee01_R.png", (0.9, 0.7, 0.6), 0.35),
}

def create_preview_material(stage, mat_path, diffuse_tex, normal_tex, roughness_tex, fallback_color, roughness_val):
    """Create a proper UsdPreviewSurface material."""
    material = UsdShade.Material.Define(stage, mat_path)
    
    # Main shader
    shader = UsdShade.Shader.Define(stage, mat_path.AppendPath("Shader"))
    shader.CreateIdAttr("UsdPreviewSurface")
    
    # Set roughness (lower = shinier)
    shader.CreateInput("roughness", Sdf.ValueTypeNames.Float).Set(roughness_val)
    shader.CreateInput("metallic", Sdf.ValueTypeNames.Float).Set(0.0)
    shader.CreateInput("ior", Sdf.ValueTypeNames.Float).Set(1.5)
    
    # UV reader
    uv_reader = UsdShade.Shader.Define(stage, mat_path.AppendPath("UVReader"))
    uv_reader.CreateIdAttr("UsdPrimvarReader_float2")
    uv_reader.CreateInput("varname", Sdf.ValueTypeNames.Token).Set("st")
    uv_reader.CreateOutput("result", Sdf.ValueTypeNames.Float2)
    
    # Diffuse texture
    diffuse_path = os.path.join(TEXTURES_DIR, diffuse_tex)
    if os.path.exists(diffuse_path):
        tex = UsdShade.Shader.Define(stage, mat_path.AppendPath("DiffuseTex"))
        tex.CreateIdAttr("UsdUVTexture")
        tex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"../textures/{diffuse_tex}")
        tex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
        tex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
        tex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).ConnectToSource(tex.ConnectableAPI(), "rgb")
    else:
        shader.CreateInput("diffuseColor", Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*fallback_color))
    
    # Normal texture
    if normal_tex:
        normal_path = os.path.join(TEXTURES_DIR, normal_tex)
        if os.path.exists(normal_path):
            ntex = UsdShade.Shader.Define(stage, mat_path.AppendPath("NormalTex"))
            ntex.CreateIdAttr("UsdUVTexture")
            ntex.CreateInput("file", Sdf.ValueTypeNames.Asset).Set(f"../textures/{normal_tex}")
            ntex.CreateInput("st", Sdf.ValueTypeNames.Float2).ConnectToSource(uv_reader.ConnectableAPI(), "result")
            ntex.CreateInput("wrapS", Sdf.ValueTypeNames.Token).Set("repeat")
            ntex.CreateInput("wrapT", Sdf.ValueTypeNames.Token).Set("repeat")
            ntex.CreateOutput("rgb", Sdf.ValueTypeNames.Float3)
            shader.CreateInput("normal", Sdf.ValueTypeNames.Normal3f).ConnectToSource(ntex.ConnectableAPI(), "rgb")
    
    # Connect material output
    material.CreateSurfaceOutput().ConnectToSource(shader.ConnectableAPI(), "surface")
    
    return material

def fix_fruit(fruit_name):
    """Fix a single fruit USD."""
    config = FRUIT_CONFIG.get(fruit_name)
    if not config:
        print(f"  SKIP: No config for {fruit_name}")
        return False
    
    diffuse_tex, normal_tex, roughness_tex, fallback_color, roughness_val = config
    input_path = os.path.join(FRUITS_DIR, f"{fruit_name}.usd")
    output_path = os.path.join(OUTPUT_DIR, f"{fruit_name}.usd")
    
    print(f"\n{'='*60}")
    print(f"Processing: {fruit_name}")
    print(f"{'='*60}")
    
    # Open and flatten
    stage = Usd.Stage.Open(input_path)
    if not stage:
        print(f"  ERROR: Could not open {input_path}")
        return False
    
    flattened = stage.Flatten()
    new_stage = Usd.Stage.CreateNew(output_path)
    Sdf.CopySpec(flattened, Sdf.Path.absoluteRootPath, new_stage.GetRootLayer(), Sdf.Path.absoluteRootPath)
    
    # Get root prim
    root_prim = new_stage.GetDefaultPrim()
    if not root_prim:
        for prim in new_stage.GetPseudoRoot().GetChildren():
            if prim.IsActive():
                root_prim = prim
                break
    
    print(f"  Root: {root_prim.GetPath()}")
    
    # Apply RigidBodyAPI
    UsdPhysics.RigidBodyAPI.Apply(root_prim)
    print(f"  Applied RigidBodyAPI")
    
    # Apply MassAPI
    mass_api = UsdPhysics.MassAPI.Apply(root_prim)
    mass_api.GetMassAttr().Set(0.15)
    
    # Create new material
    mat_path = root_prim.GetPath().AppendPath("FruitMaterial")
    material = create_preview_material(new_stage, mat_path, diffuse_tex, normal_tex, roughness_tex, fallback_color, roughness_val)
    print(f"  Created material with roughness={roughness_val}")
    
    # Find meshes, apply collision and bind material
    for prim in new_stage.Traverse():
        if prim.GetTypeName() == 'Mesh':
            # Apply collision with convexHull
            if not prim.HasAPI(UsdPhysics.CollisionAPI):
                UsdPhysics.CollisionAPI.Apply(prim)
            if not prim.HasAPI(UsdPhysics.MeshCollisionAPI):
                mesh_col = UsdPhysics.MeshCollisionAPI.Apply(prim)
                mesh_col.GetApproximationAttr().Set("convexHull")
            
            # Bind material
            binding = UsdShade.MaterialBindingAPI(prim)
            binding.Bind(material)
    
    # Set metadata
    new_stage.SetDefaultPrim(root_prim)
    UsdGeom.SetStageUpAxis(new_stage, UsdGeom.Tokens.z)
    UsdGeom.SetStageMetersPerUnit(new_stage, 1.0)
    
    new_stage.GetRootLayer().Save()
    print(f"  Saved: {output_path}")
    
    return True

def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Remove old files
    for f in os.listdir(OUTPUT_DIR):
        if f.endswith('.usd'):
            os.remove(os.path.join(OUTPUT_DIR, f))
    
    success = 0
    for name in FRUIT_CONFIG.keys():
        if fix_fruit(name):
            success += 1
    
    print(f"\n{'='*60}")
    print(f"Done! {success}/{len(FRUIT_CONFIG)} fruits fixed")
    print(f"{'='*60}")

if __name__ == "__main__":
    main()
