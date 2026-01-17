import sys
import os

# Automatically handle LD_LIBRARY_PATH and QT_PLUGIN_PATH for Qt/PySide6 compatibility
if "LD_LIBRARY_PATH" not in os.environ or "PySide6" not in os.environ["LD_LIBRARY_PATH"]:
    # Attempt to locate PySide6 library path
    try:
        import PySide6
        pyside_dir = os.path.dirname(PySide6.__file__)
        qt_lib_path = os.path.join(pyside_dir, "Qt", "lib")
        qt_plugin_path = os.path.join(pyside_dir, "Qt", "plugins")
        
        if os.path.exists(qt_lib_path):
            print(f"Setting environment for Qt compatibility and restarting...")
            new_env = os.environ.copy()
            current_ld = new_env.get("LD_LIBRARY_PATH", "")
            new_env["LD_LIBRARY_PATH"] = f"{qt_lib_path}:{current_ld}"
            new_env["QT_PLUGIN_PATH"] = qt_plugin_path
            new_env["QT_QPA_PLATFORM"] = "offscreen" # Force offscreen rendering
            
            # Re-execute the script with the new environment
            os.execve(sys.executable, [sys.executable] + sys.argv, new_env)
    except Exception as e:
        print(f"Warning: Could not auto-set Qt environment: {e}")

from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from ovito.vis import Viewport, TachyonRenderer

# 1. Setup paths
input_file = "simulation/melt.lammpstrj"
output_base = "data"
os.makedirs(f"{output_base}/train/solid", exist_ok=True)
os.makedirs(f"{output_base}/train/liquid", exist_ok=True)

# 2. Load the trajectory
pipeline = import_file(input_file)

# 3. Add PTM Modifier to identify structure
# This identifies if atoms are FCC (Solid) or 'Other' (Liquid)
ptm = PolyhedralTemplateMatchingModifier()
pipeline.modifiers.append(ptm)

# 4. Set up rendering
pipeline.add_to_scene()
vp = Viewport()
vp.type = Viewport.Type.Perspective
vp.camera_dir = (-1, -1, -1) # High-angle view
vp.zoom_all()

# 5. Iterate through frames and save snapshots
for frame in range(pipeline.source.num_frames):
    data = pipeline.compute(frame)
    
    # Logic: PTM Type 0 = Unknown/Amorphous (Liquid in our case)
    # PTM Type 1 = FCC (Solid Copper)
    fcc_count = data.attributes.get('PolyhedralTemplateMatching.counts.FCC', 0)
    total_atoms = data.particles.count
    
    # If more than 50% is crystalline, label as solid
    label = "solid" if (fcc_count / total_atoms) > 0.5 else "liquid"
    
    # Render and save
    filename = f"{output_base}/train/{label}/frame_{frame:04d}.png"
    vp.render_image(filename=filename, size=(224, 224), frame=frame, renderer=TachyonRenderer())
    print(f"Frame {frame}: Classified as {label}")