#!/usr/bin/env python3
"""
Generate 1000+ training images by running 10 LAMMPS simulations with varied
seeds/box sizes, then rendering each trajectory with OVITO + PTM labeling.

Run with:
    conda run -n phaseovito python generation/generate_dataset.py

Optional flags:
    --lammps   PATH   Path to lmp_mpi binary (default: auto-detected)
    --mpi      N      MPI ranks per simulation (default: 4)
    --sims     N      Number of simulations to run (default: 10, max: 10)
    --skip-sim        Skip LAMMPS runs, only re-render existing trajectories
"""

import sys
import os

# Re-exec with correct environment if needed:
#   - PYTHONNOUSERSITE=1 prevents ~/.local ovito pip package from shadowing conda's ovito
#   - QT_QPA_PLATFORM=offscreen enables headless rendering
#   - LD_LIBRARY_PATH points to conda env's Qt libs
_needs_reexec = (
    os.environ.get("PYTHONNOUSERSITE") != "1"
    or "QT_QPA_PLATFORM" not in os.environ
)
if _needs_reexec:
    try:
        import PySide6
        pyside_dir = os.path.dirname(PySide6.__file__)
        qt_lib_path = os.path.join(pyside_dir, "Qt", "lib")
        qt_plugin_path = os.path.join(pyside_dir, "Qt", "plugins")
        new_env = os.environ.copy()
        new_env["PYTHONNOUSERSITE"] = "1"
        new_env["QT_QPA_PLATFORM"] = "offscreen"
        if os.path.exists(qt_lib_path):
            new_env["LD_LIBRARY_PATH"] = f"{qt_lib_path}:{new_env.get('LD_LIBRARY_PATH', '')}"
            new_env["QT_PLUGIN_PATH"] = qt_plugin_path
        os.execve(sys.executable, [sys.executable] + sys.argv, new_env)
    except Exception as e:
        print(f"Warning: env setup failed: {e}")

import argparse
import shutil
import subprocess
from pathlib import Path

from ovito.io import import_file
from ovito.modifiers import PolyhedralTemplateMatchingModifier
from ovito.vis import Viewport, TachyonRenderer

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT  = Path(__file__).resolve().parent.parent
SIM_DIR    = REPO_ROOT / "simulation"
RUNS_DIR   = SIM_DIR / "runs"
DATA_SOLID = REPO_ROOT / "data" / "train" / "solid"
DATA_LIQ   = REPO_ROOT / "data" / "train" / "liquid"
EAM_FILE   = SIM_DIR / "Cu_u3.eam"

DEFAULT_LAMMPS = Path("/home/anson/LAMMPS/lammps/src/lmp_mpi")

# ---------------------------------------------------------------------------
# Simulation parameter table
# 10 configs × ~100 frames each = ~1000 images
# Vary: random seed, box size (unit cells), final heating temperature
# ---------------------------------------------------------------------------
CONFIGS = [
    # (sim_id, velocity_seed, box_x, box_y, box_z, final_temp_K)
    ("sim_01", 12345,  10, 10, 10, 2000),
    ("sim_02", 54321,  10, 10, 10, 2000),
    ("sim_03", 98765,  10, 10, 10, 2000),
    ("sim_04", 11111,  10, 10, 10, 1700),  # Lower T -> transition happens later -> more solid
    ("sim_05", 22222,  10, 10, 10, 2300),  # Higher T -> faster melt -> more liquid
    ("sim_06", 33333,   8,  8,  8, 2000),  # Smaller box (2048 atoms)
    ("sim_07", 44444,   8,  8,  8, 2000),
    ("sim_08", 55555,  12, 12, 12, 2000),  # Larger box (6912 atoms)
    ("sim_09", 66666,  12, 12, 12, 2000),
    ("sim_10", 77777,  10, 10, 10, 2000),
]

LAMMPS_TEMPLATE = """\
# Initialization
units           metal
dimension       3
boundary        p p p
atom_style      atomic

# System Definition
lattice         fcc 3.615
region          box block 0 {bx} 0 {by} 0 {bz}
create_box      1 box
create_atoms    1 box

# Interatomic Potential
pair_style      eam
pair_coeff      * * Cu_u3.eam

# Settings
neighbor        0.3 bin
neigh_modify    delay 0 every 1 check yes

# Equilibration (300 K)
velocity        all create 300 {seed} loop geom
fix             1 all npt temp 300 300 0.1 iso 0 0 1.0
thermo          1000
run             5000

# Production run: heat 300 K -> {final_temp} K
unfix           1
fix             2 all npt temp 300 {final_temp} 0.1 iso 0 0 1.0

dump            1 all custom 1000 melt.lammpstrj id type x y z

thermo_style    custom step temp pe etotal press vol
thermo          1000
run             100000
"""

# ---------------------------------------------------------------------------
# Step 1: Write LAMMPS input file
# ---------------------------------------------------------------------------
def write_lammps_input(run_dir: Path, seed: int, bx: int, by: int, bz: int, final_temp: int):
    content = LAMMPS_TEMPLATE.format(
        seed=seed, bx=bx, by=by, bz=bz, final_temp=final_temp
    )
    (run_dir / "in.melt_copper").write_text(content)
    # Copy EAM potential into run directory (LAMMPS reads from CWD)
    shutil.copy(EAM_FILE, run_dir / "Cu_u3.eam")

# ---------------------------------------------------------------------------
# Step 2: Run LAMMPS
# ---------------------------------------------------------------------------
def run_lammps(run_dir: Path, lammps_bin: Path, mpi_ranks: int) -> bool:
    traj = run_dir / "melt.lammpstrj"
    if traj.exists():
        print(f"  [skip] Trajectory already exists: {traj}")
        return True

    cmd = [
        "mpirun", "-n", str(mpi_ranks),
        str(lammps_bin),
        "-in", "in.melt_copper",
        "-log", "lammps.log",
    ]
    print(f"  Running: {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=run_dir, capture_output=False)
    if result.returncode != 0:
        print(f"  ERROR: LAMMPS failed (exit {result.returncode}). Check {run_dir}/lammps.log")
        return False
    print(f"  Done. Trajectory: {traj}")
    return True

# ---------------------------------------------------------------------------
# Step 3: Render trajectory with OVITO + PTM labeling
# ---------------------------------------------------------------------------
def render_trajectory(sim_id: str, traj_path: Path):
    pipeline = import_file(str(traj_path))
    ptm = PolyhedralTemplateMatchingModifier()
    pipeline.modifiers.append(ptm)
    pipeline.add_to_scene()

    vp = Viewport()
    vp.type = Viewport.Type.Perspective
    vp.camera_dir = (-1, -1, -1)
    vp.zoom_all()

    n_frames = pipeline.source.num_frames
    solid_count = 0
    liquid_count = 0

    for frame in range(n_frames):
        data = pipeline.compute(frame)
        fcc_count   = data.attributes.get("PolyhedralTemplateMatching.counts.FCC", 0)
        total_atoms = data.particles.count
        label       = "solid" if (fcc_count / total_atoms) > 0.5 else "liquid"

        out_dir  = DATA_SOLID if label == "solid" else DATA_LIQ
        filename = str(out_dir / f"{sim_id}_frame{frame:04d}.png")

        vp.render_image(
            filename=filename,
            size=(224, 224),
            frame=frame,
            renderer=TachyonRenderer(),
        )

        if label == "solid":
            solid_count += 1
        else:
            liquid_count += 1

        print(f"  Frame {frame:03d}/{n_frames-1} -> {label}  ({filename.split('/')[-1]})")

    pipeline.remove_from_scene()
    print(f"  Summary: {solid_count} solid, {liquid_count} liquid")
    return solid_count, liquid_count

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Generate PhaseNet training data")
    parser.add_argument("--lammps",   default=str(DEFAULT_LAMMPS), help="Path to lmp_mpi")
    parser.add_argument("--mpi",      type=int, default=4,          help="MPI ranks per sim")
    parser.add_argument("--sims",     type=int, default=10,         help="Number of sims (1-10)")
    parser.add_argument("--skip-sim", action="store_true",          help="Skip LAMMPS, only render")
    args = parser.parse_args()

    lammps_bin = Path(args.lammps)
    if not args.skip_sim and not lammps_bin.exists():
        print(f"ERROR: LAMMPS binary not found at {lammps_bin}")
        sys.exit(1)

    DATA_SOLID.mkdir(parents=True, exist_ok=True)
    DATA_LIQ.mkdir(parents=True, exist_ok=True)
    RUNS_DIR.mkdir(parents=True, exist_ok=True)

    configs = CONFIGS[: args.sims]
    total_solid = total_liquid = 0

    for sim_id, seed, bx, by, bz, final_temp in configs:
        print(f"\n{'='*60}")
        print(f"  {sim_id}  seed={seed}  box={bx}x{by}x{bz}  T_final={final_temp}K")
        print(f"{'='*60}")

        run_dir = RUNS_DIR / sim_id
        run_dir.mkdir(exist_ok=True)
        traj_path = run_dir / "melt.lammpstrj"

        # --- Run simulation ---
        if not args.skip_sim:
            write_lammps_input(run_dir, seed, bx, by, bz, final_temp)
            ok = run_lammps(run_dir, lammps_bin, args.mpi)
            if not ok:
                print(f"  Skipping render for {sim_id} due to LAMMPS failure.")
                continue

        if not traj_path.exists():
            print(f"  No trajectory at {traj_path}, skipping render.")
            continue

        # --- Render frames ---
        s, l = render_trajectory(sim_id, traj_path)
        total_solid += s
        total_liquid += l

    print(f"\n{'='*60}")
    print(f"  DONE")
    print(f"  Total solid images : {total_solid}")
    print(f"  Total liquid images: {total_liquid}")
    print(f"  Grand total        : {total_solid + total_liquid}")
    print(f"  Output dirs:")
    print(f"    {DATA_SOLID}")
    print(f"    {DATA_LIQ}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
