import ovito
from ovito.data import DataCollection
from ovito.pipeline import StaticSource, Pipeline
from ovito.vis import Viewport
import numpy as np
import os

def create_fcc_pipeline(lattice_constant, count):
    """
    Creates an OVITO pipeline with a static source containing an FCC lattice.
    """
    a = lattice_constant
    nx, ny, nz = count
    
    # Basis for FCC (face-centered cubic)
    basis = np.array([[0,0,0], [0,0.5,0.5], [0.5,0,0.5], [0.5,0.5,0]]) * a
    
    # Generate grid of unit cell origins
    x = np.arange(nx)
    y = np.arange(ny)
    z = np.arange(nz)
    X, Y, Z = np.meshgrid(x, y, z, indexing='ij')
    
    # Shape: (nx, ny, nz, 3) -> (N_cells, 3)
    grid_points = np.stack([X,Y,Z], axis=-1).reshape(-1, 3) * a
    
    # Add basis vectors to each grid point
    # grid_points: (N_cells, 1, 3)
    # basis: (1, 4, 3)
    # result: (N_cells, 4, 3) -> (N_atoms, 3)
    all_positions = grid_points[:, None, :] + basis[None, :, :]
    all_positions = all_positions.reshape(-1, 3)
    
    # Create DataCollection
    data = DataCollection()
    particles = data.create_particles(count=len(all_positions))
    particles.create_property('Position', data=all_positions)
    
    # Create Simulation Cell
    # Note: For strict periodicity/rendering, cell should be established.
    cell_matrix = np.diag([nx*a, ny*a, nz*a])
    data.create_cell(cell_matrix, pbc=(True, True, True))
    
    # Create Pipeline with StaticSource
    pipeline = Pipeline(source=StaticSource(data=data))
    return pipeline

def create_liquid_pipeline(num_atoms, box_size):
    """
    Creates an OVITO pipeline with a static source containing a random liquid-like configuration.
    """
    # Generate random positions within the box
    positions = np.random.uniform(0, box_size, size=(num_atoms, 3))
    
    # Create DataCollection
    data = DataCollection()
    particles = data.create_particles(count=len(positions))
    particles.create_property('Position', data=positions)
    
    # Create Simulation Cell
    cell_matrix = np.diag([box_size, box_size, box_size])
    data.create_cell(cell_matrix, pbc=(True, True, True))
    
    # Create Pipeline with StaticSource
    pipeline = Pipeline(source=StaticSource(data=data))
    return pipeline

def generate_ovito_snapshot(filename="ovito_fcc_test.png", structure_type="solid"):
    # 1. Create pipeline based on type
    if structure_type == "solid":
        # Create a clean Copper FCC lattice (3.615 A is standard for Cu)
        pipeline = create_fcc_pipeline(lattice_constant=3.615, count=(6, 6, 6))
    elif structure_type == "liquid":
        # Approximate density and volume for liquid
        # 6x6x6 FCC has 4 * 6^3 = 864 atoms. 
        # Box size is ~ 6 * 3.615 = 21.69 A.
        pipeline = create_liquid_pipeline(num_atoms=864, box_size=21.69)
    else:
        raise ValueError(f"Unknown structure_type: {structure_type}")

    pipeline.add_to_scene()
    
    # 2. Style the atoms to match your training data
    # Modify the source data directly for StaticSource to ensure visualization applies
    data = pipeline.source.data
    data.particles.vis.radius = 1.4
    
    # Choose color based on type
    if structure_type == "solid":
        color = (0.4, 1.0, 0.4) # Light green
    else:
        color = (1.0, 1.0, 1.0) # White for liquid
        
    # Use Color property
    color_data = np.tile(color, (data.particles.count, 1))
    data.particles.create_property("Color", data=color_data)
    
    # 3. Setup Viewport (Perspective to match frame_0005.png)
    vp = Viewport(
        type=Viewport.Type.Perspective,
        camera_dir=(-1, -1, -1), # Tilted view
        camera_pos=(10, 10, 10)
    )
    
    # 4. Render exactly 224x224
    vp.zoom_all()
    
    # Ensure directory exists
    os.makedirs(os.path.dirname(filename) or '.', exist_ok=True)
    
    # Render
    vp.render_image(
        filename=filename, 
        size=(224, 224), 
        background=(1, 1, 1), # White background
        renderer=ovito.vis.TachyonRenderer() # Uses the real 3D shaders
    )
    print(f"Generated true OVITO snapshot: {filename}")

if __name__ == "__main__":
    output_dir = os.path.expanduser("~/PhaseNet/dummy_data")
    generate_ovito_snapshot(os.path.join(output_dir, "ovito_solid_test.png"), structure_type="solid")
    generate_ovito_snapshot(os.path.join(output_dir, "ovito_liquid_test.png"), structure_type="liquid")