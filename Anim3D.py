import numpy as np
import trimesh
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation
from mpl_toolkits.mplot3d import Axes3D
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial.transform import Rotation as R

class DroneVisualizer3D:
    def __init__(self, step_file_path, thrust_scale=0.1, axis_limits=(-5, 5)):
        """
        Initialize the 3D drone visualizer with a STEP file model.
        
        Parameters:
            step_file_path: Path to .step/.stp CAD file
            thrust_scale: Scaling factor for thrust visualization
            axis_limits: Tuple (min, max) for all axes
        """
        self.fig = plt.figure(figsize=(10, 8))
        self.ax = self.fig.add_subplot(111, projection='3d')
        self.thrust_scale = thrust_scale
        self.axis_limits = axis_limits
        
        # Load and prepare drone model
        self.drone_mesh = self.load_step_file(step_file_path)
        self.drone_artist = None
        self.thrust_lines = [self.ax.plot([], [], [], 'r-', lw=2)[0] for _ in range(4)]
        
        self.setup_axes()
        
    def load_step_file(self, file_path):
        """Load and prepare STEP file for visualization."""
        mesh = trimesh.load(file_path)
        
        # Center and scale the model if needed
        mesh.apply_translation(-mesh.center_mass)
        mesh.apply_scale(0.001)  # Adjust scale if model is too large/small
        
        return mesh

    def setup_axes(self):
        """Configure 3D axes settings."""
        self.ax.set_xlim(self.axis_limits)
        self.ax.set_ylim(self.axis_limits)
        self.ax.set_zlim(self.axis_limits)
        self.ax.set_xlabel('X (m)')
        self.ax.set_ylabel('Y (m)')
        self.ax.set_zlabel('Z (m)')
        self.ax.set_title('3D Drone Animation with CAD Model')
        self.ax.grid(True)
        self.ax.view_init(elev=20, azim=45)

    def update_drone_pose(self, x, y, z, phi, theta, psi):
        """Update drone position and orientation."""
        # Clear previous drone artist
        if self.drone_artist is not None:
            self.drone_artist.remove()
        
        # Create rotation and translation
        rotation = R.from_euler('xyz', [phi, theta, psi]).as_matrix()
        translation = np.array([x, y, z])
        
        # Transform vertices
        vertices = np.dot(self.drone_mesh.vertices, rotation.T) + translation
        
        # Create new artist
        faces = self.drone_mesh.faces
        self.drone_artist = Poly3DCollection(
            vertices[faces],
            alpha=0.8,
            linewidths=1,
            edgecolor='k',
            facecolor='lightgray'
        )
        self.ax.add_collection3d(self.drone_artist)
        
        return vertices

    def update_thrust_vectors(self, motor_positions, thrust_values):
        """Update thrust vector visualization."""
        for i in range(4):
            if i < len(thrust_values):
                thrust = thrust_values[i] * self.thrust_scale
                self.thrust_lines[i].set_data(
                    [motor_positions[i][0], motor_positions[i][0]],
                    [motor_positions[i][1], motor_positions[i][1]]
                )
                self.thrust_lines[i].set_3d_properties(
                    [motor_positions[i][2], motor_positions[i][2] - thrust]
                )

    def update_animation(self, frame, drone_state_history, motor_thrust_history):
        """Update function for animation frames."""
        if frame >= len(drone_state_history):
            return []
            
        state = drone_state_history[frame]
        x, y, z = state[0], state[1], state[2]
        phi, theta, psi = state[6], state[7], state[8]
        
        # Update drone pose
        vertices = self.update_drone_pose(x, y, z, phi, theta, psi)
        
        # Estimate motor positions (adjust based on your CAD model)
        motor_positions = [
            vertices[0],  # Replace with actual motor vertex indices
            vertices[1],
            vertices[2],
            vertices[3]
        ]
        
        # Update thrust vectors
        if frame < len(motor_thrust_history):
            self.update_thrust_vectors(motor_positions, motor_thrust_history[frame])
        
        return [self.drone_artist] + self.thrust_lines

    def animate(self, drone_state_history, motor_thrust_history, dt, save_path=None):
        """Run the animation."""
        ani = FuncAnimation(
            self.fig, 
            lambda i: self.update_animation(i, drone_state_history, motor_thrust_history),
            frames=len(drone_state_history),
            interval=dt*1000,
            blit=True,
            repeat=False
        )
        
        if save_path:
            try:
                ani.save(save_path, writer='ffmpeg', fps=int(1/dt), dpi=300)
            except Exception as e:
                print(f"Could not save animation: {e}")
        
        plt.show()
        return ani