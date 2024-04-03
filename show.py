import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.animation import FuncAnimation


def plot_joints_anim(data):
    # Normalizing the data
    def normalize(data):
        min_val = data.min(axis=(0, 1), keepdims=True)
        max_val = data.max(axis=(0, 1), keepdims=True)
        return (data - min_val) / (max_val - min_val)

    normalized_data = normalize(data)

    # Set up the figure and 3D axis
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])
    ax.set_zlim([0, 1])

    # Initialize a plot which we'll update during each frame
    plot = ax.scatter(normalized_data[0,:,0], normalized_data[0,:,1], normalized_data[0,:,2])

    # Update function for animation
    def update(frame):
        plot._offsets3d = (normalized_data[frame,:,0], normalized_data[frame,:,1], normalized_data[frame,:,2])
        return plot,

    # Create animation
    ani = FuncAnimation(fig, update, frames=142, interval=50, blit=True)

    plt.show()

if __name__ == "__main__":
    # Sample data: Replace this with your actual data
    data = np.load("/home/tony/local-git-repo/HumanML3D/pose_data/EKUT/125/SLP101_poses.npy")
    plot_joints_anim(data)