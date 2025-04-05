import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import matplotlib.patches as patches
from scipy.optimize import least_squares
from rssi_values import WifiTowerScanner

class Multilateration:
    def __init__(self, simulate_tower_down=True, resolution=1.0):
        """
        Initializes the multilateration class.

        Args:
            simulate_tower_down (bool): If True, randomly remove one tower from multilateration.
            resolution (float): Grid resolution (in feet) for checking common intersection.
        """
        tower_positions = {
            "Tower 1": np.array([0, 0]),
            "Tower 2": np.array([300, 0]),
            "Tower 3": np.array([300, 300]),
            "Tower 4": np.array([0, 300])
        }
        self.tower_positions = tower_positions
        self.resolution = resolution
        self.scanner = WifiTowerScanner(-40, 2)
        self.simulator = self.scanner.calculator
        self.towers_for_multilateration = None
        self.estimated_position = None
        self.down_tower = None
        self.ani = None
        self.rings = []
        self.static_circles = []  # To store static distance circles
        self.fig = None
        self.ax = None
        self.est_pos_artist = None
        self.coord_text = None

        # --- Animation Parameters ---  
        self.MAX_RADIUS_PULSE = 40
        self.ANIMATION_SPEED = 2
        self.RSSI_MIN = -90
        self.RSSI_MAX = -20
        self.ANIMATION_INTERVAL = 50

    def select_towers_for_multilateration(self):
        """
        Select towers that have sufficient readings.
        """
        self.towers_for_multilateration = []
        self.down_tower = None
        all_towers = list(self.tower_positions.keys())

        active_towers = []
        down_towers = []

        for tower in all_towers:
            if len(self.simulator.get_readings(tower)) > 0:
                active_towers.append(tower)
            else:
                down_towers.append(tower)

        self.towers_for_multilateration = active_towers
        if down_towers:
            self.down_tower = down_towers

        print(f"Active towers for multilateration: {self.towers_for_multilateration}")
        if self.down_tower:
            print(f"Down towers: {self.down_tower}")

    def multilaterate(self):
        """
        Calculates the estimated device position using least squares optimization.
        Uses the towers selected.
        """
        self.select_towers_for_multilateration()

        if len(self.towers_for_multilateration) < 3:
            print("Not enough active towers (<3) for multilateration.")
            self.estimated_position = None
            return self.estimated_position

        readings = {}
        distances = {}
        print("\n--- Readings & Distances ---")
        for tower in self.towers_for_multilateration:
            rssi_reading = self.simulator.get_average_reading(tower)
            readings[tower] = rssi_reading
            print(f"{tower}: RSSI = {rssi_reading:.2f} dBm")

            avg_distance = self.simulator.get_average_distance(tower)
            if avg_distance > 1000:
                print(f"{tower}: Ignoring unreasonable distance {avg_distance:.2f} ft")
            else:
                distances[tower] = avg_distance
                print(f"{tower}: Average Distance = {avg_distance:.2f} ft")
        print("---------------------------\n")

        towers_with_valid_distance = list(distances.keys())
        if len(towers_with_valid_distance) < 3:
            print("Not enough towers with valid distances (<3) for multilateration.")
            self.estimated_position = None
            return self.estimated_position

        def residuals(point):
            return [
                np.linalg.norm(point - self.tower_positions[tower]) - distances[tower]
                for tower in towers_with_valid_distance
            ]

        initial_guess = np.mean([self.tower_positions[tower] for tower in towers_with_valid_distance], axis=0)
        bounds = ([-50, -50], [350, 350])

        try:
            result = least_squares(residuals, initial_guess, bounds=bounds)
            if result.success:
                self.estimated_position = result.x
                print(f"Estimated position (X, Y): ({self.estimated_position[0]:.2f}, {self.estimated_position[1]:.2f}) ft")
            else:
                print(f"Least squares optimization failed: {result.message}")
                self.estimated_position = None
        except Exception as e:
            print(f"Error during least squares optimization: {e}")
            self.estimated_position = None

        return self.estimated_position

    def _setup_plot_elements(self):
        """Sets up the static parts of the plot and initial ring patches."""
        if self.fig:
            plt.close(self.fig)

        self.fig, self.ax = plt.subplots()
        self.ax.grid(True)
        tower_marker_colors = {
            "Tower 1": 'green', "Tower 2": 'darkcyan',
            "Tower 3": 'navy', "Tower 4": 'purple'
        }
        number_adjustments = {
            "Tower 1": [20, 20], "Tower 2": [-10, 20],
            "Tower 3": [-8, -8], "Tower 4": [20, -10]
        }

        # Colormap from red to green
        cmap = plt.cm.get_cmap('RdYlGn') # '_r' reverses the default
        norm = plt.Normalize(vmin=self.RSSI_MIN, vmax=self.RSSI_MAX)

        self.rings = []
        self.static_circles = []
        active_tower_data = []

        for tower_name in self.tower_positions:
            pos = self.tower_positions[tower_name]
            is_down = tower_name not in self.towers_for_multilateration

            if not is_down:
                rssi = self.simulator.get_average_reading(tower_name)
                rssi_clamped = np.clip(rssi, self.RSSI_MIN, self.RSSI_MAX)
                ring_color = cmap(norm(rssi_clamped))
                marker_color = tower_marker_colors.get(tower_name, 'black')
                active_tower_data.append({'name': tower_name, 'pos': pos, 'color': ring_color})

                # Add static distance circle
                avg_distance = self.simulator.get_average_distance(tower_name)
                if avg_distance > 0 and avg_distance < 1000: # Basic sanity check
                    static_circle = patches.Circle(pos, radius=avg_distance, edgecolor='gray',
                                                    facecolor='none', linestyle='--', alpha=0.6)
                    self.ax.add_patch(static_circle)
                    self.static_circles.append(static_circle)

            else:
                marker_color = "grey"

            self.ax.plot(pos[0], pos[1], 's', markersize=8, color=marker_color)
            tower_label = tower_name.split()[-1]
            adj = number_adjustments.get(tower_name, [10, 10])
            self.ax.text(pos[0] + adj[0], pos[1] + adj[1], tower_label,
                         fontsize=12, color=marker_color)

        # Create initial pulsing ring patches
        for data in active_tower_data:
            ring = patches.Circle(data['pos'], radius=0.1, edgecolor=data['color'],
                                 facecolor='none', linewidth=1.5)
            self.ax.add_patch(ring)
            self.rings.append(ring)

        actual_coordinates_text = ""
        self.est_pos_artist = None
        if self.estimated_position is not None:
            line = self.ax.plot(self.estimated_position[0], self.estimated_position[1],
                                'ro', markersize=10, label='Estimated Position')
            self.est_pos_artist = line[0]
            actual_coordinates_text = f"Est. PDU Coordinates:\nX: {int(self.estimated_position[0])} ft, Y: {int(self.estimated_position[1])} ft"
        else:
            actual_coordinates_text = "Est. PDU Coordinates:\nNot Available"

        self.ax.set_title(f"Multilateration based on RSSI (Animated)\n")
        secax_x = self.ax.secondary_xaxis('top')
        secax_y = self.ax.secondary_yaxis('right')
        secax_x.set_xlabel(f"X (ft)")
        secax_y.set_ylabel("Y (ft)")
        self.coord_text = self.ax.text(0.5, -0.1, actual_coordinates_text,
                                     transform=self.ax.transAxes,
                                     ha='center', va='top', fontsize=9)

        self.ax.set_xlim(-5, 305)
        self.ax.set_ylim(-5, 305)
        self.ax.invert_xaxis()
        self.ax.invert_yaxis()

        self.ax.tick_params(axis="x", bottom=False, labelbottom=False)
        self.ax.tick_params(axis="y", left=False, labelleft=False)
        secax_x.set_xlim(self.ax.get_xlim())
        secax_y.set_ylim(self.ax.get_ylim())
        secax_x.invert_xaxis()
        secax_y.invert_yaxis()

        self.ax.set_aspect('equal', 'box')
        self.fig.tight_layout(rect=[0, 0.05, 1, 0.95])

    def _init_animation(self):
        """Initializes the animation."""
        for ring in self.rings:
            ring.set_radius(0.1)
            ring.set_alpha(1.0)
        if self.est_pos_artist:
            self.est_pos_artist.set_data([], [])
            return self.rings + [self.est_pos_artist]
        else:
            return self.rings

    def _update_animation(self, frame):
        """Updates the animation for each frame."""
        pulse_progress = (frame * self.ANIMATION_SPEED) % self.MAX_RADIUS_PULSE
        current_radius = pulse_progress
        fade_start_radius = self.MAX_RADIUS_PULSE / 4
        if current_radius > fade_start_radius:
            alpha_progress = (current_radius - fade_start_radius) / (self.MAX_RADIUS_PULSE - fade_start_radius)
            current_alpha = max(0, 1.0 - alpha_progress)
        else:
            current_alpha = 1.0

        for ring in self.rings:
            ring.set_radius(current_radius)
            ring.set_alpha(current_alpha)

        artists_to_return = list(self.rings)
        if self.est_pos_artist:
            if self.estimated_position is not None:
                self.est_pos_artist.set_data([self.estimated_position[0]], [self.estimated_position[1]])
                self.est_pos_artist.set_visible(True)
                coord_txt = f"Est. PDU Coordinates:\nX: {int(self.estimated_position[0])} ft, Y: {int(self.estimated_position[1])} ft"
                self.coord_text.set_text(coord_txt)
            else:
                self.est_pos_artist.set_visible(False)
                self.coord_text.set_text("Est. PDU Coordinates:\nNot Available")

            artists_to_return.extend([self.est_pos_artist, self.coord_text])

        return artists_to_return

    def run_animation(self, frames=60):
        """Sets up the plot and starts the animation.

        Args:
            frames (int): Number of frames to run the animation for.
                          Set to None for continuous running.
        """
        if self.estimated_position is None:
            print("Warning: No estimated position calculated yet. Running multilaterate()...")
            self.multilaterate()

        self._setup_plot_elements()

        self.ani = animation.FuncAnimation(
            self.fig,
            self._update_animation,
            init_func=self._init_animation,
            frames=frames,
            interval=self.ANIMATION_INTERVAL,
            blit=True,
            repeat=True
        )

    def save_animation(self, filename="multilateration_animation.gif", fps=60):
        """Saves the animation to a file (e.g., GIF or MP4)."""
        if self.ani:
            if fps is None:
                fps = 1000 / self.ANIMATION_INTERVAL
            if not self.fig:
                self._setup_plot_elements()

            print(f"Saving animation to {filename} (FPS: {fps})...")
            try:
                writer = 'pillow' if filename.lower().endswith('.gif') else 'ffmpeg'
                self.ani.save(filename, writer=writer, fps=fps)
                print("Save complete.")
            except Exception as e:
                print(f"Error saving animation: {e}")
                print(f"Ensure you have a suitable writer installed (e.g., 'pip install pillow' for GIFs, or ffmpeg for MP4)")
                print(f"Attempted writer: {writer}")
        else:
            print("Animation not generated yet. Run run_animation() first.")

# Example usage:
def main():
    multilaterator = Multilateration(simulate_tower_down=False)
    multilaterator.multilaterate()
    multilaterator.run_animation(frames=200)
    # multilaterator.save_animation("my_animation.gif")

if __name__ == "__main__":
    main()