import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
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
        # Define tower positions in a 300 ft x 300 ft square
        tower_positions = {
            "Tower 1": np.array([0, 0]),
            "Tower 2": np.array([300, 0]),
            "Tower 3": np.array([300, 300]),
            "Tower 4": np.array([0, 300])
        }
        self.tower_positions = tower_positions
        self.simulate_tower_down = simulate_tower_down
        self.resolution = resolution
        self.simulator = None
        self.towers_for_multilateration = None
        self.estimated_position = None
        self.down_tower = None  # To store the tower that is down (if any)

        self.scanner = WifiTowerScanner(-36, 2)
        self.simulator = self.scanner.calculator

    # def generate_random_readings(self):
    #     """
    #     Generates random readings for all towers.
    #     """

    #     self.simulator.generate_random_readings(self.variance)

    def select_towers_for_multilateration(self):
        """
        Select towers to use for multilateration. Optionally simulates one tower being down.
        """

        self.towers_for_multilateration = self.simulator.towers.copy()
        for tower in self.towers_for_multilateration[::-1]:
            if len(self.simulator.get_readings(tower)) < self.simulator.max_readings:
                self.towers_for_multilateration.remove(tower)

    def multilaterate(self):
        """
        multilaterates the estimated device position using least squares optimization.
        Uses the towers selected (which might be fewer than four if one is down).
        """

        # Select the towers to use for multilateration.
        self.select_towers_for_multilateration()

        # Get and output the average distance for each tower.
        readings = {}
        for tower in self.towers_for_multilateration:
            rssi_reading = self.simulator.get_average_reading(tower)
            readings[tower] = rssi_reading
            print(f"{tower}: RSSI = {rssi_reading} dBm")

        # Get and output the average distance for each tower.
        distances = {}
        for tower in self.towers_for_multilateration:
            avg_distance = self.simulator.get_average_distance(tower)
            distances[tower] = avg_distance
            print(f"{tower}: Average Distance = {avg_distance:.2f} ft")
        
        # Define the residuals function for least squares optimization.
        def residuals(point):
            return [
                np.linalg.norm(point - self.tower_positions[tower]) - distances[tower]
                for tower in self.towers_for_multilateration
            ]
        
        # Guess the center then adjust with least squares.
        initial_guess = np.array([0, 0])
        result = least_squares(residuals, initial_guess)
        self.estimated_position = result.x
        print(f"Estimated position: {self.estimated_position}")
        return self.estimated_position

    def plot(self):
        fig = plt.figure(figsize=(10, 6))
        gs = fig.add_gridspec(1, 2, width_ratios=[3, 1], wspace=0.4)
        
        ax = fig.add_subplot(gs[0])
        ax.grid(True)
        tower_colors = ['green', 'darkcyan', 'navy', 'purple']

        number_adjustments = [[20, 20], [-10, 20], [-8, -8], [20, -10]]

        rssi_values = []
        tower_labels = []

        for i, tower in enumerate(self.simulator.towers):
            tower_number = int(tower.split()[-1]) - 1
            pos = self.tower_positions[tower]
            tower_down = len(self.simulator.get_readings(tower)) == 0

            if not tower_down:
                rssi_reading = self.simulator.get_average_reading(tower)
                avg_distance = self.simulator.get_average_distance(tower)
                color = tower_colors[i % len(tower_colors)]
                rssi_values.append(rssi_reading)
                tower_labels.append(f"Tower {tower_number + 1}")
            else:
                color = "grey"
                rssi_values.append(None)
                tower_labels.append(f"Tower {tower_number + 1}")

            ax.plot(pos[0], pos[1], 'o', markersize=8, color=color)
            ax.text(pos[0] + number_adjustments[tower_number][0],
                    pos[1] + number_adjustments[tower_number][1],
                    str(tower_number + 1), fontsize=12, color=color)

            if not tower_down:
                circle = plt.Circle((pos[0], pos[1]), avg_distance, color=color, fill=False)
                ax.add_artist(circle)

        if self.estimated_position is not None:
            ax.plot(self.estimated_position[0], self.estimated_position[1], 'ro', markersize=10)
            coords = f"Current PDU Coordinates:\nX: {int(self.estimated_position[0])} ft, Y: {int(self.estimated_position[1])} ft"
            ax.set_xlabel(f"\n{coords}")

        ax.set_title("Multilateration based on RSSI")
        secax_x = ax.secondary_xaxis('top')
        secax_y = ax.secondary_yaxis('right')
        secax_x.set_xlabel(f"X (ft)")
        secax_y.set_ylabel("Y (ft)")
        ax.set_xlim(-5, 305)
        ax.set_ylim(-5, 305)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.tick_params(axis="y", left=False, labelleft=False)
        secax_x.set_xlim(-5, 305)
        secax_y.set_ylim(-5, 305)
        secax_x.invert_xaxis()
        secax_y.invert_yaxis()
        ax.set_aspect('equal', 'box')

        # ======== BAR PLOT FOR RSSI VALUES ========
        ax_bar = fig.add_subplot(gs[1])

        rssi_min, rssi_max = -90, -20
        y_pos = np.arange(len(rssi_values))

        # Create a red-yellow-green colormap
        cmap = LinearSegmentedColormap.from_list("signal", ["red", "yellow", "green"])
        for i, rssi in enumerate(rssi_values):
            if rssi is not None:
                # Normalize from 0 (weak) to 1 (strong)
                norm_rssi = (rssi - rssi_min) / (rssi_max - rssi_min)
                color = cmap(norm_rssi)
                ax_bar.barh(i, norm_rssi, color=color, edgecolor='black', height=0.3)

                # Show actual RSSI value as label at the end of bar
                ax_bar.text(1.05, i, f"{rssi:.0f} dBm", va='center', ha='left', fontsize=9)

            else:
                ax_bar.barh(i, 1.05, color='lightgrey', edgecolor='black', hatch='//', height=0.3)
                ax_bar.text(1.05, i, "Down", va='center', fontsize=9, color='gray')

        ax_bar.set_yticks(y_pos)
        ax_bar.set_yticklabels(tower_labels)
        ax_bar.set_xlim(0, 1)
        ax_bar.invert_yaxis()
        ax_bar.set_title("Signal Strength")
        ax_bar.set_xlabel("Strength (Relative)")
        ax_bar.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
        ax_bar.set_xticklabels(["Very Weak\n(-90db)", "", "Moderate", "", "Very High\n(-20db)"])
        ax_bar.grid(True, axis='x')


        plt.savefig("multilateration.png")
        plt.close()

# Example usage:
def main():
    multilaterator = Multilateration(simulate_tower_down=True, resolution=1.0, variance=2.0)
    multilaterator.multilaterate()
    multilaterator.plot()

if __name__ == "__main__":
    main()
