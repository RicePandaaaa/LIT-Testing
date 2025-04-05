import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import least_squares
from rssi import RSSI
from rssi_values import WifiTowerScanner, RSSI_Calculator

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
        """
        Plots each tower as a colored dot (each tower a different color) with a circle 
        (radius = average distance) and the estimated device position as a red dot.
        The down tower (if any) does not have a circle drawn around it.
        Each tower is labeled with its number (e.g. 1 for Tower 1).
        """
        fig, ax = plt.subplots()
        ax.grid(True)
        tower_colors = ['green', 'darkcyan', 'navy', 'purple']

        number_adjustments = [[20, 20],
                              [-10, 20],
                              [-8, -8],
                              [20, -10]]

        # Plot each tower.
        for i, tower in enumerate(self.simulator.towers):
            tower_number = int(tower.split()[-1]) - 1
            pos = self.tower_positions[tower]

            # Check if a tower is down
            tower_down = len(self.simulator.get_readings(tower)) == 0

            # If tower is actually up
            if not tower_down:
                rssi_reading = self.simulator.get_average_reading(tower)
                avg_distance = self.simulator.get_average_distance(tower)
                color = tower_colors[i % len(tower_colors)]
            else:
                color = "grey"

            # Plot the tower as a dot.
            ax.plot(pos[0], pos[1], 'o', markersize=8, color=color)
            # Label the tower with just its number.
            tower_label = tower.split()[-1]  # e.g., "Tower 1" -> "1"
            ax.text(pos[0] + number_adjustments[tower_number][0], pos[1] + number_adjustments[tower_number][1], tower_label, fontsize=12, color=color)


            # Only draw the circle if the tower is up
            if not tower_down:
                circle = plt.Circle((pos[0], pos[1]), rssi_reading, linestyle="--", color=color, fill=False)
                ax.add_artist(circle)
                circle = plt.Circle((pos[0], pos[1]), avg_distance, color=color, fill=False, label="Distance (ft)")
                ax.add_artist(circle)
        
        # Plot the estimated position as a red dot.
        actual_coordinates = ""
        if self.estimated_position is not None:
            ax.plot(self.estimated_position[0], self.estimated_position[1], 'ro', markersize=10)
            actual_coordinates = f"Current PDU Coordinates:\nX: {int(self.estimated_position[0])} ft, Y: {int(self.estimated_position[1])} ft"

        ax.set_title(f"Multilateration based on RSSI\n")
        secax_x = ax.secondary_xaxis('top')
        secax_y = ax.secondary_yaxis('right')
        secax_x.set_xlabel(f"X (ft)")
        secax_y.set_ylabel("Y (ft)")
        ax.set_xlabel(f"\n{actual_coordinates}")
        ax.set_xlim(-5, 305)
        ax.set_ylim(-5, 305)
        ax.invert_xaxis()
        ax.invert_yaxis()
        ax.tick_params(axis="x", bottom=False, labelbottom=False)
        ax.tick_params(axis="y",left=False, labelleft=False)
        secax_x.set_xlim(-5, 305)
        secax_y.set_ylim(-5, 305)
        secax_x.invert_xaxis()
        secax_y.invert_yaxis()
        ax.legend(bbox_to_anchor=(1.2, 1.3), loc='upper left', fontsize=7, fancybox=True, shadow=True)
        ax.set_aspect('equal', 'box')


        # Save and show the plot.
        plt.tight_layout()
        plt.savefig("multilateration.png")
        # Uncomment to show the plot.
        #plt.show()

# Example usage:
def main():
    multilaterator = Multilateration(simulate_tower_down=True, resolution=1.0, variance=2.0)
    multilaterator.multilaterate()
    multilaterator.plot()

if __name__ == "__main__":
    main()
