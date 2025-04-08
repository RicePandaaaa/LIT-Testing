import rssi
import random
import time
import numpy as np
from collections import deque

class WifiTowerScanner:
    def __init__(self, RSSI_naught: float, n: float):
        # Mapping of MAC addresses to Tower names
        self.tower_map = {
            "94:2a:6f:22:d1:7c": "Tower 1",
            "9a:2a:6f:22:d6:77": "Tower 2",
            "9a:2a:6f:24:9f:09": "Tower 3",
            "9a:2a:6f:22:a2:7e": "Tower 4"
        }

        self.calculator = RSSI_Calculator(RSSI_naught, n)
        self.sampling_period = 5  # Seconds

    def get_tower_signal(self):
        start_time = time.time()
        readings = {tower: [] for tower in self.calculator.towers}

        while time.time() - start_time < self.sampling_period:
            # Simulate getting RSSI readings
            for i, tower in enumerate(self.calculator.towers):
                readings[tower].append(random.randint(-80, -20))
            time.sleep(0.1)  # Simulate a short delay between readings

        for tower in self.calculator.towers:
            self.calculator.add_readings_and_distances(tower, readings[tower])


class RSSI_Calculator:
    def __init__(self, RSSI_NAUGHT: float, n: float):
        """
        Initialize the RSSI_Values class with a RSSI instance and a
        dictionary of tower RSSI distances to device
        """
        self.RSSI = rssi.RSSI(RSSI_NAUGHT, n)
        self.towers = ["Tower 1", "Tower 2", "Tower 3", "Tower 4"]

        # RSSI readings from each tower
        self.readings = {
            "Tower 1": deque(maxlen=100),
            "Tower 2": deque(maxlen=100),
            "Tower 3": deque(maxlen=100),
            "Tower 4": deque(maxlen=100),
        }

        # Distances from each tower to the device
        self.distances = {
            "Tower 1": deque(maxlen=100),
            "Tower 2": deque(maxlen=100),
            "Tower 3": deque(maxlen=100),
            "Tower 4": deque(maxlen=100),
        }


    """
    *******************************************
    Functions for adding readings and distances
    *******************************************
    """

    def add_reading_and_distance(self, tower: str, reading: float):
        """
        Add a reading to the corresponding tower and calculate the distance.

        Args:
            tower (str): The tower to add the reading to.
            reading (float): The reading to add.
        """

        self.readings[tower].append(reading)
        self.add_distance(tower, reading)

    def add_readings_and_distances(self, tower: str, readings: list[float]):
        """
        Add a list of readings to the corresponding tower and calculate the distances.

        Args:
            tower (str): The tower to add the readings to.
            readings (list[float]): The readings to add.
        """
        # Filter outliers from readings before adding them
        filtered_readings = self.filter_outliers(readings)
        self.readings[tower].extend(filtered_readings)
        self.add_distances(tower, filtered_readings)

    def add_distance(self, tower: str, reading: float):
        """
        Helper function to add a distance to the corresponding tower.

        Args:
            tower (str): The tower to add the distance to.
            reading (float): The rssi reading
        """

        self.distances[tower].append(self.RSSI.get_distance(reading))

    def add_distances(self, tower: str, readings: list[float]):
        """
        Helper function to add distances to the corresponding tower.

        Args:
            tower (str): The tower to add the distance to.
            readings (list[float]): The rssi readings
        """
        self.distances[tower].extend([self.RSSI.get_distance(reading) for reading in readings])


    """
    *******************************************************************
    Functions for getting all readings and distances for a single tower
    *******************************************************************
    """

    def get_readings(self, tower: str) -> list[float]:
        """
        Get the readings from the corresponding tower.

        Args:
            tower (str): The tower to get the reading from.

        Returns:
            float: The reading from the corresponding tower.
        """

        return list(self.readings[tower])

    def get_distances(self, tower: str) -> list[float]:
        """
        Get the distance from the corresponding tower.

        Args:
            tower (str): The tower to get the distance from.

        Returns:
            float: The distance from the corresponding tower.
        """

        return list(self.distances[tower])


    """
    *************************************************************************
    Functions for getting the average reading and distance for a single tower
    *************************************************************************
    """

    def get_average_reading(self, tower: str) -> float:
        """
        Get the average reading from the corresponding tower.

        Args:
            tower (str): The tower to get the average reading from.

        Returns:
            float: The average reading from the corresponding tower.
        """

        if self.readings[tower]:
            return sum(self.readings[tower]) / len(self.readings[tower])
        return 0

    def get_average_distance(self, tower: str) -> float:
        """
        Get the average distance from the corresponding tower.

        Args:
            tower (str): The tower to get the average distance from.

        Returns:
            float: The average distance from the corresponding tower.
        """
        if self.distances[tower]:
            return sum(self.distances[tower]) / len(self.distances[tower])
        return 0
    
    """
    ********************************
    Functions for filtering outliers
    ********************************
    """

    def filter_outliers(self, data: list[float], method: str = 'z_score', threshold: float = 2) -> list[float]:
        """
        Filters outliers from a list of RSSI readings.

        Args:
            data (list[float]): The list of RSSI readings.
            method (str, optional): The outlier detection method to use.
                                    Options: 'z_score', 'iqr'. Defaults to 'z_score'.
            threshold (float, optional): The threshold for outlier detection. Defaults to 2.

        Returns:
            list[float]: The list of RSSI readings with outliers filtered out.
        """

        if not data:
            return []

        if method == 'z_score':
            return self.z_score_filter(data, threshold)
        elif method == 'iqr':
            return self.iqr_filter(data, threshold)
        else:
            raise ValueError(f"Invalid outlier detection method: {method}")

    def z_score_filter(self, data: list[float], threshold: float) -> list[float]:
        """
        Filters outliers based on the Z-score.
        """
        avg = np.mean(data)
        stdev = np.std(data)
        return [x for x in data if stdev == 0 or abs((x - avg) / stdev) < threshold]

    def iqr_filter(self, data: list[float], threshold: float) -> list[float]:
        """
        Filters outliers based on the Interquartile Range (IQR).
        """
        q1, q3 = np.percentile(data, [25, 75])
        iqr = q3 - q1
        lower_bound = q1 - threshold * iqr
        upper_bound = q3 + threshold * iqr
        return [x for x in data if lower_bound <= x <= upper_bound]

    """
    *******************************
    Functions for testing the class
    *******************************
    """

    def __repr__(self) -> str:
        """
        Returns a string representation of the RSSI_Calculator object to
        show all readings, distances, and average readings and distances
        for all towers.
        """
        header_string = f"RSSI_Calculator(RSSI_NAUGHT={self.RSSI.RSSI_NAUGHT}, n={self.RSSI.n})"

        for tower in self.towers:
            tower_string = f"{tower}:\n" \
                f"Readings: {list(self.readings[tower])}\n" \
                f"Distances: {list(self.distances[tower])}\n" \
                f"Average Reading: {self.get_average_reading(tower)}\n" \
                f"Average Distance: {self.get_average_distance(tower)}\n"

            header_string += "\n" + tower_string

        return header_string