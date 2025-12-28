# utils/tsp_loader.py

import os
import math
import urllib.request

TSPLIB_BASE_URL = "https://raw.githubusercontent.com/mastqe/tsplib/master/"


class TSPInstance:
    """
    Lightweight TSP instance for GA solver
    """

    def __init__(self, name, coords):
        """
        Parameters
        ----------
        name : str
            Instance name
        coords : list of (x, y)
            City coordinates
        """
        self.name = name
        self.coords = coords
        self.num_cities = len(coords)
        self.distance_matrix = self._compute_distance_matrix()

    def _compute_distance_matrix(self):
        n = self.num_cities
        dist = [[0.0] * n for _ in range(n)]

        for i in range(n):
            x1, y1 = self.coords[i]
            for j in range(i + 1, n):
                x2, y2 = self.coords[j]
                d = math.hypot(x1 - x2, y1 - y2)
                dist[i][j] = d
                dist[j][i] = d

        return dist

    def evaluate(self, tour):
        """
        Compute total tour length

        Parameters
        ----------
        tour : iterable of int
            A permutation of city indices

        Returns
        -------
        float
            Total tour length
        """
        length = 0.0
        n = len(tour)
        for i in range(n):
            a = tour[i]
            b = tour[(i + 1) % n]
            length += self.distance_matrix[a][b]
        return length

    def __repr__(self):
        return f"<TSPInstance {self.name}, cities={self.num_cities}>"


# -------------------------------------------------
# TSPLIB loading & downloading
# -------------------------------------------------

def download_tsp(filename, save_path):
    """
    Download a .tsp file from TSPLIB if not present
    """
    url = TSPLIB_BASE_URL + filename
    print(f"[INFO] Downloading {url}")
    urllib.request.urlretrieve(url, save_path)
    print(f"[INFO] Saved to {save_path}")


def parse_tsp_file(filepath):
    """
    Minimal TSPLIB parser (supports EUC_2D only)
    """
    name = None
    dimension = None
    edge_weight_type = None
    coords = []

    with open(filepath, "r", encoding="utf-8") as f:
        lines = f.readlines()

    reading_coords = False

    for line in lines:
        line = line.strip()
        if not line:
            continue

        if line.startswith("NAME"):
            name = line.split(":", 1)[1].strip()

        elif line.startswith("DIMENSION"):
            dimension = int(line.split(":", 1)[1])

        elif line.startswith("EDGE_WEIGHT_TYPE"):
            edge_weight_type = line.split(":", 1)[1].strip()

        elif line.startswith("NODE_COORD_SECTION"):
            reading_coords = True

        elif line.startswith("EOF"):
            break

        elif reading_coords:
            parts = line.split()
            if len(parts) >= 3:
                _, x, y = parts[:3]
                coords.append((float(x), float(y)))

    if edge_weight_type != "EUC_2D":
        raise NotImplementedError(
            f"EDGE_WEIGHT_TYPE '{edge_weight_type}' is not supported"
        )

    if dimension is not None and len(coords) != dimension:
        raise ValueError(
            f"DIMENSION mismatch: expected {dimension}, got {len(coords)}"
        )

    return name, coords


def load_tsp(path):
    """
    Load a TSPLIB .tsp file.
    If the file does not exist, it will be downloaded automatically.
    """
    directory = os.path.dirname(path)
    if directory:
        os.makedirs(directory, exist_ok=True)

    filename = os.path.basename(path)

    if not os.path.exists(path):
        download_tsp(filename, path)

    name, coords = parse_tsp_file(path)
    tsp = TSPInstance(name, coords)

    print(f"[INFO] Loaded {tsp}")
    return tsp
