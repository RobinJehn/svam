import numpy as np
import matplotlib.pyplot as plt
import random
import math
from copy import deepcopy

# Define the size of the forest grid
GRID_SIZE = 50

# Species traits
SPECIES_INFO = {
    "Oak": {
        "competition_rate": 4,
        "seed_dispersal_range": 2,
        "shade_tolerance": 9,
        "color": [0.13, 0.55, 0.13],  # Dark green
        "V_max": 5.0,  # Maximum volume in cubic meters
        "k": 0.1,  # Growth rate constant
        "A_mid": 50,  # Age at which volume is half of V_max
    },
    "Pine": {
        "competition_rate": 6,
        "seed_dispersal_range": 3,
        "shade_tolerance": 3,
        "color": [0.0, 0.5, 0.0],  # Green
        "V_max": 4.0,
        "k": 0.12,
        "A_mid": 40,
    },
    "Birch": {
        "competition_rate": 4.8,
        "seed_dispersal_range": 1,
        "shade_tolerance": 5,
        "color": [0.6, 0.8, 0.2],  # Light green
        "V_max": 3.0,
        "k": 0.15,
        "A_mid": 30,
    },
}

HARVEST_AGE_THRESHOLD = 50  # Minimum age for a tree to be eligible for harvest
HARVEST_PROBABILITY = 0.5  # Probability of harvesting a mature tree
HARVEST_RANDOM_PROBABILITY = 0.1  # Probability of random harvesting
CLEARCUTTING_AREA = 7  # Size of the clearcutting area


class Tree:
    def __init__(self, species, age: float = 0.0):
        self.species = species
        self.age = age
        self.competition_rate = SPECIES_INFO[species]["competition_rate"]
        self.shade_tolerance = SPECIES_INFO[species]["shade_tolerance"]
        # Volume parameters
        self.V_max = SPECIES_INFO[species]["V_max"]
        self.k = SPECIES_INFO[species]["k"]
        self.A_mid = SPECIES_INFO[species]["A_mid"]

    def grow(self, neighbor_count):
        # Growth affected by competition (more neighbors may slow growth)
        competition_factor = max(1, neighbor_count)
        self.age += self.competition_rate / competition_factor

    def get_volume(self):
        # Logistic growth model
        A = self.age
        V = self.V_max / (1 + math.exp(-self.k * (A - self.A_mid)))
        return V


def initialize_forest(density: float = 0.5):
    forest_grid = np.empty((GRID_SIZE, GRID_SIZE), dtype=object)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            if random.random() < density:
                species = random.choice(list(SPECIES_INFO.keys()))
                age = random.randint(1, 100)
                forest_grid[i][j] = Tree(species, age)
    return forest_grid


def get_neighbors(x, y, forest_grid):
    neighbors = []
    for dx in [-1, 0, 1]:
        for dy in [-1, 0, 1]:
            if dx != 0 or dy != 0:
                nx, ny = x + dx, y + dy
                if 0 <= nx < GRID_SIZE and 0 <= ny < GRID_SIZE:
                    neighbor = forest_grid[nx][ny]
                    if neighbor:
                        neighbors.append((neighbor, (nx, ny)))
    return neighbors


def simulate_growth(forest_grid):
    new_forest_grid = deepcopy(forest_grid)
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tree = forest_grid[i][j]
            if tree:
                neighbors = get_neighbors(i, j, forest_grid)
                tree.grow(len(neighbors))
                new_forest_grid[i][j] = tree
            else:
                # Seed dispersal from neighboring trees
                potential_trees = []
                for neighbor, (nx, ny) in get_neighbors(i, j, forest_grid):
                    species = neighbor.species
                    dispersal_range = SPECIES_INFO[species]["seed_dispersal_range"]
                    if (
                        abs(i - nx) <= dispersal_range
                        and abs(j - ny) <= dispersal_range
                    ):
                        potential_trees.append(species)
                if potential_trees:
                    # Chance for a new sapling to grow based on neighboring species
                    new_species = random.choice(potential_trees)
                    # Shade tolerance check
                    shade = len([n[0] for n in get_neighbors(i, j, forest_grid) if n])
                    shade_tolerance = SPECIES_INFO[new_species]["shade_tolerance"]
                    if shade < shade_tolerance:
                        new_forest_grid[i][j] = Tree(new_species, age=1)
    return new_forest_grid


def simulate_harvest(forest_grid, strategy: str, clearcutting_index: int):
    harvested_volume = 0
    if strategy == "selective":
        # Harvest trees that are above the age threshold
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                tree = forest_grid[i][j]
                if tree and tree.age >= HARVEST_AGE_THRESHOLD:
                    harvest_probability = HARVEST_PROBABILITY
                    if random.random() < harvest_probability:
                        harvested_volume += tree.get_volume()
                        forest_grid[i][j] = None
    elif strategy == "clearcutting":
        # Remove all trees in a specific area
        x_start = random.randint(0, GRID_SIZE - CLEARCUTTING_AREA)
        y_start = random.randint(0, GRID_SIZE - CLEARCUTTING_AREA)
        for i in range(x_start, min(x_start + CLEARCUTTING_AREA, GRID_SIZE)):
            for j in range(y_start, min(y_start + CLEARCUTTING_AREA, GRID_SIZE)):
                tree = forest_grid[i][j]
                if tree:
                    harvested_volume += tree.get_volume()
                    forest_grid[i][j] = None
    elif strategy == "random":
        # Randomly harvest trees
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                tree = forest_grid[i][j]
                if tree:
                    if random.random() < HARVEST_RANDOM_PROBABILITY:
                        harvested_volume += tree.get_volume()
                        forest_grid[i][j] = None
    elif strategy == "all":
        # Clearcut the entire forest
        for i in range(GRID_SIZE):
            for j in range(GRID_SIZE):
                tree = forest_grid[i][j]
                if tree:
                    harvested_volume += tree.get_volume()
                    forest_grid[i][j] = None
    elif strategy == "systematic_clearcutting":
        number_of_rows = math.ceil(GRID_SIZE / CLEARCUTTING_AREA)
        number_of_grids = number_of_rows * number_of_rows
        grid_col = clearcutting_index % number_of_rows
        grid_row = clearcutting_index // number_of_rows
        clearcutting_index = (clearcutting_index + 1) % number_of_grids

        x_start = grid_col * CLEARCUTTING_AREA
        y_start = grid_row * CLEARCUTTING_AREA
        # Clearcut the grid
        for i in range(x_start, min(x_start + CLEARCUTTING_AREA, GRID_SIZE)):
            for j in range(y_start, min(y_start + CLEARCUTTING_AREA, GRID_SIZE)):
                tree = forest_grid[i][j]
                if tree:
                    harvested_volume += tree.get_volume()
                    forest_grid[i][j] = None
    return forest_grid, harvested_volume, clearcutting_index


def get_max_tree_age(forest_grid):
    max_age = 0
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tree = forest_grid[i][j]
            if tree:
                max_age = max(max_age, tree.age)
    return max_age


def display_forest(forest_grid, step):
    max_tree_age = get_max_tree_age(forest_grid)
    image = np.zeros((GRID_SIZE, GRID_SIZE, 3))
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            tree = forest_grid[i][j]
            if tree:
                tree_age = tree.age
                color = SPECIES_INFO[tree.species]["color"]
                # Normalize age to a value between 0 and 1 for color intensity
                age_intensity = tree_age / max(max_tree_age, 100)
                image[i][j] = [c * age_intensity for c in color]
            else:
                image[i][j] = [0.9, 0.9, 0.9]  # Light gray for empty cells
    plt.figure(figsize=(6, 6))
    plt.imshow(image)
    plt.title(f"Forest State at Year {step}")
    plt.axis("off")
    plt.show()


def run_simulation(years=50, strategy="selective"):
    forest_grid = initialize_forest(density=0.6)
    clearcutting_index = 0
    total_harvested_volume = []
    for year in range(1, years + 1):
        forest_grid = simulate_growth(forest_grid)
        forest_grid, harvested_volume, clearcutting_index = simulate_harvest(
            forest_grid, strategy, clearcutting_index
        )
        total_harvested_volume.append(harvested_volume)
        if year % 10 == 0 or year == 1 or year == years:
            display_forest(forest_grid, year)
    print(
        f"Total harvested volume over {years} years: {sum(total_harvested_volume)} cubic meters"
    )


if __name__ == "__main__":
    # Run the simulation with a chosen harvesting strategy
    strategy = "systematic_clearcutting"  # Options: 'selective', 'clearcutting', 'all', 'random'
    run_simulation(years=100, strategy=strategy)
