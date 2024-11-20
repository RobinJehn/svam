import networkx as nx
import matplotlib.pyplot as plt
import random
from networkx.classes import Graph
import copy
import numpy as np


def simulate_sir(
    graph: Graph, alpha: float, gamma: float, initial_infected: list, max_time: int
):
    """
    Simulate the SIR model on a given graph.

    Parameters:
        graph: NetworkX graph
        alpha: infection rate
        gamma: recovery rate
        initial_infected: list of initially infected nodes
        max_time: maximum number of time steps to simulate

    Returns:
        S: list of number of susceptible individuals over time
        I: list of number of infected individuals over time
        R: list of number of recovered individuals over time
    """
    # Initialize node statuses: S for susceptible, I for infected, R for recovered
    status = {node: "S" for node in graph.nodes()}
    for node in initial_infected:
        status[node] = "I"

    S_history = []
    I_history = []
    R_history = []

    for t in range(max_time):
        new_status = status.copy()
        for node in graph.nodes():
            if status[node] == "I":
                # Infected node may recover
                if random.random() < gamma:
                    new_status[node] = "R"
                else:
                    # Infected node may infect its susceptible neighbors
                    neighbors = list(graph.neighbors(node))
                    for neighbor in neighbors:
                        if status[neighbor] == "S" and random.random() < alpha:
                            new_status[neighbor] = "I"
        status = new_status.copy()

        # Count S, I, R
        S_count = list(status.values()).count("S")
        I_count = list(status.values()).count("I")
        R_count = list(status.values()).count("R")

        S_history.append(S_count)
        I_history.append(I_count)
        R_history.append(R_count)

        # Stop if no more infected individuals
        if I_count == 0:
            break

    return S_history, I_history, R_history


def create_social_distancing_graph(graph: Graph, fraction_to_remove: float) -> Graph:
    sd_graph = copy.deepcopy(graph)

    # Remove a fraction of edges to simulate social distancing
    edges = list(sd_graph.edges())
    num_edges_to_remove = int(len(edges) * fraction_to_remove)
    edges_to_remove = random.sample(edges, num_edges_to_remove)

    # Remove edges
    sd_graph.remove_edges_from(edges_to_remove)
    return sd_graph


def init_graphs(N: int, k: int, p_rewire: float, fraction_to_remove: float):
    """
    Initialize two graphs: a Watts-Strogatz small-world graph and a social distancing graph.

    Parameters:
        N: The number of nodes in the graph.
        k: Each node is joined with its k nearest neighbors in a ring topology.
        p_rewire: The probability of rewiring each edge.
        fraction_to_remove: The fraction of edges to remove for the social distancing graph.

    Returns:
        A tuple containing the ws graph and the social distancing graph.
    """
    ws_graph = nx.watts_strogatz_graph(N, k, p_rewire)
    sd_graph = create_social_distancing_graph(ws_graph, fraction_to_remove)

    return ws_graph, sd_graph


def plot_results(
    S_ws: list, I_ws: list, R_ws: list, S_sd: list, I_sd: list, R_sd: list
):
    """
    Plots the results of the SIR model for both a ws graph and a social distancing graph.

    Parameters:
        S_ws: Time series data for the susceptible population in the ws graph.
        I_ws: Time series data for the infected population in the ws graph.
        R_ws: Time series data for the recovered population in the ws graph.
        S_sd: Time series data for the susceptible population in the social distancing graph.
        I_sd: Time series data for the infected population in the social distancing graph.
        R_sd: Time series data for the recovered population in the social distancing graph.
    """
    # Plot the results
    plt.figure(figsize=(12, 5))

    # Plot for ws graph
    plt.subplot(1, 2, 1)
    plt.plot(S_ws, label="Susceptible")
    plt.plot(I_ws, label="Infected")
    plt.plot(R_ws, label="Recovered")
    plt.title("SIR Model on Watts Strogatz Graph")
    plt.xlabel("Time")
    plt.ylabel("Number of Individuals")
    plt.legend()

    # Plot for social distancing graph
    plt.subplot(1, 2, 2)
    plt.plot(S_sd, label="Susceptible")
    plt.plot(I_sd, label="Infected")
    plt.plot(R_sd, label="Recovered")
    plt.title("SIR Model on Social Distancing Graph")
    plt.xlabel("Time")
    plt.ylabel("Number of Individuals")
    plt.legend()

    plt.tight_layout()
    plt.show()


def plot_graphs(ws_graph: Graph, sd_graph: Graph):
    """
    Plots the two graphs: a Watts-Strogatz graph and a social distancing graph.

    Parameters:
        ws_graph: The Watts-Strogatz graph.
        sd_graph: The social distancing graph.
    """
    plt.figure(figsize=(12, 5))

    # Plot the Watts-Strogatz graph
    plt.subplot(1, 2, 1)
    nx.draw(
        ws_graph,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        edge_color="gray",
        font_size=15,
        font_color="black",
        font_weight="bold",
    )
    plt.title("Watts-Strogatz Graph")

    # Plot the social distancing graph
    plt.subplot(1, 2, 2)
    nx.draw(
        sd_graph,
        with_labels=True,
        node_color="skyblue",
        node_size=2000,
        edge_color="gray",
        font_size=15,
        font_color="black",
        font_weight="bold",
    )
    plt.title("Social Distancing Graph")

    plt.tight_layout()
    plt.show()


def print_graph_statistics(graph: Graph):
    # Analyze properties of the graph
    num_nodes = graph.number_of_nodes()
    num_edges = graph.number_of_edges()
    density = nx.density(graph)
    clustering_coefficient = nx.average_clustering(graph)

    print(f"Number of nodes: {num_nodes}")
    print(f"Number of edges: {num_edges}")
    print(f"Density: {density:.4f}")
    print(f"Average clustering coefficient: {clustering_coefficient:.4f}")


def run_multiple_simulations(
    graph: Graph,
    alpha: float,
    gamma: float,
    initial_infected: list,
    max_time: int,
    num_simulations: int,
):
    """
    Runs the SIR simulation multiple times on a given graph and collects the total fraction of population infected.

    Parameters:
        graph: NetworkX graph
        alpha: infection rate
        gamma: recovery rate
        initial_infected: list of initially infected nodes
        max_time: maximum number of time steps to simulate
        num_simulations: number of simulations to run

    Returns:
        List of total fraction of population infected in each simulation
    """
    total_infected_list = []
    N = graph.number_of_nodes()

    for _ in range(num_simulations):
        S_history, I_history, R_history = simulate_sir(
            graph, alpha, gamma, initial_infected, max_time
        )
        total_infected = R_history[-1]  # Total recovered at the end
        total_infected_list.append(
            total_infected / N
        )  # Fraction of population infected

    return total_infected_list


def plot_histograms(infected_ws: list, infected_sd: list):
    """
    Plots histograms of the total fraction of population infected over multiple simulations for both graphs.

    Parameters:
        infected_ws: List of fractions from simulations on the Watts-Strogatz graph
        infected_sd: List of fractions from simulations on the social distancing graph
    """
    plt.figure(figsize=(10, 6))
    bins = np.linspace(0, 1, 50)
    plt.hist(infected_ws, bins=bins, alpha=0.5, label="Watts-Strogatz Graph")
    plt.hist(infected_sd, bins=bins, alpha=0.5, label="Social Distancing Graph")
    plt.xlabel("Fraction of Population Infected")
    plt.ylabel("Frequency")
    plt.legend()
    plt.title("Histogram of Total Infections over Simulations")
    plt.show()


if __name__ == "__main__":
    # Parameters for the simulation
    alpha = 0.3  # Infection probability
    gamma = 0.2  # Recovery probability
    max_time = 1000  # Maximum number of time steps
    initial_infected = [0, 1, 2, 3, 4, 5]  # Initially infected nodes

    # Graph parameters
    N = 1000  # Number of nodes
    k = 6  # Each node is connected to k nearest neighbors
    p_rewire = 0.1  # Probability of rewiring each edge
    fraction_to_remove = 0.3  # Fraction of edges to remove for social distancing

    # Run the simulation on both graphs
    ws_graph, sd_graph = init_graphs(N, k, p_rewire, fraction_to_remove)
    S_ws, I_ws, R_ws = simulate_sir(ws_graph, alpha, gamma, initial_infected, max_time)
    S_sd, I_sd, R_sd = simulate_sir(sd_graph, alpha, gamma, initial_infected, max_time)

    # plot_graphs
    print_graph_statistics(ws_graph)
    print_graph_statistics(sd_graph)
    plot_results(S_ws, I_ws, R_ws, S_sd, I_sd, R_sd)

    # Run multiple simulations
    num_simulations = 1000
    infected_ws = run_multiple_simulations(
        ws_graph, alpha, gamma, initial_infected, max_time, num_simulations
    )
    infected_sd = run_multiple_simulations(
        sd_graph, alpha, gamma, initial_infected, max_time, num_simulations
    )

    # Plot histograms
    plot_histograms(infected_ws, infected_sd)
