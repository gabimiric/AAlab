import time
import random
import matplotlib.pyplot as plt
import inspect
from collections import deque

# Generate a directed, weighted graph with optional density
def generate_graph(n_nodes, dense=False):
    graph = {i: {} for i in range(n_nodes)}
    edge_probability = 0.8 if dense else 0.2  # Higher probability for dense graphs
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and random.random() < edge_probability:
                weight = random.randint(1, 10)
                graph[i][j] = weight
    return graph

# Compare execution times of two graph traversal algorithms
def analyze_algorithms(alg1, alg2, alg1_name="Alg1", alg2_name="Alg2", node_counts=None):
    if node_counts is None:
        node_counts = [10, 50, 100, 200, 300, 400, 500]
        # node_counts = [100, 200, 500, 750, 1000]
        # node_counts = [10, 20, 30, 40, 50, 75, 100]

    alg1_times = []
    alg2_times = []

    for n in node_counts:
        sparse_graph = generate_graph(n, dense=False)
        dense_graph = generate_graph(n, dense=True)
        start_node = 0

        # Time Algorithm 1
        start = time.perf_counter()
        if len(inspect.signature(alg1).parameters) == 1:
            alg1(sparse_graph)
            alg1(dense_graph)
        else:
            alg1(sparse_graph, start_node)
            alg1(dense_graph, start_node)
        end = time.perf_counter()
        alg1_times.append(end - start)

        # Time Algorithm 2
        start = time.perf_counter()
        if len(inspect.signature(alg2).parameters) == 1:
            alg2(sparse_graph)
            alg2(dense_graph)
        else:
            alg2(sparse_graph, start_node)
            alg2(dense_graph, start_node)
        end = time.perf_counter()
        alg2_times.append(end - start)

    # Plot timing results
    plt.figure(figsize=(12, 6))
    plt.plot(node_counts, alg1_times, label=f"{alg1_name} Time", marker='o')
    plt.plot(node_counts, alg2_times, label=f"{alg2_name} Time", marker='s')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Empirical Analysis: {alg1_name} vs {alg2_name}')
    plt.legend()
    plt.grid(True)

    # Summary output
    avg_alg1 = sum(alg1_times) / len(alg1_times)
    avg_alg2 = sum(alg2_times) / len(alg2_times)
    print(f"Average time for {alg1_name}: {avg_alg1:.6f} seconds")
    print(f"Average time for {alg2_name}: {avg_alg2:.6f} seconds")
    if avg_alg1 < avg_alg2:
        print(f"Conclusion: {alg1_name} is faster on average.")
    else:
        print(f"Conclusion: {alg2_name} is faster on average.")
    plt.show()

# Depth-First Search
def dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend(reversed(graph[node]))  # Ensure consistent traversal order

# Breadth-First Search
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend(graph[node])

# Run analysis comparing DFS and BFS
analyze_algorithms(dfs, bfs, "DFS", "BFS")
