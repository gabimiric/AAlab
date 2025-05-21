import time
import random
import matplotlib.pyplot as plt
import inspect
import networkx as nx
from collections import deque

# ---------------------------------------
# Graph Generation Functions
# ---------------------------------------

# General graph generator (sparse or dense based on `dense` flag)
def generate_graph(n_nodes, dense=False):
    graph = {i: {} for i in range(n_nodes)}
    edge_probability = 0.8 if dense else 0.2  # Higher probability for dense graphs

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):  # Avoid self-loops and duplicate edges
            if random.random() < edge_probability:
                weight = random.randint(1, 10)
                graph[i][j] = weight
                graph[j][i] = weight  # Ensure undirected graph
    return graph

# Generate a complete graph where every pair of nodes is connected
def generate_complete_graph(n):
    return {i: {j: random.randint(1, 10) for j in range(n) if j != i} for i in range(n)}

# Generate a bipartite graph with two equal partitions
def generate_bipartite_graph(n):
    graph = {i: {} for i in range(n)}
    set1 = set(range(n//2))
    set2 = set(range(n//2, n))
    for u in set1:
        for v in set2:
            graph[u][v] = random.randint(1, 10)
            graph[v][u] = random.randint(1, 10)
    return graph

# Generate a cyclic graph where nodes are connected in a cycle
def generate_cyclic_graph(n):
    return {i: {((i + 1) % n): random.randint(1, 10)} for i in range(n)}

# Generate a star graph with a central node connected to all others
def generate_star_graph(n):
    graph = {0: {i: random.randint(1, 10) for i in range(1, n)}}
    for i in range(1, n):
        graph[i] = {}
    return graph

# Generate a path graph (linear sequence of nodes)
def generate_path_graph(n):
    graph = {i: {i+1: random.randint(1, 10)} for i in range(n-1)}
    graph[n-1] = {}
    return graph

# Generate a tree by connecting each new node to a random previous node
def generate_tree_graph(n):
    graph = {i: {} for i in range(n)}
    for i in range(1, n):
        parent = random.randint(0, i-1)
        graph[parent][i] = random.randint(1, 10)
    return graph

# Generate a Directed Acyclic Graph (DAG)
def generate_dag(n):
    graph = {i: {} for i in range(n)}
    # Guarantee a path from 0 to n-1
    for i in range(n - 1):
        graph[i][i + 1] = random.randint(1, 10)
    # Add more edges without introducing cycles
    for i in range(n):
        for j in range(i + 2, n):
            if random.random() < 0.3:
                graph[i][j] = random.randint(1, 10)
    return graph

# Generate a sparse graph using default settings
def generate_random_sparse_graph(n):
    return generate_graph(n, dense=False)

# Generate a dense graph using default settings
def generate_random_dense_graph(n):
    return generate_graph(n, dense=True)

# ---------------------------------------
# Graph Traversal Algorithms
# ---------------------------------------

# Depth-First Search
def dfs(graph, start):
    visited = set()
    stack = [start]
    while stack:
        node = stack.pop()
        if node not in visited:
            visited.add(node)
            stack.extend([n for n in reversed(graph[node]) if n not in visited])

# Breadth-First Search
def bfs(graph, start):
    visited = set()
    queue = deque([start])
    while queue:
        node = queue.popleft()
        if node not in visited:
            visited.add(node)
            queue.extend([n for n in graph[node] if n not in visited])

# ---------------------------------------
# Timing & Comparison Function
# ---------------------------------------

# Compares execution time of two traversal algorithms across varying graph sizes
def analyze_algorithms(alg1, alg2, alg1_name="Alg1", alg2_name="Alg2", node_counts=None):
    if node_counts is None:
        node_counts = [10, 50, 100, 200, 300, 400, 500]

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

    # Plot timing comparison
    plt.figure(figsize=(12, 6))
    plt.plot(node_counts, alg1_times, label=f"{alg1_name} Time", marker='o')
    plt.plot(node_counts, alg2_times, label=f"{alg2_name} Time", marker='s')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Empirical Analysis: {alg1_name} vs {alg2_name}')
    plt.legend()
    plt.grid(True)

    # Display summary statistics
    avg_alg1 = sum(alg1_times) / len(alg1_times)
    avg_alg2 = sum(alg2_times) / len(alg2_times)
    print(f"Average time for {alg1_name}: {avg_alg1:.6f} seconds")
    print(f"Average time for {alg2_name}: {avg_alg2:.6f} seconds")
    if avg_alg1 < avg_alg2:
        print(f"Conclusion: {alg1_name} is faster on average.")
    else:
        print(f"Conclusion: {alg2_name} is faster on average.")
    plt.show()

# ---------------------------------------
# Visualization Function for Traversal
# ---------------------------------------

# Visualizes the step-by-step traversal using DFS or BFS
def visualize_traversal(graph, start, traversal_func, title="Graph Traversal"):
    G = nx.DiGraph()
    for u in graph:
        for v in graph[u]:
            G.add_edge(u, v, weight=graph[u][v])

    pos = nx.spring_layout(G)
    visited = []
    last_node = None
    first_visit = True

    # Generator for DFS visualization
    def dfs_viz(node):
        visited.clear()
        stack = [node]
        while stack:
            n = stack.pop()
            if n not in visited:
                visited.append(n)
                yield n
                stack.extend([neighbor for neighbor in reversed(graph[n]) if neighbor not in visited])

    # Generator for BFS visualization
    def bfs_viz(node):
        visited.clear()
        queue = deque([node])
        while queue:
            n = queue.popleft()
            if n not in visited:
                visited.append(n)
                yield n
                queue.extend([neighbor for neighbor in graph[n] if neighbor not in visited])

    gen_func = dfs_viz if traversal_func == dfs else bfs_viz

    fig = plt.figure()
    while plt.fignum_exists(fig.number):
        gen = gen_func(start)
        first_visit = True
        last_node = None
        for node in gen:
            if not plt.fignum_exists(fig.number):
                break

            # Set node colors based on traversal state
            node_colors = []
            for n in G.nodes:
                if first_visit and n == node and node == start:
                    node_colors.append("green")  # Starting node
                elif n == node:
                    node_colors.append("red")  # Current node
                elif n == last_node:
                    node_colors.append("orange")  # Previously visited node
                else:
                    node_colors.append("lightblue")  # Default

            plt.clf()
            nx.draw(G, pos, with_labels=True, node_color=node_colors)
            plt.title(f"{title}: visiting node {node}")
            plt.pause(0.5)

            last_node = node
            if first_visit and node == start:
                first_visit = False

    plt.close(fig)

# ---------------------------------------
# Visual Comparison Across Graph Types
# ---------------------------------------

# Run traversal and visualize for various graph types
def analyze_graph_types(traversal_func, name):
    graph_types = {
        "Complete": generate_complete_graph,
        "Bipartite": generate_bipartite_graph,
        "Cyclic": generate_cyclic_graph,
        "Star": generate_star_graph,
        "Tree": generate_tree_graph,
        "DAG": generate_dag,
        "Random Sparse": generate_random_sparse_graph,
        "Random Dense": generate_random_dense_graph
    }

    for name_type, gen_func in graph_types.items():
        print(f"--- Visualizing {name} on {name_type} Graph ---")
        graph = gen_func(10)  # Small size for clear visualization
        visualize_traversal(graph, 0, traversal_func, title=f"{name_type} - {name}")

# ---------------------------------------
# Run Analysis
# ---------------------------------------

# Compare DFS and BFS empirically
analyze_algorithms(dfs, bfs, "DFS", "BFS")

# Visualize DFS and BFS on various graph structures
analyze_graph_types(dfs, "DFS")
analyze_graph_types(bfs, "BFS")
