import time
import random
import matplotlib.pyplot as plt
import heapq
import inspect

# Generate a sparse or dense graph
def generate_graph(n_nodes, dense=False):
    graph = {i: {} for i in range(n_nodes)}
    edge_probability = 0.8 if dense else 0.2  # 80% for dense, 20% for sparse
    for i in range(n_nodes):
        for j in range(n_nodes):
            if i != j and random.random() < edge_probability:
                weight = random.randint(1, 10)
                graph[i][j] = weight
    return graph

# Analyze the algorithms by comparing execution times on sparse and dense graphs
def analyze_algorithms(alg1, alg2, alg1_name="Alg1", alg2_name="Alg2", node_counts=None):
    if node_counts is None:
        node_counts = [5, 10, 20, 50, 100]

    dijkstra_sparse_times = []
    dijkstra_dense_times = []
    floyd_sparse_times = []
    floyd_dense_times = []

    for n in node_counts:
        start_node = 0

        # Dijkstra on sparse
        sparse_graph1 = generate_graph(n, dense=False)
        start = time.perf_counter()
        alg1(sparse_graph1, start_node)
        end = time.perf_counter()
        dijkstra_sparse_times.append(end - start)

        # Dijkstra on dense
        dense_graph1 = generate_graph(n, dense=True)
        start = time.perf_counter()
        alg1(dense_graph1, start_node)
        end = time.perf_counter()
        dijkstra_dense_times.append(end - start)

        # Floyd-Warshall on sparse
        sparse_graph2 = generate_graph(n, dense=False)
        start = time.perf_counter()
        alg2(sparse_graph2)
        end = time.perf_counter()
        floyd_sparse_times.append(end - start)

        # Floyd-Warshall on dense
        dense_graph2 = generate_graph(n, dense=True)
        start = time.perf_counter()
        alg2(dense_graph2)
        end = time.perf_counter()
        floyd_dense_times.append(end - start)

    # Plot the results
    plt.figure(figsize=(8, 6))
    plt.plot(node_counts, floyd_sparse_times, label=f"{alg2_name} (Sparse)", marker='s')
    plt.plot(node_counts, dijkstra_sparse_times, label=f"{alg1_name} (Sparse)", marker='o')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Performance Comparison: {alg1_name} vs {alg2_name} : Sparse Graphs')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(8, 6))
    plt.plot(node_counts, floyd_dense_times, label=f"{alg2_name} (Dense)", marker='s')
    plt.plot(node_counts, dijkstra_dense_times, label=f"{alg1_name} (Dense)", marker='o')
    plt.xlabel('Number of Nodes')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Performance Comparison: {alg1_name} vs {alg2_name} : Dense Graphs')
    plt.legend()
    plt.grid(True)

    # Summary Output
    avg_dijkstra_sparse = sum(dijkstra_sparse_times) / len(dijkstra_sparse_times)
    avg_dijkstra_dense = sum(dijkstra_dense_times) / len(dijkstra_dense_times)
    avg_floyd_sparse = sum(floyd_sparse_times) / len(floyd_sparse_times)
    avg_floyd_dense = sum(floyd_dense_times) / len(floyd_dense_times)

    print(f"Average time for {alg1_name} on sparse graphs: {avg_dijkstra_sparse:.6f} seconds")
    print(f"Average time for {alg1_name} on dense graphs:  {avg_dijkstra_dense:.6f} seconds")
    print(f"Average time for {alg2_name} on sparse graphs: {avg_floyd_sparse:.6f} seconds")
    print(f"Average time for {alg2_name} on dense graphs:  {avg_floyd_dense:.6f} seconds")

    if avg_dijkstra_sparse < avg_floyd_sparse:
        print(f"On sparse graphs, {alg1_name} is faster.")
    else:
        print(f"On sparse graphs, {alg2_name} is faster.")

    if avg_dijkstra_dense < avg_floyd_dense:
        print(f"On dense graphs, {alg1_name} is faster.")
    else:
        print(f"On dense graphs, {alg2_name} is faster.")
    plt.show()

# Dijkstra's algorithm (single source shortest path)
def dijkstra(graph, start):
    distances = {node: float('inf') for node in graph}
    distances[start] = 0
    priority_queue = [(0, start)]

    while priority_queue:
        current_distance, current_node = heapq.heappop(priority_queue)

        if current_distance > distances[current_node]:
            continue

        for neighbor, weight in graph[current_node].items():
            distance = current_distance + weight
            if distance < distances[neighbor]:
                distances[neighbor] = distance
                heapq.heappush(priority_queue, (distance, neighbor))
    return distances

# Floyd-Warshall algorithm (all pairs shortest path)
def floyd_warshall(graph):
    nodes = list(graph.keys())
    n = len(nodes)
    dist = [[float('inf')] * n for _ in range(n)]

    # Initialize distances based on the graph
    for i in range(n):
        dist[i][i] = 0  # Distance to itself is always zero
    for i in graph:
        for j in graph[i]:
            dist[i][j] = graph[i][j]  # Set the direct edges' distances

    # Dynamic programming to update shortest paths
    for k in range(n):  # Check for intermediate nodes
        for i in range(n):  # From node i
            for j in range(n):  # To node j
                if dist[i][k] + dist[k][j] < dist[i][j]:
                    dist[i][j] = dist[i][k] + dist[k][j]  # Update if a shorter path is found
    return dist

# Run the analysis on Dijkstra's and Floyd-Warshall algorithms
analyze_algorithms(dijkstra, floyd_warshall, "Dijkstra", "Floyd-Warshall")
