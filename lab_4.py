import random
import matplotlib.pyplot as plt
import networkx as nx
from collections import deque
import heapq
import time


# Generate a sparse or dense graph
def generate_graph(n_nodes, dense=False):
    graph = {i: {} for i in range(n_nodes)}
    edge_probability = 0.8 if dense else 0.2  # 80% for dense, 20% for sparse

    # Ensure the graph is connected by creating a path through all nodes
    for i in range(n_nodes - 1):
        weight = random.randint(1, 10)
        graph[i][i + 1] = weight

    # Add additional random edges based on density
    for i in range(n_nodes):
        for j in range(n_nodes):
            # Skip self-loops and existing edges
            if i != j and j not in graph[i] and random.random() < edge_probability:
                weight = random.randint(1, 10)
                graph[i][j] = weight

    return graph


def generate_complete_graph(n):
    return {i: {j: random.randint(1, 10) for j in range(n) if j != i} for i in range(n)}


def generate_bipartite_graph(n):
    graph = {i: {} for i in range(n)}
    set1 = set(range(n // 2))
    set2 = set(range(n // 2, n))
    for u in set1:
        for v in set2:
            w = random.randint(1, 10)
            graph[u][v] = w
            graph[v][u] = w
    return graph


def generate_cyclic_graph(n):
    graph = {i: {((i + 1) % n): random.randint(1, 10)} for i in range(n)}
    return graph


def generate_star_graph(n):
    graph = {0: {i: random.randint(1, 10) for i in range(1, n)}}
    for i in range(1, n):
        graph[i] = {}
    return graph


def generate_tree_graph(n):
    graph = {i: {} for i in range(n)}
    for i in range(1, n):
        parent = random.randint(0, i - 1)
        graph[parent][i] = random.randint(1, 10)
    return graph


def generate_dag(n):
    graph = {i: {} for i in range(n)}

    # Ensure the DAG is connected by creating a path through all nodes
    for i in range(n - 1):
        graph[i][i + 1] = random.randint(1, 10)

    # Add additional forward edges to maintain the DAG property
    for i in range(n):
        for j in range(i + 2, n):  # Skip direct successors that already have edges
            if random.random() < 0.3:
                graph[i][j] = random.randint(1, 10)
    return graph


def generate_random_sparse_graph(n):
    graph = {i: {} for i in range(n)}
    edge_prob = 0.2

    # Ensure the graph is connected by creating a path through all nodes
    for i in range(n - 1):
        w = random.randint(1, 10)
        graph[i][i + 1] = w
        graph[i + 1][i] = w

    # Add additional random edges based on sparse probability
    for i in range(n):
        for j in range(i + 1, n):
            # Skip existing edges (from the path)
            if j != i + 1 and random.random() < edge_prob:
                w = random.randint(1, 10)
                graph[i][j] = w
                graph[j][i] = w
    return graph


def generate_random_dense_graph(n):
    graph = {i: {} for i in range(n)}
    edge_prob = 0.8

    # Ensure the graph is connected by creating a path through all nodes
    for i in range(n - 1):
        w = random.randint(1, 10)
        graph[i][i + 1] = w
        graph[i + 1][i] = w

    # Add additional random edges based on dense probability
    for i in range(n):
        for j in range(i + 1, n):
            # Skip existing edges (from the path)
            if j != i + 1 and random.random() < edge_prob:
                w = random.randint(1, 10)
                graph[i][j] = w
                graph[j][i] = w
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


# Visualization function for both algorithms
def visualize_algorithm(graph, start, algorithm, title="Graph Algorithm"):
    G = nx.DiGraph()
    for u in graph:
        for v in graph[u]:
            G.add_edge(u, v, weight=graph[u][v])

    pos = nx.spring_layout(G, seed=42)
    last_node = None
    first_visit = True

    # Dijkstra's algorithm generator to yield nodes in the order they're finalized (visited)
    def dijkstra_generator(graph, start):
        distances = {node: float('inf') for node in graph}
        distances[start] = 0
        visited = set()
        pq = [(0, start)]

        while pq:
            dist, node = heapq.heappop(pq)
            if node in visited:
                continue
            visited.add(node)
            yield node  # yield the node when it's "visited" / finalized
            for neighbor, weight in graph[node].items():
                if neighbor not in visited:
                    new_dist = dist + weight
                    if new_dist < distances[neighbor]:
                        distances[neighbor] = new_dist
                        heapq.heappush(pq, (new_dist, neighbor))

    # Floyd-Warshall animation: yield the intermediate node k after each k iteration
    def floyd_warshall_generator(graph):
        nodes = list(graph.keys())
        n = len(nodes)
        dist = [[float('inf')] * n for _ in range(n)]
        for i in range(n):
            dist[i][i] = 0
        for i in graph:
            for j in graph[i]:
                dist[i][j] = graph[i][j]

        for k in range(n):
            for i in range(n):
                for j in range(n):
                    if dist[i][k] + dist[k][j] < dist[i][j]:
                        dist[i][j] = dist[i][k] + dist[k][j]
            yield k  # yield intermediate node k after finishing updates

    fig = plt.figure()
    while plt.fignum_exists(fig.number):
        if algorithm == 'dijkstra':
            gen = dijkstra_generator(graph, start)
        elif algorithm == 'floyd':
            gen = floyd_warshall_generator(graph)
        else:
            raise ValueError("Algorithm must be 'dijkstra' or 'floyd'")

        last_node = None
        first_visit = True
        for node in gen:
            if not plt.fignum_exists(fig.number):
                break

            node_colors = []
            for n in G.nodes:
                if first_visit and n == node and node == start:
                    node_colors.append("green")
                elif n == node:
                    node_colors.append("red")
                elif n == last_node:
                    node_colors.append("orange")
                else:
                    node_colors.append("lightblue")

            plt.clf()
            nx.draw(G, pos, with_labels=True, node_color=node_colors)
            edge_labels = nx.get_edge_attributes(G, 'weight')
            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
            if algorithm == 'floyd':
                plt.title(f"{title}: After intermediate node {node} updates")
            else:
                plt.title(f"{title}: Visiting node {node}")
            plt.pause(0.5)

            last_node = node
            if first_visit and node == start:
                first_visit = False

    plt.close(fig)


# Analyze all graph types with animation for a given algorithm
def visualize_weighted_traversal(algorithm, name):
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

    for graph_name, gen_func in graph_types.items():
        print(f"--- Visualizing {name} on {graph_name} Graph ---")
        graph = gen_func(7)
        visualize_algorithm(graph, 0, algorithm, title=f"{graph_name} - {name}")


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

# Visualization:
visualize_weighted_traversal('dijkstra', 'Dijkstra')
visualize_weighted_traversal('floyd', 'Floyd-Warshall')
