import time
import random
import matplotlib.pyplot as plt
import networkx as nx
import heapq
import inspect


# Generate a random undirected graph usable by Prim (adj list) and Kruskal (edge list)
def generate_graph(n_nodes, dense=False):
    graph_adj_list = {i: [] for i in range(n_nodes)}
    edges = []

    # Adjust edge probability for sparse or dense graphs
    if dense:
        edge_probability = 0.5  # Higher probability for dense graphs
    else:
        edge_probability = 0.1  # Lower probability for sparse graphs

    for i in range(n_nodes):
        for j in range(i + 1, n_nodes):
            if random.random() < edge_probability:
                weight = random.randint(1, 10)
                graph_adj_list[i].append((j, weight))
                graph_adj_list[j].append((i, weight))
                edges.append((weight, i, j))

    return graph_adj_list, edges  # Return both formats for different algorithms


# Weighted graph generators

def generate_complete_graph(n):
    """Generate a complete graph where every node connects to every other node"""
    graph = {i: [] for i in range(n)}
    for i in range(n):
        for j in range(n):
            if i != j:  # No self-loops
                weight = random.randint(1, 10)
                # Check if edge already exists to avoid duplicates
                if not any(neighbor[0] == j for neighbor in graph[i]):
                    graph[i].append((j, weight))
                    graph[j].append((i, weight))
    return graph


def generate_bipartite_graph(n):
    graph = {i: [] for i in range(n)}
    set1 = set(range(n // 2))
    set2 = set(range(n // 2, n))
    for u in set1:
        for v in set2:
            w = random.randint(1, 10)
            graph[u].append((v, w))
            graph[v].append((u, w))
    return graph


def generate_cyclic_graph(n):
    graph = {i: [] for i in range(n)}
    for i in range(n):
        j = (i + 1) % n
        w = random.randint(1, 10)
        graph[i].append((j, w))
        graph[j].append((i, w))
    return graph


def generate_star_graph(n):
    graph = {i: [] for i in range(n)}
    for i in range(1, n):
        w = random.randint(1, 10)
        graph[0].append((i, w))
        graph[i].append((0, w))
    return graph


# Generate a sparse graph with exactly n nodes and approximately n+2 edges
def generate_random_sparse_graph(n):
    graph = {i: [] for i in range(n)}

    # First ensure connectivity by creating a path through all nodes
    for i in range(n - 1):
        weight = random.randint(1, 10)
        graph[i].append((i + 1, weight))
        graph[i + 1].append((i, weight))

    # Add a few more random edges to make it more interesting but still sparse
    # For a sparse graph, we'll add approximately n/3 more random edges
    additional_edges = max(2, n // 3)
    edges_added = 0

    while edges_added < additional_edges:
        u = random.randrange(n)
        v = random.randrange(n)

        # Avoid self-loops and existing edges
        if u != v and not any(neighbor[0] == v for neighbor in graph[u]):
            weight = random.randint(1, 10)
            graph[u].append((v, weight))
            graph[v].append((u, weight))
            edges_added += 1

    return graph


# Generate a dense graph with approximately n*(n-1)/2 * 0.7 edges"""
def generate_random_dense_graph(n):
    graph = {i: [] for i in range(n)}

    # For dense graphs, we want approximately 70% of all possible edges
    density = 0.7
    max_possible_edges = n * (n - 1) // 2
    target_edges = int(max_possible_edges * density)

    # First ensure connectivity by creating a path through all nodes
    for i in range(n - 1):
        weight = random.randint(1, 10)
        graph[i].append((i + 1, weight))
        graph[i + 1].append((i, weight))

    # Count existing edges
    edges_count = n - 1

    # Add more random edges to reach density target
    attempts = 0
    max_attempts = max_possible_edges * 5  # Avoid infinite loops

    while edges_count < target_edges and attempts < max_attempts:
        attempts += 1
        u = random.randrange(n)
        v = random.randrange(n)

        # Avoid self-loops and existing edges
        if u != v and not any(neighbor[0] == v for neighbor in graph[u]):
            weight = random.randint(1, 10)
            graph[u].append((v, weight))
            graph[v].append((u, weight))
            edges_count += 1

    return graph


# Converts both {u: {v: w}} and {u: [(v, w)]} formats to {u: [(v, w)]}.
def adj_dict_to_list(adj_dict):
    converted = {}
    for u in adj_dict:
        neighbors = adj_dict[u]
        if isinstance(neighbors, dict):
            converted[u] = [(v, w) for v, w in neighbors.items()]
        elif isinstance(neighbors, list):
            converted[u] = list(neighbors)  # already in correct format
        else:
            raise ValueError(f"Unexpected format for neighbors of node {u}: {neighbors}")
    return converted


# Converts adj dict to unique (w, u, v) edges
def adj_dict_to_edges(adj_dict):
    seen = set()
    edges = []
    for u in adj_dict:
        for v, w in adj_dict[u]:
            if (u, v) not in seen and (v, u) not in seen:
                edges.append((w, u, v))
                seen.add((u, v))
                seen.add((v, u))
    return edges


def analyze_algorithms(alg1, alg2, alg1_name="Alg1", alg2_name="Alg2", node_counts=None, high_node_counts=None):
    if node_counts is None:
        node_counts = [5, 10, 15, 20, 30, 50, 75, 100]  # Example set of low node sizes

    if high_node_counts is None:
        high_node_counts = [150, 200, 250, 300, 350, 400]  # Example set of high node sizes

    # Initialize time lists
    alg1_times = []
    alg2_times = []

    # Low node analysis
    for n in node_counts:
        low_node_graph_adj_list, low_node_graph_edges = generate_graph(n, dense=False)
        high_node_graph_adj_list, high_node_graph_edges = generate_graph(n, dense=True)
        start_node = 0

        start = time.perf_counter()
        if len(inspect.signature(alg1).parameters) == 1:
            alg1(low_node_graph_adj_list)
            alg1(high_node_graph_adj_list)
        else:
            alg1(low_node_graph_adj_list, start_node)
            alg1(high_node_graph_adj_list, start_node)
        end = time.perf_counter()
        alg1_times.append(end - start)

        start = time.perf_counter()
        if len(inspect.signature(alg2).parameters) == 1:
            alg2(low_node_graph_edges, n)
            alg2(high_node_graph_edges, n)
        else:
            alg2(low_node_graph_edges, n)
            alg2(high_node_graph_edges, n)
        end = time.perf_counter()
        alg2_times.append(end - start)

    # High node analysis with different node counts
    high_alg1_times = []
    high_alg2_times = []

    for n in high_node_counts:
        low_node_graph_adj_list, low_node_graph_edges = generate_graph(n, dense=False)
        high_node_graph_adj_list, high_node_graph_edges = generate_graph(n, dense=True)
        start_node = 0

        start = time.perf_counter()
        if len(inspect.signature(alg1).parameters) == 1:
            alg1(low_node_graph_adj_list)
            alg1(high_node_graph_adj_list)
        else:
            alg1(low_node_graph_adj_list, start_node)
            alg1(high_node_graph_adj_list, start_node)
        end = time.perf_counter()
        high_alg1_times.append(end - start)

        start = time.perf_counter()
        if len(inspect.signature(alg2).parameters) == 1:
            alg2(low_node_graph_edges, n)
            alg2(high_node_graph_edges, n)
        else:
            alg2(low_node_graph_edges, n)
            alg2(high_node_graph_edges, n)
        end = time.perf_counter()
        high_alg2_times.append(end - start)

    # Plotting execution time vs node count for low node count graphs
    plt.figure(figsize=(8, 6))
    plt.plot(node_counts, alg1_times, label=f"{alg1_name} Time (Low Node)", marker='o')
    plt.plot(node_counts, alg2_times, label=f"{alg2_name} Time (Low Node)", marker='s')
    plt.xlabel('Number of Nodes (Low Node Count)')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Empirical Analysis: {alg1_name} vs {alg2_name}: Low Node Count')
    plt.legend()
    plt.grid(True)

    plt.figure(figsize=(8, 6))
    # Plotting execution time vs node count for high node count graphs
    plt.plot(high_node_counts, high_alg1_times, label=f"{alg1_name} Time (High Node)", marker='x')
    plt.plot(high_node_counts, high_alg2_times, label=f"{alg2_name} Time (High Node)", marker='^')

    plt.xlabel('Number of Nodes (High Node Count)')
    plt.ylabel('Execution Time (seconds)')
    plt.title(f'Empirical Analysis: {alg1_name} vs {alg2_name}: High Node Count')
    plt.legend()
    plt.grid(True)

    avg_alg1 = sum(alg1_times) / len(alg1_times)
    avg_alg2 = sum(alg2_times) / len(alg2_times)
    avg_high_alg1 = sum(high_alg1_times) / len(high_alg1_times)
    avg_high_alg2 = sum(high_alg2_times) / len(high_alg2_times)

    print(f"Average time for {alg1_name} (Low Node Count): {avg_alg1:.6f} seconds")
    print(f"Average time for {alg2_name} (Low Node Count): {avg_alg2:.6f} seconds")
    print(f"Average time for {alg1_name} (High Node Count): {avg_high_alg1:.6f} seconds")
    print(f"Average time for {alg2_name} (High Node Count): {avg_high_alg2:.6f} seconds")

    if avg_alg1 < avg_alg2:
        print(f"Conclusion: {alg1_name} is faster on average for low node count graphs.")
    else:
        print(f"Conclusion: {alg2_name} is faster on average for low node count graphs.")

    if avg_high_alg1 < avg_high_alg2:
        print(f"Conclusion: {alg1_name} is faster on average for high node count graphs.")
    else:
        print(f"Conclusion: {alg2_name} is faster on average for high node count graphs.")

    plt.show()


def visualize_algorithm(graph, start, algorithm, title="Graph Algorithm"):
    if algorithm.__name__ == "prim":
        def visualize_prim(graph_adj_list, title):
            G = nx.Graph()
            for u in graph_adj_list:
                for v, w in graph_adj_list[u]:
                    G.add_edge(u, v, weight=w)

            pos = nx.spring_layout(G, seed=42)
            fig = plt.figure()

            # Main visualization loop - will continue until window is closed
            while plt.fignum_exists(fig.number):
                # Inner algorithm loop - repeats the full algorithm visualization
                while plt.fignum_exists(fig.number):
                    visited = set()
                    mst_edges = []
                    mst_edge_set = set()
                    start_node = list(graph_adj_list.keys())[0]
                    min_heap = [(0, start_node, None)]
                    last_node = None
                    first_visit = True

                    # Visualize each step of Prim's algorithm
                    while min_heap and plt.fignum_exists(fig.number):
                        weight, node, from_node = heapq.heappop(min_heap)
                        if node in visited:
                            continue
                        visited.add(node)

                        if from_node is not None:
                            mst_edges.append((from_node, node))
                            mst_edge_set.add((from_node, node))
                            mst_edge_set.add((node, from_node))

                        plt.clf()
                        node_colors = []
                        for n in G.nodes:
                            if first_visit and n == node and node == start_node:
                                node_colors.append("green")
                            elif n == node:
                                node_colors.append("red")
                            elif last_node and n == last_node:
                                node_colors.append("orange")
                            elif n in visited:
                                node_colors.append("lightgreen")
                            else:
                                node_colors.append("lightblue")

                        edge_colors = []
                        for e in G.edges:
                            if e in mst_edges or (e[1], e[0]) in mst_edges:
                                edge_colors.append("red")
                            elif e in mst_edge_set or (e[1], e[0]) in mst_edge_set:
                                edge_colors.append("orange")
                            else:
                                edge_colors.append("gray")

                        nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors)
                        edge_labels = nx.get_edge_attributes(G, "weight")
                        nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                        plt.title(f"{title}: Adding node {node}")
                        plt.pause(0.6)

                        mst_edges.clear()
                        last_node = node
                        if first_visit:
                            first_visit = False

                        for neighbor, w in graph_adj_list[node]:
                            if neighbor not in visited:
                                heapq.heappush(min_heap, (w, neighbor, node))

                    # Short pause between full algorithm repetitions
                    if plt.fignum_exists(fig.number):
                        plt.pause(1)

            plt.close(fig)

        # Convert to adjacency list if needed
        if isinstance(graph, dict):
            adj_list = adj_dict_to_list(graph)
        else:
            adj_list = graph
        visualize_prim(adj_list, title)

    elif algorithm.__name__ == "kruskal":
        def visualize_kruskal(graph_edges, n, title):
            G = nx.Graph()
            # Create a set of all nodes that exist in the graph
            all_nodes = set()
            for w, u, v in graph_edges:
                G.add_edge(u, v, weight=w)
                all_nodes.add(u)
                all_nodes.add(v)

            # Ensure n is accurate based on actual node count
            n = max(all_nodes) + 1

            pos = nx.spring_layout(G, seed=42)
            fig = plt.figure()

            # Main visualization loop - will continue until window is closed
            while plt.fignum_exists(fig.number):
                # Inner algorithm loop - repeats the full algorithm visualization
                while plt.fignum_exists(fig.number):
                    disjoint_set = DisjointSet(n)
                    mst_edge_set = set()
                    sorted_edges = sorted(graph_edges)

                    # Visualize each step of Kruskal's algorithm
                    for weight, u, v in sorted_edges:
                        if not plt.fignum_exists(fig.number):
                            break

                        if disjoint_set.union(u, v):
                            current_edge = (u, v)
                            mst_edge_set.add(current_edge)
                            mst_edge_set.add((v, u))

                            plt.clf()
                            node_colors = []
                            nodes_in_mst = {node for e in mst_edge_set for node in e}

                            # Only iterate through actual nodes in the graph
                            for node in G.nodes:
                                if node == u:
                                    node_colors.append("red")
                                elif node == v:
                                    node_colors.append("orange")
                                elif node in nodes_in_mst:
                                    node_colors.append("lightgreen")
                                else:
                                    node_colors.append("lightblue")

                            edge_colors = []
                            for e in G.edges:
                                if e == current_edge or (e[1], e[0]) == current_edge:
                                    edge_colors.append("red")
                                elif e in mst_edge_set or (e[1], e[0]) in mst_edge_set:
                                    edge_colors.append("orange")
                                else:
                                    edge_colors.append("gray")

                            nx.draw(G, pos, with_labels=True, node_color=node_colors, edge_color=edge_colors)
                            edge_labels = nx.get_edge_attributes(G, "weight")
                            nx.draw_networkx_edge_labels(G, pos, edge_labels=edge_labels)
                            plt.title(f"{title}: Adding edge ({u}, {v}) weight {weight}")
                            plt.pause(0.6)

                    # Short pause between full algorithm repetitions
                    if plt.fignum_exists(fig.number):
                        plt.pause(1)

            plt.close(fig)

        # Convert to edge list format
        if isinstance(graph, dict):
            adj_list = adj_dict_to_list(graph)
            edges = adj_dict_to_edges(adj_list)
        else:
            edges = graph
        visualize_kruskal(edges, len(graph), title)

    else:
        raise ValueError("Unsupported algorithm for MST visualization.")


# Run visualization across graph types
def visualize_weighted_traversal(algorithm, name):
    graph_types = {
        "Complete": generate_complete_graph,
        "Bipartite": generate_bipartite_graph,
        "Cyclic": generate_cyclic_graph,
        "Star": generate_star_graph,
        "Random Sparse": generate_random_sparse_graph,
        "Random Dense": generate_random_dense_graph
    }

    for graph_name, gen_func in graph_types.items():
        print(f"--- Visualizing {name} on {graph_name} Graph ---")
        # Use a consistent node count
        n_nodes = 7
        graph = gen_func(n_nodes)

        start_node = 0
        try:
            visualize_algorithm(graph, start_node, algorithm, title=f"{graph_name} - {name}")
        except Exception as e:
            print(f"Error visualizing {graph_name} graph: {e}")


# Prim's algorithm using min-heap (priority queue)
def prim(graph_adj_list, start_node=None):
    if start_node is None:
        start_node = list(graph_adj_list.keys())[0]

    mst = []
    total_weight = 0
    visited = set()
    min_heap = [(0, start_node, None)]  # (weight, node, from_node)

    while min_heap:
        weight, node, from_node = heapq.heappop(min_heap)
        if node in visited:
            continue

        visited.add(node)
        if from_node is not None:  # Skip the first node which has no parent
            mst.append((from_node, node, weight))
            total_weight += weight

        for neighbor, edge_weight in graph_adj_list[node]:
            if neighbor not in visited:
                heapq.heappush(min_heap, (edge_weight, neighbor, node))

    return mst, total_weight


# Union-Find data structure
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])
        return self.parent[u]

    def union(self, u, v):
        root_u = self.find(u)
        root_v = self.find(v)
        if root_u != root_v:
            if self.rank[root_u] > self.rank[root_v]:
                self.parent[root_v] = root_u
            elif self.rank[root_u] < self.rank[root_v]:
                self.parent[root_u] = root_v
            else:
                self.parent[root_v] = root_u
                self.rank[root_u] += 1
            return True
        return False


# Kruskal's algorithm using sorted edge list and disjoint set
def kruskal(graph_edges, n):
    mst = []
    total_weight = 0
    edges = sorted(graph_edges)
    disjoint_set = DisjointSet(n)

    for weight, u, v in edges:
        if disjoint_set.union(u, v):
            mst.append((u, v, weight))
            total_weight += weight

    return mst, total_weight


# Run comparing Prim and Kruskal on multiple node counts
analyze_algorithms(prim, kruskal, "Prim", "Kruskal")

# Run visualizations
visualize_weighted_traversal(prim, "Prim's MST Algorithm")
visualize_weighted_traversal(kruskal, "Kruskal's MST Algorithm")
