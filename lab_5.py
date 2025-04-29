import time
import random
import matplotlib.pyplot as plt
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


def analyze_algorithms(alg1, alg2, alg1_name="Alg1", alg2_name="Alg2", node_counts=None, high_node_counts=None):
    if node_counts is None:
        node_counts = [10, 20, 30, 40, 50, 75, 100]  # Example set of low node sizes

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


# Prim's algorithm using min-heap (priority queue)
def prim(graph_adj_list):
    mst = []
    total_weight = 0
    start_node = list(graph_adj_list.keys())[0]
    visited = set()
    min_heap = [(0, start_node)]  # (weight, node)

    while min_heap:
        weight, node = heapq.heappop(min_heap)
        if node not in visited:
            visited.add(node)
            mst.append((node, weight))
            total_weight += weight
            for neighbor, edge_weight in graph_adj_list[node]:
                if neighbor not in visited:
                    heapq.heappush(min_heap, (edge_weight, neighbor))

    return mst, total_weight


# Union-Find data structure with path compression and union by rank
class DisjointSet:
    def __init__(self, n):
        self.parent = list(range(n))
        self.rank = [0] * n

    def find(self, u):
        if self.parent[u] != u:
            self.parent[u] = self.find(self.parent[u])  # Path compression
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
