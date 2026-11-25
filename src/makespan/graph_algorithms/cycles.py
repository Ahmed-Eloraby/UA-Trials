def find_all_simple_cycles(graph):
    """
    Find all distinct simple cycles in an undirected graph.

    Args:
        graph: Adjacency list, represented as a dictionary.
               Example: {0: [1, 2], 1: [0, 2], 2: [0, 1]}

    Returns:
        A list of cycles. Each cycle is a list of vertices,
        without repeating the starting vertex at the end.

    Assumptions:
        - The graph is undirected.
        - The graph has no parallel edges.
        - Cycles must contain at least 3 vertices.
    """

    # Build an undirected adjacency list.
    # This also adds vertices that appear only as neighbors.
    adj = {}

    for node, neighbors in graph.items():
        adj.setdefault(node, set())

        for neighbor in neighbors:
            adj.setdefault(neighbor, set())
            adj[node].add(neighbor)
            adj[neighbor].add(node)

    # Assign each vertex a unique ordering.
    # This helps us avoid finding the same cycle multiple times.
    order = {node: i for i, node in enumerate(adj)}

    cycles = []

    # Try every vertex as the starting vertex.
    for start in adj:
        path = [start]
        in_path = {start}

        def dfs(current):
            for neighbor in adj[current]:

                # We have returned to the starting vertex.
                # The current path is now a cycle.
                if neighbor == start:
                    if len(path) >= 3 and order[path[1]] < order[path[-1]]:
                        cycles.append(path.copy())

                    continue

                # The starting vertex must be the smallest
                # vertex in the cycle, according to our ordering.
                if order[neighbor] <= order[start]:
                    continue

                # A simple cycle cannot repeat a vertex.
                if neighbor in in_path:
                    continue

                # Extend the current path.
                path.append(neighbor)
                in_path.add(neighbor)

                # Explore all possible continuations.
                dfs(neighbor)

                # Backtrack to explore other paths.
                in_path.remove(neighbor)
                path.pop()

        dfs(start)
    cycles.sort(key=len)
    return cycles


def split_at_bridges(graph):
    """
    Find all bridges in an undirected graph, then split the graph
    into connected components by removing all bridges.

    Args:
        graph: Adjacency list, e.g. {"A": ["B"], "B": ["A"]}.

    Returns:
        bridges: List of bridge edges, each represented as (u, v).
        components: List of sets of vertices after removing bridges.
    """

    # Normalize the adjacency list and include missing vertices.
    adj = {node: set(neighbors) for node, neighbors in graph.items()}

    for node, neighbors in list(adj.items()):
        for neighbor in neighbors:
            adj.setdefault(neighbor, set()).add(node)

    # tin[u]: DFS discovery time of u.
    # low[u]: Earliest discovery time reachable from u's subtree.
    tin = {}
    low = {}
    timer = 0

    bridges = []
    bridge_keys = set()

    def dfs(u, parent=None):
        nonlocal timer

        tin[u] = low[u] = timer
        timer += 1

        for v in adj[u]:
            if v == parent:
                continue

            if v in tin:
                # Back edge to an already visited vertex.
                low[u] = min(low[u], tin[v])
            else:
                # DFS tree edge.
                dfs(v, u)

                low[u] = min(low[u], low[v])

                # No back edge from v's subtree reaches u
                # or any ancestor of u.
                if low[v] > tin[u]:
                    bridges.append((u, v))
                    bridge_keys.add(frozenset((u, v)))

    # Handle disconnected input graphs as well.
    for node in adj:
        if node not in tin:
            dfs(node)

    # Remove all bridges logically and find the remaining components.
    components = []
    visited = set()

    for start in adj:
        if start in visited:
            continue

        component = set()
        stack = [start]
        visited.add(start)

        while stack:
            u = stack.pop()
            component.add(u)

            for v in adj[u]:
                # Do not cross a bridge.
                if frozenset((u, v)) in bridge_keys:
                    continue

                if v not in visited:
                    visited.add(v)
                    stack.append(v)

        components.append(component)

    return bridges, components