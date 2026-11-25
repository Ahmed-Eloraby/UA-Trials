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


def find_components(graph):
    """
    Find all connected components in an undirected graph.

    Args:
        graph: Adjacency list, represented as a dictionary.

    Returns:
        A list of sets, where each set contains one connected component.
    """

    # Include every vertex, even if it appears only as a neighbor.
    adj = {node: set(neighbors) for node, neighbors in graph.items()}

    for node, neighbors in list(adj.items()):
        for neighbor in neighbors:
            adj.setdefault(neighbor, set()).add(node)

    visited = set()
    components = []

    # Start a DFS from every unvisited vertex.
    for start in adj:
        if start in visited:
            continue

        component = set()
        stack = [start]
        visited.add(start)

        while stack:
            current = stack.pop()
            component.add(current)

            for neighbor in adj[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    stack.append(neighbor)

        components.append(component)

    return components