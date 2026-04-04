import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

class Grafo:
    def __init__(self, nodes=None, edges=None, kind=None):
        if isinstance(nodes, list):
            self.nodes = {n: {} for n in nodes}  
        elif isinstance(nodes, dict):
            self.nodes = nodes
        elif nodes is None:
            self.nodes = {}
        else:
            raise TypeError("nodes must be a list or dict")
        self.edges = edges if edges is not None else {}
        self.kind = "Undirected" if kind is None else kind

        self.validate()

        self.num_nodes = len(self.nodes)
        self.num_edges = sum(len(link) for link in self.edges.values())

    def validate(self):
        """Checks if all edges connect existing nodes."""
        for u in self.edges:
            if u not in self.nodes:
                raise Exception(f"Source node {u} not found.")

            for v in self.edges[u]:
                if v not in self.nodes:
                    raise Exception(f"Target node {v} not found.")

                if self.kind == 'Directed':
                    if v in self.edges and u in self.edges[v]:
                        raise ValueError(f"Não é permitido criar {u}->{v} porque {v}->{u} já existe.")
        return True

    def insert_node(self, index, **kwargs):
        if index not in self.nodes:
            self.nodes[index] = kwargs
            self.edges[index] = {}
            self.num_nodes += 1

    def insert_edge(self, u, v, **kwargs):
        if u not in self.nodes or v not in self.nodes:
            raise Exception(f"Nodes {u} and {v} must exist.")

        if self.kind == 'Directed':
            if v in self.edges and u in self.edges[v]:
                raise ValueError(f"Não é permitido criar {u}->{v} porque {v}->{u} já existe.")

        if u not in self.edges:
            self.edges[u] = {}

        if v not in self.edges[u]:
            self.num_edges += 1

        self.edges[u][v] = kwargs

    def get_edge_data(self, u, v):
        return self.edges.get(u, {}).get(v, None)

    def remove_node(self, index):
        if index not in self.nodes:
            raise Exception(f"Node {index} not found.")

        # remover arestas saindo do nó
        if index in self.edges:
            self.num_edges -= len(self.edges[index])
            del self.edges[index]

        # remover arestas entrando no nó
        for u in self.edges:
            if index in self.edges[u]:
                del self.edges[u][index]
                self.num_edges -= 1

        del self.nodes[index]
        self.num_nodes -= 1

    def compute_connection_matrix(self) -> np.ndarray:
        """
        Returns the incidence (connection) matrix for a directed graph.

        Each row represents an edge.
        +1 -> source node
        -1 -> target node
        """

        if self.kind != "Directed":
            raise ValueError("Connection matrix is defined only for directed graphs.")

        # map node -> column index
        node_list = list(self.nodes.keys())
        node_index = {node: i for i, node in enumerate(node_list)}

        # initialize matrix
        B = np.zeros((self.num_edges, self.num_nodes))

        row = 0
        for u in self.edges:
            for v in self.edges[u]:
                B[row, node_index[u]] = 1    # source
                B[row, node_index[v]] = -1   # target
                row += 1

        return B

    def get_network(self):
        G = nx.DiGraph() if self.kind == "Directed" else nx.Graph()

        # Adiciona os nós e suas propriedades
        for u, attrs in self.nodes.items():
            G.add_node(u, **attrs)

        # Adiciona as arestas
        for u in self.edges:
            for v, attrs in self.edges[u].items():
                G.add_edge(u, v, **attrs)

        return G


    def plot(self, show_node_labels=True, show_edge_labels=True, precision=2, layout="planar"):
        G = self.get_network()

        if layout == "planar":
            try:
                pos = nx.planar_layout(G)
            except:
                pos = nx.spring_layout(G)
        elif layout == "spring":
            pos = nx.spring_layout(G)
        elif layout == "circular":
            pos = nx.circular_layout(G)
        elif layout == "shell":
            pos = nx.shell_layout(G)
        elif layout == "kamada_kawai":
            pos = nx.kamada_kawai_layout(G)
        elif layout == "spectral":
            pos = nx.spectral_layout(G)
        else:
            raise ValueError(
                "layout must be one of: 'planar', 'spring', 'circular', 'shell', 'kamada_kawai', 'spectral'"
            )

        plt.figure(figsize=(8, 6))

        nx.draw_networkx_nodes(G, pos, node_size=700, node_color='skyblue')

        if self.kind == "Directed":
            nx.draw_networkx_edges(G, pos, arrowstyle='-', arrowsize=20)
        else:
            nx.draw_networkx_edges(G, pos)

        if show_node_labels:
            node_labels = {
                node: node + "\n" + "\n".join([
                    f"{k[0].upper()}:{val:.{precision}f}" if isinstance(val, (int, float)) else f"{k[0].upper()}:{val}"
                    for k, val in attr.items()
                ])
                for node, attr in G.nodes(data=True)
            }

            nx.draw_networkx_labels(
                G,
                pos,
                labels=node_labels,
                font_color='black',
                font_size=9
            )

        if show_edge_labels:
            edge_labels = {
                (u, v): "\n".join([
                    f"{k[0].upper()}:{val:.{precision}f}" if isinstance(val, (int, float)) else f"{k[0].upper()}:{val}"
                    for k, val in data.items()
                ])
                for u, v, data in G.edges(data=True)
            }
            nx.draw_networkx_edge_labels(
                G,
                pos,
                edge_labels=edge_labels,
                font_color='red',
                font_size=9
            )

        plt.title("Visualização do Grafo")
        plt.axis('off')
        plt.show()