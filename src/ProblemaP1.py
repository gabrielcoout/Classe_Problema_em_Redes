import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from src.Except import *
from src.Grafo import *

class ProblemaP1(Grafo):
    def __init__(self, nodes, edges, mu, patm, Q_ext=None):
        """
        nodes: lista ou dict de nós
        edges: dict de edges {u:{v:{'area':..., 'length':...}}}
        mu: viscosidade
        patm: pressão de referência
        Q_ext: dict {node: fluxo externo}
        """
        super().__init__(nodes, edges)
        self.kind = 'Directed'
        self.mu = mu
        self.Q_ext = None if Q_ext is None else Q_ext
        self.patm = patm
        self.validate_problem_schema()

        self.M = None
        self.K = None
        self.is_fitted = False

        if Q_ext is not None:
            self.set_Q_ext(Q_ext)  

    def setup(self):
        if self.Q_ext is None:
            raise ValueError("Q_ext must be defined before fitting. Use set_Q_ext().")
        
        self.K = K = self._compute_physical_matrix()
        A = self.compute_connection_matrix()
        self.M = A.T @ K @ A
        self.is_fitted = True
        return self

    def solve(self):
        if self.Q_ext is None:
            raise ValueError("Define Q_ext before solving the system of equations. Use set_Q_ext() method.")
        if not self.is_fitted:
            raise NotFittedError()
        
        # Transform dict Q_ext em array na ordem dos nós
        node_keys = list(self.nodes.keys())
        b = np.array([self.Q_ext[key] for key in node_keys])

        fixed_indices = [i for i, ext_flow in enumerate(b) if ext_flow < 0]
        if not fixed_indices:
            raise ValueError("No node with negative flow found for reference pressure.")
        n = len(node_keys)
        all_indices = np.arange(n)
        free_indices = np.array([i for i in all_indices if i not in fixed_indices])
        
        p = np.zeros(n, dtype=float)  # Inicializa vetor de pressões
        p[fixed_indices] = self.patm

        # Sistema reduzido:
        free_free_matrix = self.M[np.ix_(free_indices, free_indices)]
        free_fixed_matrix = self.M[np.ix_(free_indices, fixed_indices)]
        free_rhs = b[free_indices] - free_fixed_matrix @ p[fixed_indices]

        free_pressures = np.linalg.solve(free_free_matrix, free_rhs)
        p[free_indices] = free_pressures
        self.p = p

        # Atualiza self.nodes com pressão
        for idx, key in enumerate(node_keys):
            self.nodes[key]['pressao'] = float(self.p[idx])

        return self.p

    def set_Q_ext(self, Q_ext: dict):
        """
        Define Q_ext como um dicionário {node: fluxo} e valida:
        - todos os nós estão presentes
        - soma dos fluxos é ~0
        """
        if not isinstance(Q_ext, dict):
            raise TypeError("Q_ext must be a dictionary {node: flow}.")
        
        missing_nodes = [node for node in self.nodes if node not in Q_ext]
        if missing_nodes:
            raise ValueError(f"Q_ext missing values for nodes: {missing_nodes}")
        
        total_flux = sum(Q_ext.values())
        if abs(total_flux) > 1e-6:
            raise ValueError(f"External flux does not conserve mass. Sum(Q_ext)={total_flux}")

        self.Q_ext = Q_ext
    
        for node, q in Q_ext.items():
                self.nodes[node]["fluxo_externo"] = float(q)
        
    def validate_problem_schema(self):
        if not isinstance(self.edges, dict):
            raise TypeError("Edges must be a dictionary.")

        for u in self.edges:
            if not isinstance(self.edges[u], dict):
                raise TypeError(f"Edges from node {u} must be a dictionary.")

            for v, attrs in self.edges[u].items():

                # checar se attrs é dict
                if not isinstance(attrs, dict):
                    raise TypeError(f"Edge ({u}->{v}) must have attribute dictionary.")

                # checar chaves obrigatórias
                required_keys = ["area", "length"]
                for key in required_keys:
                    if key not in attrs:
                        raise ValueError(f"Edge ({u}->{v}) missing required attribute '{key}'.")

                A = attrs["area"]
                L = attrs["length"]

                # checar tipo numérico
                if not isinstance(A, (int, float)):
                    raise TypeError(f"Edge ({u}->{v}) area must be numeric.")
                if not isinstance(L, (int, float)):
                    raise TypeError(f"Edge ({u}->{v}) length must be numeric.")

                # checar valores físicos válidos
                if A <= 0:
                    raise ValueError(f"Edge ({u}->{v}) area must be > 0.")
                if L <= 0:
                    raise ValueError(f"Edge ({u}->{v}) length must be > 0.")

        return True
    
    def assert_solvability(self, tol=1e-6):
        total_flux = sum(self.Q_ext.values())
        if abs(total_flux) > 1e-6:
            raise ValueError(f"External flux does not conserve mass. Sum(Q_ext)={total_flux}")

    def _compute_conductivity(self, area, length):
        if self.mu <= 0:
            raise ValueError("mu<=0")
        D = np.sqrt(4 * area / np.pi)
        return (np.pi * D**4) / (128 * self.mu * length)

    def _compute_physical_matrix(self) -> np.ndarray:
        K = np.zeros((self.num_edges, self.num_edges))

        idx = 0
        for u in self.edges:
            for v, attrs in self.edges[u].items():
                A = float(attrs["area"])
                L = float(attrs["length"])

                c = self._compute_conductivity(A, L)

                K[idx, idx] = c
                idx += 1

        self.K = K
        return K
    
    def validate_Q_ext(self):
        if len(self.Q_ext) != self.num_nodes:
            raise IndexError("Q_ext should be the same size as the number of nodes.")

    # override   
    def insert_edge(self, u, v, **kwargs):
        if u not in self.nodes or v not in self.nodes:
            raise Exception(f"Nodes {u} and {v} must exist.")
        
        required_fields = ["area", "length"]
        missing_fields = [field for field in required_fields if field not in kwargs]
        if missing_fields:
            raise ValueError(f"Missing required edge attributes: {', '.join(missing_fields)}")

        if self.kind == 'Directed':
            if v in self.edges and u in self.edges[v]:
                raise ValueError(f"Não é permitido criar {u}->{v} porque {v}->{u} já existe.")

        if u not in self.edges:
            self.edges[u] = {}

        if v not in self.edges[u]:
            self.num_edges += 1

        self.edges[u][v] = kwargs
        
    
