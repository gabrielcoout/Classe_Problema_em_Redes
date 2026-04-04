import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import random
from src.ProblemaP1 import ProblemaP1

class ProblemaP1Generator:
    def __init__(self, seed=None):
        if seed is not None:
            np.random.seed(seed)
            random.seed(seed)

    def generate(
        self,
        num_nodes=5,
        edge_prob=0.4,
        mu=1,
        patm=0,
        q_mode="integer",
        sink_ratio=0.4,
        single_sink=False,
    ):
        """
        Gera uma instância aleatória de ProblemaP1.

        O grafo gerado é garantidamente conexo. Os fluxos externos são
        gerados de acordo com o modo aleatório selecionado e sempre satisfazem a
        condição de solvabilidade.

        Parâmetros
        ----------
        num_nodes : int, opcional
            Número de nós do grafo. Deve ser no mínimo 2.
        edge_prob : float, opcional
            Probabilidade de adicionar arestas extras após a construção
            inicial da árvore geradora.
        mu : float, opcional
            Parâmetro de viscosidade do fluido passado para ProblemaP1.
        patm : float, opcional
            Pressão atmosférica de referência passada para ProblemaP1.
        q_mode : str, opcional
            Estratégia usada para gerar os fluxos externos. Os valores
            suportados são:
            `"integer"`, `"uniform"` e `"gaussian"`.
        sink_ratio : float, opcional
            Fração aproximada de nós que receberão fluxo externo negativo.
            Ignorado se `single_sink=True`.
        single_sink : bool, opcional
            Se True, gera exatamente um nó sumidouro.

        Retorna
        -------
        ProblemaP1
            Uma instância aleatória de ProblemaP1 com estrutura de grafo conexa
            e fluxos externos balanceados.
        """
        nodes = {f"node_{i}": {} for i in range(num_nodes)}
        node_names = list(nodes.keys())

        edges = self._generate_edges(node_names, edge_prob)
        q_values = self._generate_q_values(
            num_nodes=num_nodes,
            mode=q_mode,
            sink_ratio=sink_ratio,
            single_sink=single_sink
        )

        Q_ext = {node_names[i]: q_values[i] for i in range(num_nodes)}
        assert abs(sum(Q_ext.values())) < 1e-9, "Q_ext must sum to 0."

        return ProblemaP1(nodes, edges, mu=mu, patm=patm, Q_ext=Q_ext)

    def _generate_edges(self, node_names, edge_prob):
        edges = {name: {} for name in node_names}

        for i in range(1, len(node_names)):
            u = node_names[random.randint(0, i - 1)]
            v = node_names[i]
            self._add_random_edge(edges, u, v)

        for i in range(len(node_names)):
            for j in range(i + 1, len(node_names)):
                u = node_names[i]
                v = node_names[j]
                if v not in edges[u] and random.random() < edge_prob:
                    self._add_random_edge(edges, u, v)

        return edges

    def _add_random_edge(self, edges, u, v):
        edges[u][v] = {
            "area": round(np.random.uniform(0.5, 2.0), 4),
            "length": round(np.random.uniform(1.0, 5.0), 2)
        }

    def _generate_q_values(self, num_nodes, mode="integer", sink_ratio=0.4, single_sink=False):
        if num_nodes < 2:
            raise ValueError("num_nodes must be at least 2.")

        num_sinks = 1 if single_sink else max(1, min(num_nodes - 1, round(num_nodes * sink_ratio)))
        num_sources = num_nodes - num_sinks

        if mode == "integer":
            negatives = self._generate_integer_negatives(num_sinks, num_sources)
            positive_total = -sum(negatives)
            positives = self._integer_partition(positive_total, num_sources)

        elif mode == "uniform":
            negatives = [-float(x) for x in np.random.uniform(0.5, 5.0, size=num_sinks)]
            positive_total = -sum(negatives)
            positives = self._float_partition(positive_total, num_sources)

        elif mode == "gaussian":
            negatives = []
            while len(negatives) < num_sinks:
                x = abs(np.random.normal(1.0, 0.5))
                if x > 1e-8:
                    negatives.append(-float(x))
            positive_total = -sum(negatives)
            positives = self._float_partition(positive_total, num_sources)

        else:
            raise ValueError("mode must be one of: 'integer', 'uniform', 'gaussian'")

        q_values = negatives + positives
        random.shuffle(q_values)

        if mode == "integer":
            q_values[0] += -sum(q_values)
        else:
            q_values[-1] += -sum(q_values)

        return q_values

    def _generate_integer_negatives(self, num_sinks, num_sources):
        while True:
            negatives = [-random.randint(1, 5) for _ in range(num_sinks)]
            if -sum(negatives) >= num_sources:
                return negatives

    def _integer_partition(self, total, parts):
        if parts == 1:
            return [total]

        cuts = sorted(random.sample(range(1, total), parts - 1))
        values = []
        prev = 0

        for cut in cuts:
            values.append(cut - prev)
            prev = cut

        values.append(total - prev)
        return values

    def _float_partition(self, total, parts):
        if parts == 1:
            return [float(total)]

        cuts = sorted(np.random.uniform(0, total, parts - 1))
        values = []
        prev = 0.0

        for cut in cuts:
            values.append(float(cut - prev))
            prev = cut

        values.append(float(total - prev))
        return values