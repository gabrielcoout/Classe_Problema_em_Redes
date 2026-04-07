import numpy as np
import copy
from src.Except import *
from src.ProblemaP1 import ProblemaP1

class ProblemaP2(ProblemaP1):
    """
    Problema Probabilístico Direto.
    Herda ProblemaP1 e adiciona análise de Monte Carlo
    para estimar a probabilidade de falha por excesso de pressão.
    """

    def __init__(self, *args, p1_instance=None, r_prob=0.1, alpha=2.0, P_max=1e5, n_samples=1000, seed=None, **kwargs):
        """
        """
        if p1_instance is not None:
            if not isinstance(p1_instance, ProblemaP1):
                raise TypeError("p1_instance deve ser uma instância de ProblemaP1.")
            self.__dict__.update(copy.deepcopy(p1_instance).__dict__)
        else:
            super().__init__(*args, **kwargs)

        self.r_prob = r_prob
        self.alpha = alpha
        self.P_max = P_max
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def _sample_obstruction(self, area_threshold=None):
        """
        Sorteia quais arestas estão obstruídas.
        Se area_threshold for definido, canos com área > area_threshold 
        nunca serão obstruídos.
        """
        obstructions = {}
        for u in self.edges:
            for v, attr in self.edges[u].items():
                # Se não houver threshold ou se a área for menor que o limite, sorteia
                if area_threshold is None or attr.get('area', 0) <= area_threshold:
                    is_obstructed = self.rng.random() < self.r_prob
                else:
                    # Cano muito grande: impossível obstruir
                    is_obstructed = False
                    
                obstructions[(u, v)] = is_obstructed
                
        return obstructions

    def _get_problem(self, obstruction_state):
        scenario = ProblemaP1.__new__(ProblemaP1)
        scenario.__dict__.update(copy.deepcopy(
            {k: v for k, v in self.__dict__.items()
            if k not in ('r_prob', 'alpha', 'P_max', 'n_samples',
                        'rng', 'failures', 'results',
                        'max_pressures', 'node_failure_counts')}
        ))
        # NÃO chama scenario.setup() — K base já veio do deepcopy

        # mapeia (u, v) -> índice diagonal de K
        edge_index = {
            (u, v): idx
            for idx, (u, v) in enumerate(
                (u, v)
                for u in self.edges
                for v in self.edges[u]
            )
        }

        # aplica obstrução: divide condutância por alpha
        for (u, v), obstructed in obstruction_state.items():
            if obstructed:
                i = edge_index[(u, v)]
                scenario.K[i, i] /= self.alpha

        # reconstrói M com K modificado
        A = scenario.compute_connection_matrix()
        scenario.M = A.T @ scenario.K @ A

        return scenario

    def _run_single(self):
        """Executa uma única simulação. Retorna apenas booleano (falha ou não)."""
        obstruction = self._sample_obstruction()
        problem = self._get_problem(obstruction)
        problem.solve()
        return bool(np.any(problem.p > self.P_max))
    
    def _run_single_detailed(self):
        """Executa uma única simulação. Retorna (pressões, pressão_máxima, houve_falha)."""
        obstruction = self._sample_obstruction()
        problem = self._get_problem(obstruction)
        problem.solve()
        
        pressures = problem.p
        max_p = np.max(pressures)
        failure = bool(np.any(pressures > self.P_max))
        
        return pressures, max_p, failure

    # ------------------------------------------------------------------
    # Interface pública
    # ------------------------------------------------------------------

    def run(self):
        """Executa todas as n_samples simulações e armazena os resultados."""
        N = len(self.nodes)
        self.node_failure_counts = np.zeros(N)
        self.results = []
        self.max_pressures = []
        self.failures = 0

        for _ in range(self.n_samples):
            pressures, max_p, failure = self._run_single_detailed()  # Mude aqui!
            
            self.results.append(pressures)
            self.max_pressures.append(max_p)
            
            if failure:
                self.failures += 1
                self.node_failure_counts += (pressures > self.P_max)
        
        self.results = np.array(self.results)
        self.max_pressures = np.array(self.max_pressures)

    def estimate_pf(self, n_iter: int, confidence: float = 0.95) -> tuple:
        """
        Roda n_iter simulações e retorna (P_f, ic_lower, ic_upper).
        Stateless: nenhum resultado é armazenado na instância.
        """
        from scipy.stats import norm

        failures = sum(self._run_single() for _ in range(n_iter))
        pf = failures / n_iter

        z = norm.ppf(1 - (1 - confidence) / 2)
        margin = z * np.sqrt(pf * (1 - pf) / n_iter)

        return pf, max(0.0, pf - margin), min(1.0, pf + margin)

    def probability_of_failure(self):
        """P_f = P(max_i p_i > P_max)"""
        return self.failures / self.n_samples

    def node_failure_probability(self):
        """Probabilidade de excedência por nó."""
        return self.node_failure_counts / self.n_samples

    def confidence_interval(self, confidence=0.95):
        """
        Intervalo de confiança para P_f via aproximação normal.
        IC = P_f ± z * sqrt(P_f*(1-P_f)/n)
        """
        from scipy.stats import norm
        p = self.probability_of_failure()
        z = norm.ppf(1 - (1 - confidence) / 2)
        margin = z * np.sqrt(p * (1 - p) / self.n_samples)
        return (max(0.0, p - margin), min(1.0, p + margin))

    def summary(self):
        """Resumo dos resultados do P2."""
        p_f = self.probability_of_failure()
        ci = self.confidence_interval()
        return {
            "n_samples":              self.n_samples,
            "P_fail":                 p_f,
            "IC_95%":                 ci,
            "mean_max_pressure":      float(np.mean(self.max_pressures)),
            "std_max_pressure":       float(np.std(self.max_pressures)),
            "max_pressure_observed":  float(np.max(self.max_pressures)),
            "node_failure_prob":      self.node_failure_probability().tolist(),
        }

