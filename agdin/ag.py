from dataclasses import dataclass

import numpy as np

from .benchmark import MovingPeaks
from .politicas import PoliticaTamanho


@dataclass
class ConfigAG:
    taxa_cruzamento: float = 0.9
    sigma_mutacao: float = 4.0
    taxa_mutacao_por_gene: float | None = None  # padrão: 1 / dimensão
    tamanho_torneio: int = 3
    # Todas as políticas partem do MESMO tamanho inicial (população aleatória
    # idêntica por semente); a política só atua a partir da 1ª geração.
    tamanho_inicial_comum: int = 100
    # "filhos": crescimento só por mais filhos (padrão). "imigrantes": quando a
    # política pede mais indivíduos que a população atual, o excedente é formado
    # por indivíduos aleatórios uniformes (imigrantes aleatórios).
    crescimento: str = "filhos"
    # Fração dos filhos da 1ª geração após cada mudança substituída por imigrantes
    # aleatórios (controle: mesma injeção de diversidade para políticas fixas).
    fracao_imigrantes_na_mudanca: float = 0.0

    def validar(self) -> None:
        if not 0.0 <= self.taxa_cruzamento <= 1.0:
            raise ValueError("taxa_cruzamento deve estar em [0, 1]")
        if self.sigma_mutacao < 0:
            raise ValueError("sigma_mutacao deve ser >= 0")
        if self.tamanho_torneio < 2:
            raise ValueError("tamanho_torneio deve ser >= 2")
        if self.tamanho_inicial_comum < 4:
            raise ValueError("tamanho_inicial_comum deve ser >= 4")
        if not 0.0 <= self.fracao_imigrantes_na_mudanca <= 1.0:
            raise ValueError("fracao_imigrantes_na_mudanca deve estar em [0, 1]")
        if self.crescimento not in ("filhos", "imigrantes"):
            raise ValueError("crescimento deve ser 'filhos' ou 'imigrantes'")


@dataclass
class ResultadoExecucao:
    nome_politica: str
    geracoes: int
    tamanho_medio: float          # média, sobre gerações, do tamanho populacional
    tamanho_por_ambiente: np.ndarray
    dispersao_por_ambiente: np.ndarray
    geracoes_por_ambiente: np.ndarray
    avaliacoes: int
    trilha_tamanho: np.ndarray    # tamanho a cada geração (para figuras)


class AlgoritmoGenetico:
    def __init__(
        self,
        politica: PoliticaTamanho,
        benchmark: MovingPeaks,
        semente: int,
        config: ConfigAG | None = None,
    ) -> None:
        self.cfg = config or ConfigAG()
        self.cfg.validar()
        self.politica = politica
        self.bench = benchmark
        self.rng = np.random.default_rng(semente)
        self.pm = (
            1.0 / benchmark.d
            if self.cfg.taxa_mutacao_por_gene is None
            else self.cfg.taxa_mutacao_por_gene
        )

    # ------------------------------------------------------------------ #
    def _filhos(self, X: np.ndarray, F: np.ndarray, n: int) -> np.ndarray:
        cfg, b, rng = self.cfg, self.bench, self.rng
        N = len(X)
        idx = rng.integers(0, N, size=(2 * n, cfg.tamanho_torneio))
        venc = idx[np.arange(2 * n), np.argmax(F[idx], axis=1)]
        p1, p2 = X[venc[:n]], X[venc[n:]]
        alfa = rng.random((n, b.d))
        filhos = alfa * p1 + (1.0 - alfa) * p2
        sem_cruz = rng.random(n) >= cfg.taxa_cruzamento
        filhos[sem_cruz] = p1[sem_cruz]
        mascara = rng.random((n, b.d)) < self.pm
        filhos = filhos + mascara * rng.normal(0.0, cfg.sigma_mutacao, size=(n, b.d))
        return np.clip(filhos, b.lo, b.hi)

    @staticmethod
    def _dispersao(X: np.ndarray) -> float:
        """Distância euclidiana média ao centroide (medida de dispersão)."""
        return float(np.mean(np.linalg.norm(X - X.mean(axis=0), axis=1)))

    # ------------------------------------------------------------------ #
    def executar(self, n_ambientes: int) -> ResultadoExecucao:
        b = self.bench
        self.politica.tamanho_inicial()  # reinicia o estado da política
        n0 = self.cfg.tamanho_inicial_comum
        X = self.rng.uniform(b.lo, b.hi, size=(n0, b.d))
        F = b.avaliar(X)
        ambiente_conhecido = b.indice_ambiente

        tamanhos: list[int] = []
        amb_da_geracao: list[int] = []
        dispersoes: list[float] = []

        while b.indice_ambiente < n_ambientes:
            mudou = b.indice_ambiente != ambiente_conhecido
            if mudou:
                ambiente_conhecido = b.indice_ambiente
                if b.indice_ambiente >= n_ambientes:
                    break
                F = b.avaliar(X)  # reavaliação após sinalização de mudança
                if b.indice_ambiente != ambiente_conhecido:
                    # (mudança durante a própria reavaliação; só ocorre se a
                    # população for maior que o intervalo entre mudanças)
                    ambiente_conhecido = b.indice_ambiente
            n_prox = self.politica.proximo_tamanho(mudou, len(X))
            elite_i = int(np.argmax(F))
            elite_x, elite_f = X[elite_i].copy(), float(F[elite_i])

            n_imig = 0
            if self.cfg.crescimento == "imigrantes":
                n_imig = max(0, n_prox - len(X))
            if mudou and self.cfg.fracao_imigrantes_na_mudanca > 0:
                n_imig = max(n_imig, int(round(self.cfg.fracao_imigrantes_na_mudanca * n_prox)))
            filhos = self._filhos(X, F, n_prox - n_imig)
            if n_imig:
                imig = self.rng.uniform(b.lo, b.hi, size=(n_imig, b.d))
                filhos = np.vstack([filhos, imig])
            F_filhos = b.avaliar(filhos)
            pior = int(np.argmin(F_filhos))
            filhos[pior] = elite_x
            F_filhos[pior] = elite_f
            X, F = filhos, F_filhos

            tamanhos.append(n_prox)
            amb_da_geracao.append(b.indice_ambiente)
            dispersoes.append(self._dispersao(X))

        tam = np.asarray(tamanhos, dtype=float)
        amb = np.asarray(amb_da_geracao)
        disp = np.asarray(dispersoes)
        t_amb = np.full(n_ambientes, np.nan)
        d_amb = np.full(n_ambientes, np.nan)
        g_amb = np.zeros(n_ambientes)
        for e in range(n_ambientes):
            sel = amb == e
            if sel.any():
                t_amb[e] = tam[sel].mean()
                d_amb[e] = disp[sel].mean()
                g_amb[e] = sel.sum()
        return ResultadoExecucao(
            nome_politica=self.politica.nome,
            geracoes=len(tam),
            tamanho_medio=float(tam.mean()),
            tamanho_por_ambiente=t_amb,
            dispersao_por_ambiente=d_amb,
            geracoes_por_ambiente=g_amb,
            avaliacoes=b.avaliacoes_total,
            trilha_tamanho=tam,
        )
