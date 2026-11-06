"""Orquestração dos experimentos: regimes de mudança x políticas x sementes."""

from __future__ import annotations

import os
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, asdict

import numpy as np
import pandas as pd

from .ag import AlgoritmoGenetico, ConfigAG
from .benchmark import MovingPeaks, LIMIARES_RECUPERACAO, N_FAIXAS
from .politicas import DenteDeSerra, Fixa, PoliticaTamanho


@dataclass(frozen=True)
class Regime:
    mudanca_a_cada: int       # frequência: avaliações por ambiente
    severidade: float         # deslocamento dos picos (parâmetro s)

    @property
    def nome(self) -> str:
        return f"f{self.mudanca_a_cada}_s{self.severidade:g}"


@dataclass(frozen=True)
class EspecPolitica:
    tipo: str                 # "fixa" | "saw"
    n: int = 50
    n_min: int = 20
    n_max: int = 100
    decaimento: int = 10
    rotulo: str = ""

    def criar(self) -> PoliticaTamanho:
        if self.tipo == "fixa":
            return Fixa(self.n, nome=self.rotulo or f"FIX-{self.n}")
        if self.tipo == "saw":
            return DenteDeSerra(
                self.n_min, self.n_max, self.decaimento, nome=self.rotulo or "SAW"
            )
        raise ValueError(self.tipo)


def executar_uma(
    espec: EspecPolitica,
    regime: Regime,
    semente: int,
    n_ambientes: int,
    config: ConfigAG,
    dimensao: int = 5,
    n_picos: int = 10,
    bench_kw: dict | None = None,
) -> tuple[list[dict], np.ndarray, dict]:
    bench = MovingPeaks(
        dimensao=dimensao,
        n_picos=n_picos,
        mudanca_a_cada=regime.mudanca_a_cada,
        severidade_deslocamento=regime.severidade,
        semente=semente,                  # ambientes dependem só desta semente
        **(bench_kw or {}),
    )
    politica = espec.criar()
    ag = AlgoritmoGenetico(politica, bench, semente=semente + 1000, config=config)
    res = ag.executar(n_ambientes)

    linhas: list[dict] = []
    curvas = np.full((n_ambientes, N_FAIXAS), np.nan)
    for reg in bench.registros[:n_ambientes]:
        e = reg.indice
        curvas[e] = reg.curva_erro
        linha = {
            "variante": politica.nome,
            "regime": regime.nome,
            "mudanca_a_cada": regime.mudanca_a_cada,
            "severidade": regime.severidade,
            "semente": semente,
            "ambiente": e,
            "erro_offline": reg.erro_offline,
            "erro_antes_mudanca": reg.erro_antes_mudanca,
            "tamanho_medio": res.tamanho_por_ambiente[e],
            "dispersao": res.dispersao_por_ambiente[e],
            "geracoes": res.geracoes_por_ambiente[e],
        }
        for lim in LIMIARES_RECUPERACAO:
            linha[f"rec_{lim:g}"] = reg.tempo_recuperacao[lim]
        linhas.append(linha)
    meta = {
        "variante": politica.nome,
        "regime": regime.nome,
        "semente": semente,
        "tamanho_medio": res.tamanho_medio,
        "avaliacoes": res.avaliacoes,
    }
    return linhas, curvas, meta


def _tarefa(args):
    return executar_uma(*args)


def executar_lote(
    especs: list[EspecPolitica],
    regimes: list[Regime],
    sementes: list[int],
    n_ambientes: int,
    config: ConfigAG,
    processos: int | None = None,
    bench_kw: dict | None = None,
    dimensao: int = 5,
    n_picos: int = 10,
) -> tuple[pd.DataFrame, dict, pd.DataFrame]:
    tarefas = [
        (e, r, s, n_ambientes, config, dimensao, n_picos, bench_kw)
        for r in regimes
        for e in especs
        for s in sementes
    ]
    processos = processos or max(1, (os.cpu_count() or 1))
    if processos == 1:
        resultados = [_tarefa(t) for t in tarefas]
    else:
        with ProcessPoolExecutor(max_workers=processos) as ex:
            resultados = list(ex.map(_tarefa, tarefas, chunksize=4))

    linhas: list[dict] = []
    curvas: dict[tuple[str, str], list[np.ndarray]] = {}
    metas: list[dict] = []
    for (espec, regime, semente, *_), (ls, cv, meta) in zip(tarefas, resultados):
        linhas.extend(ls)
        curvas.setdefault((meta["variante"], meta["regime"]), []).append(cv)
        metas.append(meta)
    curvas_agreg = {k: np.stack(v) for k, v in curvas.items()}  # (sementes, amb, faixas)
    return pd.DataFrame(linhas), curvas_agreg, pd.DataFrame(metas)
