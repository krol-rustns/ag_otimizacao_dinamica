import argparse
import json
import platform
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

from agdin.ag import ConfigAG
from agdin.experimento import EspecPolitica, Regime, executar_lote

FREQUENCIAS = (1000, 5000)       # avaliações por ambiente
SEVERIDADES = (1.0, 5.0, 10.0)   # deslocamento (parâmetro s)


def principal() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sementes", type=int, default=30)
    ap.add_argument("--ambientes", type=int, default=30)
    ap.add_argument("--saida", default="resultados")
    ap.add_argument("--processos", type=int, default=None)
    args = ap.parse_args()

    cfg = ConfigAG()
    sementes = list(range(args.sementes))
    saida = Path(args.saida)
    saida.mkdir(parents=True, exist_ok=True)

    base = [
        EspecPolitica("saw", n_min=20, n_max=100, decaimento=10, rotulo="SAW"),
        EspecPolitica("fixa", n=20, rotulo="FIX-20"),
        EspecPolitica("fixa", n=50, rotulo="FIX-50"),
        EspecPolitica("fixa", n=100, rotulo="FIX-100"),
    ]

    dfs, metas, curvas = [], [], {}
    n_eq: dict[str, int] = {}
    t0 = time.time()
    for freq in FREQUENCIAS:
        for sev in SEVERIDADES:
            reg = Regime(freq, sev)
            df, cv, meta = executar_lote(
                base, [reg], sementes, args.ambientes, cfg, args.processos
            )
            n_medio = int(round(meta[meta.variante == "SAW"].tamanho_medio.mean()))
            n_eq[reg.nome] = n_medio
            esp_eq = EspecPolitica("fixa", n=n_medio, rotulo="FIX-eq")
            df2, cv2, meta2 = executar_lote(
                [esp_eq], [reg], sementes, args.ambientes, cfg, args.processos
            )
            dfs += [df, df2]
            metas += [meta, meta2]
            curvas.update(cv)
            curvas.update(cv2)
            print(f"{reg.nome}: N_eq={n_medio}  ({time.time() - t0:.0f}s)", flush=True)

    pd.concat(dfs, ignore_index=True).to_csv(saida / "ambientes.csv", index=False)
    pd.concat(metas, ignore_index=True).to_csv(saida / "execucoes.csv", index=False)
    np.savez_compressed(
        saida / "curvas.npz",
        **{f"{v}|{r}": a for (v, r), a in curvas.items()},
    )
    (saida / "config.json").write_text(
        json.dumps(
            {
                "sementes": args.sementes,
                "ambientes_por_execucao": args.ambientes,
                "frequencias_avaliacoes": FREQUENCIAS,
                "severidades": SEVERIDADES,
                "n_equivalente_por_regime": n_eq,
                "ag": cfg.__dict__,
                "python": sys.version,
                "numpy": np.__version__,
                "plataforma": platform.platform(),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print("Concluído:", saida.resolve())


if __name__ == "__main__":
    principal()
