import sys, time, json
from pathlib import Path
import numpy as np, pandas as pd
from agdin.ag import ConfigAG
from agdin.experimento import EspecPolitica, Regime, executar_lote

REG = [Regime(f, s) for f in (1000, 5000) for s in (1.0, 5.0, 10.0)]
BASE = [EspecPolitica("saw", n_min=20, n_max=100, decaimento=10, rotulo="SAW"),
        EspecPolitica("fixa", n=20, rotulo="FIX-20"),
        EspecPolitica("fixa", n=50, rotulo="FIX-50"),
        EspecPolitica("fixa", n=100, rotulo="FIX-100")]

def salvar(dir_, dfs, metas, extra=None):
    d = Path("resultados_ext") / dir_; d.mkdir(parents=True, exist_ok=True)
    pd.concat(dfs, ignore_index=True).to_csv(d / "ambientes.csv", index=False)
    pd.concat(metas, ignore_index=True).to_csv(d / "execucoes.csv", index=False)
    if extra: (d / "info.json").write_text(json.dumps(extra, indent=1))

def bloco_com_eq(nome, sementes, cfg=None, **kw):
    cfg = cfg or ConfigAG(); dfs, metas, neq = [], [], {}
    for reg in REG:
        df, _, m = executar_lote(BASE, [reg], sementes, 30, cfg, **kw)
        n = int(round(m[m.variante == "SAW"].tamanho_medio.mean())); neq[reg.nome] = n
        df2, _, m2 = executar_lote([EspecPolitica("fixa", n=n, rotulo="FIX-eq")], [reg], sementes, 30, cfg, **kw)
        dfs += [df, df2]; metas += [m, m2]; print(nome, reg.nome, n, flush=True)
    salvar(nome, dfs, metas, {"N_eq": neq, **{k: str(v) for k, v in kw.items()}})

def principal():
    t0 = time.time()
    for b in sys.argv[1:]:
        if b == "A":
            dfs, metas = [], []
            for reg in REG:
                df, _, m = executar_lote([EspecPolitica("saw", rotulo="SAW-IMM")], [reg], list(range(30)), 30, ConfigAG(crescimento="imigrantes"))
                dfs.append(df); metas.append(m)
            salvar("A_imigrantes", dfs, metas)
        elif b == "B":
            dfs, metas = [], []
            esp = [EspecPolitica("saw", n_max=50, rotulo="SAW-Nmax50"), EspecPolitica("saw", n_max=200, rotulo="SAW-Nmax200"),
                   EspecPolitica("saw", decaimento=5, rotulo="SAW-D5"), EspecPolitica("saw", decaimento=30, rotulo="SAW-D30")]
            for reg in REG:
                df, _, m = executar_lote(esp, [reg], list(range(30)), 30, ConfigAG())
                dfs.append(df); metas.append(m)
            salvar("B_sensibilidade", dfs, metas)
        elif b == "F":
            # controle de diversidade: 80% dos filhos pós-mudança são imigrantes (= 80/100 de SAW-IMM)
            dfs, metas = [], []
            neq = {"f1000": 36, "f5000": 22}
            for reg in REG:
                esp = [EspecPolitica("fixa", n=20, rotulo="FIX-20-IMM"),
                       EspecPolitica("fixa", n=neq[reg.nome.split("_")[0]], rotulo="FIX-eq-IMM")]
                df, _, m = executar_lote(esp, [reg], list(range(30)), 30, ConfigAG(fracao_imigrantes_na_mudanca=0.8))
                dfs.append(df); metas.append(m)
            salvar("F_controle_imigrantes", dfs, metas)
        elif b == "A2":  # SAW-IMM nas sementes 30..99 (junto com A: 100 sementes)
            dfs, metas = [], []
            for reg in REG:
                df, _, m = executar_lote([EspecPolitica("saw", rotulo="SAW-IMM")], [reg], list(range(30, 100)), 30, ConfigAG(crescimento="imigrantes"))
                dfs.append(df); metas.append(m)
            salvar("A2_imigrantes_30a99", dfs, metas)
        elif b == "F2":  # controles com imigrantes nas sementes 30..99
            dfs, metas = [], []
            neq = {"f1000": 36, "f5000": 22}
            for reg in REG:
                esp = [EspecPolitica("fixa", n=20, rotulo="FIX-20-IMM"),
                       EspecPolitica("fixa", n=neq[reg.nome.split("_")[0]], rotulo="FIX-eq-IMM")]
                df, _, m = executar_lote(esp, [reg], list(range(30, 100)), 30, ConfigAG(fracao_imigrantes_na_mudanca=0.8))
                dfs.append(df); metas.append(m)
            salvar("F2_controle_30a99", dfs, metas)
        elif b == "G":  # varredura da fração de imigrantes em FIX-20 (30 sementes)
            dfs, metas = [], []
            for fr in (0.2, 0.5):
                for reg in REG:
                    esp = [EspecPolitica("fixa", n=20, rotulo=f"FIX-20-IMM{int(fr*100)}")]
                    df, _, m = executar_lote(esp, [reg], list(range(30)), 30, ConfigAG(fracao_imigrantes_na_mudanca=fr))
                    dfs.append(df); metas.append(m)
            salvar("G_fracao", dfs, metas)
        elif b == "C":
            bloco_com_eq("C_deap2", list(range(30)), bench_kw=dict(correlacao=0.5, refletir_altura_largura=True, altura_inicial_uniforme=50.0))
        elif b == "D":
            bloco_com_eq("D_sementes30a99", list(range(30, 100)))
        elif b == "E":
            bloco_com_eq("E_dim10", list(range(30)), dimensao=10, n_picos=20)
        print(f"bloco {b} ok ({time.time()-t0:.0f}s)", flush=True)

if __name__ == "__main__":
    principal()
