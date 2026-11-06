from pathlib import Path
import numpy as np, pandas as pd
from scipy import stats
from analisar import holm, fmt

import os
EXT = Path("resultados_ext")
TAB = Path(os.environ.get("AGDIN_TABELAS", "saida/tabelas"))
TAB.mkdir(parents=True, exist_ok=True)
OUT = EXT / "analise"; OUT.mkdir(parents=True, exist_ok=True)
REGS = ["f1000_s1", "f1000_s5", "f1000_s10", "f5000_s1", "f5000_s5", "f5000_s10"]
RNOME = {"f1000_s1": "1000; 1", "f1000_s5": "1000; 5", "f1000_s10": "1000; 10",
         "f5000_s1": "5000; 1", "f5000_s5": "5000; 5", "f5000_s10": "5000; 10"}
rng = np.random.default_rng(0)

def carregar(caminho):
    d = pd.read_csv(caminho); d = d[d.ambiente > 0]
    return d.groupby(["regime", "variante", "semente"]).erro_offline.mean().rename("EO").reset_index()

def boot(x, B=10000):
    m = rng.choice(x, size=(B, len(x)), replace=True).mean(axis=1)
    return np.percentile(m, [2.5, 97.5])

def par(S, reg, a, b):
    x = S[(S.regime == reg) & (S.variante == a)].set_index("semente").EO
    y = S[(S.regime == reg) & (S.variante == b)].set_index("semente").EO
    dif = (x - y.reindex(x.index)).dropna().values
    lo, hi = boot(dif)
    p = stats.wilcoxon(dif).pvalue if np.any(dif != 0) else 1.0
    return dict(regime=reg, a=a, b=b, n=len(dif), dif_media=dif.mean(), ic_inf=lo, ic_sup=hi,
                dif_mediana=np.median(dif), prop_a_melhor=(dif < 0).mean(), p=p)

def comparar(S, a, bs, regs=REGS):
    L = [par(S, r, a, b) for r in regs for b in bs]
    df = pd.DataFrame(L); df["p_holm"] = holm(df.p.values); return df

def celula(r):
    return f"{fmt(r.dif_media)} [{fmt(r.ic_inf)}; {fmt(r.ic_sup)}]"

def tabela_dif(df, bs, arq, rotulos=None):
    L = [r"\begin{tabular}{@{}l" + "c" * len(bs) + r"@{}}", r"\toprule",
         r"$\vartheta$; $s$ & " + " & ".join(f"SAW $-$ {b}" for b in bs) + r" \\", r"\midrule"]
    for reg in REGS:
        sub = df[df.regime == reg]
        L.append(RNOME[reg] + " & " + " & ".join(celula(sub[sub.b == b].iloc[0]) for b in bs) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]
    (TAB / arq).write_text("\n".join(L) + "\n")

principal_csv = Path("resultados/ambientes.csv")
S0 = carregar(principal_csv)
BS = ["FIX-eq", "FIX-20", "FIX-50", "FIX-100"]

# D: 100 sementes (principal + 30..99)
if (EXT / "D_sementes30a99/ambientes.csv").exists():
    SD = pd.concat([S0, carregar(EXT / "D_sementes30a99/ambientes.csv")])
    dD = comparar(SD, "SAW", BS); dD.to_csv(OUT / "D_100sementes.csv", index=False)
    tabela_dif(dD, BS, "tab_ext_100sementes.tex")
    r = SD.groupby(["regime", "variante"]).EO.agg(["mean", "std"]).round(3); r.to_csv(OUT / "D_resumo.csv")
    print("D\n", dD.round(3).to_string())
# C e E
for blk, arq in (("C_deap2", "tab_ext_deap2.tex"), ("E_dim10", "tab_ext_dim10.tex")):
    f = EXT / blk / "ambientes.csv"
    if f.exists():
        S = carregar(f); d = comparar(S, "SAW", BS); d.to_csv(OUT / f"{blk}.csv", index=False)
        S.groupby(["regime", "variante"]).EO.agg(["mean", "std"]).round(3).to_csv(OUT / f"{blk}_resumo.csv")
        tabela_dif(d, BS, arq); print(blk, "\n", d.round(3).to_string())
# A: imigrantes (linha de base = resultados principais, mesmas sementes)
f = EXT / "A_imigrantes/ambientes.csv"
if f.exists():
    S = pd.concat([S0, carregar(f)])
    d = comparar(S, "SAW-IMM", ["SAW", "FIX-eq", "FIX-20", "FIX-100"]); d.to_csv(OUT / "A_imigrantes.csv", index=False)
    print("A\n", d.round(3).to_string())
# B: sensibilidade
f = EXT / "B_sensibilidade/ambientes.csv"
if f.exists():
    S = pd.concat([S0, carregar(f)]); m = S.groupby(["regime", "variante"]).EO.mean().unstack()
    cols = ["SAW", "SAW-Nmax50", "SAW-Nmax200", "SAW-D5", "SAW-D30", "FIX-20", "FIX-eq"]
    m[cols].round(3).to_csv(OUT / "B_sensibilidade_medias.csv")
    L = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule",
         r"$\vartheta$; $s$ & SAW & $N_{\max}$ 50 & $N_{\max}$ 200 & $D$ 5 & $D$ 30 & FIX-20 & FIX-eq \\", r"\midrule"]
    for reg in REGS: L.append(RNOME[reg] + " & " + " & ".join(fmt(m.loc[reg, c]) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_sensib.tex").write_text("\n".join(L) + "\n")
    dd = pd.DataFrame([par(S, r, v, "FIX-20") for r in REGS for v in cols[:5]])
    dd.to_csv(OUT / "B_vs_FIX20.csv", index=False); print("B\n", m[cols].round(2).to_string()); print(dd.round(3).to_string())
# limiar de recuperação (dados principais)
d = pd.read_csv(principal_csv); d = d[d.ambiente > 0]
L = [r"\begin{tabular}{@{}llcccc@{}}", r"\toprule", r"$\vartheta$ & Limiar & SAW & FIX-eq & FIX-20 & FIX-100 \\", r"\midrule"]
rows = []
for th in (10, 5, 2, 1):
    for fr in (1000, 5000):
        sub = d[d.mudanca_a_cada == fr]
        vals = [sub[sub.variante == v][f"rec_{th}"].notna().mean() for v in ("SAW", "FIX-eq", "FIX-20", "FIX-100")]
        rows.append((fr, th, vals))
for fr in (1000, 5000):
    for th, vals in [(t, v) for f_, t, v in rows if f_ == fr]:
        L.append(f"{fr} & {th} & " + " & ".join(fmt(x) for x in vals) + r" \\")
L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_limiar.tex").write_text("\n".join(L) + "\n")
print("limiar", rows)

# F: controle com a mesma injeção de imigrantes em populações fixas
f = EXT / "F_controle_imigrantes/ambientes.csv"
if f.exists() and (EXT / "A_imigrantes/ambientes.csv").exists():
    S = pd.concat([S0, carregar(EXT / "A_imigrantes/ambientes.csv"), carregar(f)])
    d = comparar(S, "SAW-IMM", ["FIX-eq-IMM", "FIX-20-IMM"]); d.to_csv(OUT / "F_controle.csv", index=False)
    m = S.groupby(["regime", "variante"]).EO.mean().unstack()
    cols = ["SAW", "SAW-IMM", "FIX-eq", "FIX-eq-IMM", "FIX-20", "FIX-20-IMM"]
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"$\vartheta$; $s$ & SAW & SAW-IMM & FIX-eq & FIX-eq-IMM & FIX-20 & FIX-20-IMM \\", r"\midrule"]
    for reg in REGS: L.append(RNOME[reg] + " & " + " & ".join(fmt(m.loc[reg, c]) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_imigrantes.tex").write_text("\n".join(L) + "\n")
    L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"$\vartheta$; $s$ & SAW-IMM $-$ FIX-eq-IMM & SAW-IMM $-$ FIX-20-IMM \\", r"\midrule"]
    for reg in REGS:
        a_ = d[(d.regime == reg) & (d.b == "FIX-eq-IMM")].iloc[0]; b_ = d[(d.regime == reg) & (d.b == "FIX-20-IMM")].iloc[0]
        L.append(f"{RNOME[reg]} & {celula(a_)} & {celula(b_)} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_controle_imig.tex").write_text("\n".join(L) + "\n")
    m.round(3).to_csv(OUT / "A_F_medias.csv"); print("F\n", d.round(3).to_string()); print(m[["SAW","SAW-IMM","FIX-eq","FIX-eq-IMM","FIX-20","FIX-20-IMM"]].round(2).to_string())


# ---------------------------------------------------------------------------
# Fatorial tamanho x diversidade com 100 sementes (A+A2, F+F2), fração (G), dispersão
# ---------------------------------------------------------------------------
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

FIG = Path(os.environ.get("AGDIN_FIGURAS", "saida/figuras"))
FIG.mkdir(parents=True, exist_ok=True)
def tem(b): return (EXT / b / "ambientes.csv").exists()

if all(tem(b) for b in ("A_imigrantes", "A2_imigrantes_30a99", "F_controle_imigrantes", "F2_controle_30a99", "D_sementes30a99")):
    S = pd.concat([S0, carregar(EXT / "D_sementes30a99/ambientes.csv"),            # SAW, FIX-* (100 sementes)
                   carregar(EXT / "A_imigrantes/ambientes.csv"), carregar(EXT / "A2_imigrantes_30a99/ambientes.csv"),
                   carregar(EXT / "F_controle_imigrantes/ambientes.csv"), carregar(EXT / "F2_controle_30a99/ambientes.csv")])
    S = S.drop_duplicates(["regime", "variante", "semente"])
    S.to_csv(OUT / "fatorial_por_semente_100.csv", index=False)
    m = S.groupby(["regime", "variante"]).EO.agg(["mean", "std", "count"])
    cols = ["SAW", "SAW-IMM", "FIX-eq", "FIX-eq-IMM", "FIX-20", "FIX-20-IMM"]
    mm = m["mean"].unstack()
    L = [r"\begin{tabular}{@{}lcccccc@{}}", r"\toprule",
         r"$\vartheta$; $s$ & SAW & SAW-IMM & FIX-eq & FIX-eq-IMM & FIX-20 & FIX-20-IMM \\", r"\midrule"]
    for reg in REGS: L.append(RNOME[reg] + " & " + " & ".join(fmt(mm.loc[reg, c]) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_imigrantes.tex").write_text("\n".join(L) + "\n")
    # comparações principais do fatorial (100 sementes)
    pares = [("SAW-IMM", "SAW"), ("FIX-eq-IMM", "FIX-eq"), ("FIX-20-IMM", "FIX-20"),
             ("SAW-IMM", "FIX-eq-IMM"), ("SAW-IMM", "FIX-20-IMM")]
    rows = [par(S, r, a, b) for r in REGS for a, b in pares]
    dfF = pd.DataFrame(rows); dfF["p_holm"] = holm(dfF.p.values); dfF.to_csv(OUT / "fatorial_100.csv", index=False)
    def cel(r, a, b):
        x = dfF[(dfF.regime == r) & (dfF.a == a) & (dfF.b == b)].iloc[0]; return celula(x)
    L = [r"\begin{tabular}{@{}lccc@{}}", r"\toprule",
         r"$\vartheta$; $s$ & SAW-IMM $-$ SAW & FIX-20-IMM $-$ FIX-20 & FIX-eq-IMM $-$ FIX-eq \\", r"\midrule"]
    for r in REGS: L.append(f"{RNOME[r]} & {cel(r,'SAW-IMM','SAW')} & {cel(r,'FIX-20-IMM','FIX-20')} & {cel(r,'FIX-eq-IMM','FIX-eq')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_efeito_imig.tex").write_text("\n".join(L) + "\n")
    L = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule", r"$\vartheta$; $s$ & SAW-IMM $-$ FIX-eq-IMM & SAW-IMM $-$ FIX-20-IMM \\", r"\midrule"]
    for r in REGS: L.append(f"{RNOME[r]} & {cel(r,'SAW-IMM','FIX-eq-IMM')} & {cel(r,'SAW-IMM','FIX-20-IMM')} \\\\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_controle_imig.tex").write_text("\n".join(L) + "\n")
    print("FATORIAL 100 sementes\n", mm[cols].round(2).to_string()); print(dfF.round(3).to_string())
    # figura: média e IC95% por variante (bootstrap), um painel por regime
    cor = {"SAW": "#2a78d6", "FIX-eq": "#eb6834", "FIX-20": "#1baf7a"}
    fig, axs = plt.subplots(1, 6, figsize=(13, 3.2), sharey=True)
    for ax, reg in zip(axs, REGS):
        for i, c in enumerate(cols):
            x = S[(S.regime == reg) & (S.variante == c)].EO.values
            lo, hi = boot(x, 4000); base = c.replace("-IMM", "")
            ax.errorbar(i, x.mean(), yerr=[[x.mean() - lo], [hi - x.mean()]], fmt="o" if "IMM" not in c else "s",
                        color=cor[base], mfc=cor[base] if "IMM" not in c else "white", capsize=2, ms=5, lw=1.2)
        ax.set_title(RNOME[reg].replace(";", ",") , fontsize=9); ax.set_xticks(range(6))
        ax.set_xticklabels(["S", "S+I", "Eq", "Eq+I", "20", "20+I"], fontsize=8); ax.grid(axis="y", alpha=.25)
    axs[0].set_ylabel("Erro off-line (média e IC 95%)")
    fig.text(0.5, 0.0, "S: SAW; Eq: FIX-eq; 20: FIX-20; +I: com imigrantes aleatórios na mudança (marcador vazado). Títulos: frequência, severidade.", ha="center", fontsize=8)
    fig.tight_layout(rect=(0, 0.05, 1, 1)); fig.savefig(FIG / "fig_fatorial_imigrantes.pdf"); fig.savefig(FIG / "fig_fatorial_imigrantes.png", dpi=200)

# G: fração de imigrantes (FIX-20)
if tem("G_fracao") and tem("F_controle_imigrantes"):
    S = pd.concat([S0, carregar(EXT / "F_controle_imigrantes/ambientes.csv"), carregar(EXT / "G_fracao/ambientes.csv")])
    mm = S.groupby(["regime", "variante"]).EO.mean().unstack()
    cols = ["FIX-20", "FIX-20-IMM20", "FIX-20-IMM50", "FIX-20-IMM"]
    L = [r"\begin{tabular}{@{}lcccc@{}}", r"\toprule", r"$\vartheta$; $s$ & 0\% & 20\% & 50\% & 80\% \\", r"\midrule"]
    for reg in REGS: L.append(RNOME[reg] + " & " + " & ".join(fmt(mm.loc[reg, c]) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_fracao.tex").write_text("\n".join(L) + "\n")
    mm[cols].round(3).to_csv(OUT / "G_fracao_medias.csv"); print("G\n", mm[cols].round(2).to_string())

# dispersão (30 sementes): populações com e sem imigrantes
if tem("A_imigrantes") and tem("F_controle_imigrantes"):
    ds = []
    for f in (principal_csv, EXT / "A_imigrantes/ambientes.csv", EXT / "F_controle_imigrantes/ambientes.csv"):
        x = pd.read_csv(f); x = x[x.ambiente > 0]; ds.append(x[["variante", "mudanca_a_cada", "severidade", "semente", "dispersao"]])
    ds = pd.concat(ds); t = ds.groupby(["mudanca_a_cada", "variante"]).dispersao.mean().unstack()
    cols = ["SAW", "SAW-IMM", "FIX-eq", "FIX-eq-IMM", "FIX-20", "FIX-20-IMM", "FIX-100"]
    L = [r"\begin{tabular}{@{}lccccccc@{}}", r"\toprule", r"$\vartheta$ & SAW & SAW-IMM & FIX-eq & FIX-eq-IMM & FIX-20 & FIX-20-IMM & FIX-100 \\", r"\midrule"]
    for fr in (1000, 5000): L.append(f"{fr} & " + " & ".join(fmt(t.loc[fr, c]) for c in cols) + r" \\")
    L += [r"\bottomrule", r"\end{tabular}"]; (TAB / "tab_ext_dispersao.tex").write_text("\n".join(L) + "\n")
    t[cols].round(3).to_csv(OUT / "dispersao.csv"); print("DISP\n", t[cols].round(2).to_string())
