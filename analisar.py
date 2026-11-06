import argparse
import itertools
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from agdin.ag import AlgoritmoGenetico, ConfigAG
from agdin.benchmark import MovingPeaks, N_FAIXAS
from agdin.politicas import DenteDeSerra

ORDEM = ["SAW", "FIX-eq", "FIX-20", "FIX-50", "FIX-100"]
CORES = {
    "SAW": "#2a78d6", "FIX-eq": "#eb6834", "FIX-20": "#1baf7a",
    "FIX-50": "#6250d6", "FIX-100": "#e34948",
}
MARC = {"SAW": "o", "FIX-eq": "s", "FIX-20": "^", "FIX-50": "D", "FIX-100": "v"}
LIM_REC = 5


def fmt(x: float, casas: int = 2) -> str:
    return f"{x:.{casas}f}".replace(".", ",")


def holm(p: np.ndarray) -> np.ndarray:
    ordem = np.argsort(p)
    m = len(p)
    aj = np.empty(m)
    maximo = 0.0
    for rank, i in enumerate(ordem):
        valor = min(1.0, (m - rank) * p[i])
        maximo = max(maximo, valor)
        aj[i] = maximo
    return aj


def principal() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--resultados", default="resultados")
    ap.add_argument("--figuras", default="saida/figuras")
    ap.add_argument("--tabelas", default="saida/tabelas")
    args = ap.parse_args()
    res, figs, tabs = Path(args.resultados), Path(args.figuras), Path(args.tabelas)
    figs.mkdir(parents=True, exist_ok=True)
    tabs.mkdir(parents=True, exist_ok=True)

    d = pd.read_csv(res / "ambientes.csv")
    d = d[d.ambiente > 0].copy()
    d["rec_cens"] = d[f"rec_{LIM_REC}"].fillna(d["mudanca_a_cada"])
    d["recuperou"] = d[f"rec_{LIM_REC}"].notna().astype(float)

    chave = ["regime", "mudanca_a_cada", "severidade", "variante", "semente"]
    s = (
        d.groupby(chave)
        .agg(
            EO=("erro_offline", "mean"),
            BBC=("erro_antes_mudanca", "mean"),
            rec=("rec_cens", "mean"),
            taxa_rec=("recuperou", "mean"),
            tam=("tamanho_medio", "mean"),
            disp=("dispersao", "mean"),
            ger=("geracoes", "mean"),
        )
        .reset_index()
    )
    s.to_csv(res / "por_semente.csv", index=False)

    regimes = (
        s[["regime", "mudanca_a_cada", "severidade"]]
        .drop_duplicates()
        .sort_values(["mudanca_a_cada", "severidade"])
        .reset_index(drop=True)
    )

    # ---------------- resumo ---------------- #
    resumo = (
        s.groupby(["regime", "mudanca_a_cada", "severidade", "variante"])
        .agg(
            EO_m=("EO", "mean"), EO_dp=("EO", "std"),
            BBC_m=("BBC", "mean"), BBC_dp=("BBC", "std"),
            rec_m=("rec", "mean"), rec_dp=("rec", "std"),
            taxa_m=("taxa_rec", "mean"), taxa_dp=("taxa_rec", "std"),
            tam_m=("tam", "mean"), disp_m=("disp", "mean"), ger_m=("ger", "mean"),
            n=("EO", "size"),
        )
        .reset_index()
    )
    resumo.to_csv(res / "resumo.csv", index=False)

    # ---------------- testes ---------------- #
    linhas_friedman, linhas_pares = [], []
    for _, r in regimes.iterrows():
        sub = s[s.regime == r.regime].pivot(index="semente", columns="variante", values="EO")
        sub = sub[ORDEM]
        chi, p = stats.friedmanchisquare(*[sub[v] for v in ORDEM])
        linhas_friedman.append({"regime": r.regime, "chi2": chi, "p": p, "n": len(sub)})
        for base in ["FIX-eq", "FIX-20", "FIX-50", "FIX-100"]:
            dif = sub["SAW"] - sub[base]
            if np.allclose(dif, 0):
                p_w = 1.0
            else:
                p_w = stats.wilcoxon(sub["SAW"], sub[base], alternative="two-sided").pvalue
            rng_b = np.random.default_rng(12345)
            dd = dif.to_numpy()
            boot = rng_b.choice(dd, size=(10000, len(dd)), replace=True).mean(axis=1)
            linhas_pares.append(
                {
                    "regime": r.regime, "contra": base,
                    "dif_media": float(dd.mean()),
                    "ic_inf": float(np.percentile(boot, 2.5)),
                    "ic_sup": float(np.percentile(boot, 97.5)),
                    "dif_mediana": float(np.median(dd)),
                    "prop_saw_melhor": float((dd < 0).mean()),
                    "p": float(p_w),
                }
            )
    pares = pd.DataFrame(linhas_pares)
    pares["p_holm"] = holm(pares["p"].to_numpy())
    pares.to_csv(res / "comparacoes_saw.csv", index=False)
    pd.DataFrame(linhas_friedman).to_csv(res / "friedman.csv", index=False)

    # ---------------- tabelas LaTeX ---------------- #
    def tabela_principal() -> str:
        cab = " & ".join(f"\\textbf{{{v}}}" for v in ORDEM)
        out = [
            "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
            f"$\\vartheta$ & $s$ & {cab} \\\\", "\\midrule",
        ]
        for _, r in regimes.iterrows():
            cel = []
            sub = resumo[resumo.regime == r.regime].set_index("variante")
            melhor = sub.EO_m.idxmin()
            for v in ORDEM:
                txt = f"{fmt(sub.loc[v, 'EO_m'])} ({fmt(sub.loc[v, 'EO_dp'])})"
                cel.append(f"\\textbf{{{txt}}}" if v == melhor else txt)
            out.append(f"{int(r.mudanca_a_cada)} & {fmt(r.severidade, 0)} & " + " & ".join(cel) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)

    def tabela_secundaria(col_m: str, col_dp: str | None, casas: int) -> str:
        cab = " & ".join(f"\\textbf{{{v}}}" for v in ORDEM)
        out = [
            "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
            f"$\\vartheta$ & $s$ & {cab} \\\\", "\\midrule",
        ]
        for _, r in regimes.iterrows():
            sub = resumo[resumo.regime == r.regime].set_index("variante")
            cel = []
            for v in ORDEM:
                t = fmt(sub.loc[v, col_m], casas)
                if col_dp:
                    t += f" ({fmt(sub.loc[v, col_dp], casas)})"
                cel.append(t)
            out.append(f"{int(r.mudanca_a_cada)} & {fmt(r.severidade, 0)} & " + " & ".join(cel) + " \\\\")
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)

    def tabela_testes() -> str:
        out = [
            "\\begin{tabular}{@{}llccccc@{}}", "\\toprule",
            "$\\vartheta$ & $s$ & \\textbf{contra} & $\\Delta$ média [IC 95\\%] & $\\Delta$ mediana & "
            "SAW melhor & $p$ (Holm) \\\\",
            "\\midrule",
        ]
        for _, r in regimes.iterrows():
            sub = pares[pares.regime == r.regime]
            for k, (_, q) in enumerate(sub.iterrows()):
                pre = f"{int(r.mudanca_a_cada)} & {fmt(r.severidade, 0)}" if k == 0 else " & "
                p_txt = "$<0{,}001$" if q.p_holm < 0.001 else fmt(q.p_holm, 3)
                ic = f"{fmt(q.dif_media)} [{fmt(q.ic_inf)}; {fmt(q.ic_sup)}]"
                out.append(
                    f"{pre} & {q.contra} & {ic} & {fmt(q.dif_mediana)} & "
                    f"{fmt(100 * q.prop_saw_melhor, 0)}\\% & {p_txt} \\\\"
                )
            out.append("\\midrule" if r.name < len(regimes) - 1 else "")
        out = [l for l in out if l != ""]
        if out[-1] == "\\midrule":
            out.pop()
        out += ["\\bottomrule", "\\end{tabular}"]
        return "\n".join(out)

    (tabs / "tab_erro_offline.tex").write_text(tabela_principal(), encoding="utf-8")
    (tabs / "tab_taxa_recuperacao.tex").write_text(tabela_secundaria("taxa_m", "taxa_dp", 2), encoding="utf-8")
    (tabs / "tab_tamanho_geracoes.tex").write_text(
        tabela_secundaria("tam_m", None, 1), encoding="utf-8"
    )
    (tabs / "tab_geracoes.tex").write_text(tabela_secundaria("ger_m", None, 1), encoding="utf-8")
    (tabs / "tab_testes.tex").write_text(tabela_testes(), encoding="utf-8")

    # ---------------- figuras ---------------- #
    plt.rcParams.update(
        {
            "font.size": 9, "axes.spines.top": False, "axes.spines.right": False,
            "axes.edgecolor": "#52514e", "axes.labelcolor": "#0b0b0b",
            "xtick.color": "#52514e", "ytick.color": "#52514e",
            "axes.grid": True, "grid.color": "#e6e5e0", "grid.linewidth": 0.6,
            "axes.axisbelow": True, "figure.dpi": 150, "savefig.dpi": 300,
        }
    )
    freqs = sorted(regimes.mudanca_a_cada.unique())
    sevs = sorted(regimes.severidade.unique())

    # Fig 1: caixas de E_O por variante em cada regime
    fig, eixos = plt.subplots(len(freqs), len(sevs), figsize=(7.2, 4.6), sharey=False)
    for i, f in enumerate(freqs):
        for j, sv in enumerate(sevs):
            ax = eixos[i, j]
            nome = f"f{f}_s{sv:g}"
            dados = [s[(s.regime == nome) & (s.variante == v)].EO.to_numpy() for v in ORDEM]
            bp = ax.boxplot(
                dados, patch_artist=True, widths=0.6, showfliers=False,
                medianprops={"color": "#0b0b0b", "linewidth": 1.2},
                whiskerprops={"linewidth": 0.8}, capprops={"linewidth": 0.8},
            )
            for patch, v in zip(bp["boxes"], ORDEM):
                patch.set_facecolor(CORES[v]); patch.set_alpha(0.85); patch.set_edgecolor("#52514e")
            ax.set_xticks(range(1, len(ORDEM) + 1))
            ax.set_xticklabels(ORDEM, rotation=45, ha="right")
            ax.set_title(f"$\\vartheta$={f}, $s$={sv:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Erro off-line")
    fig.tight_layout()
    fig.savefig(figs / "fig_erro_offline.pdf"); fig.savefig(figs / "fig_erro_offline.png")
    plt.close(fig)

    # Fig 2: curvas de erro dentro do ambiente (recuperação)
    dados_c = np.load(res / "curvas.npz")
    fig, eixos = plt.subplots(len(freqs), len(sevs), figsize=(7.2, 4.8), sharey=False)
    for i, f in enumerate(freqs):
        for j, sv in enumerate(sevs):
            ax = eixos[i, j]
            nome = f"f{f}_s{sv:g}"
            x = (np.arange(N_FAIXAS) + 0.5) / N_FAIXAS * f
            for v in ORDEM:
                arr = dados_c[f"{v}|{nome}"][:, 1:, :]   # (sementes, amb>0, faixas)
                y = np.nanmean(arr, axis=(0, 1))
                ax.plot(
                    x, y, color=CORES[v], marker=MARC[v], markersize=3.2, linewidth=1.4,
                    label=v, markevery=2,
                )
            ax.set_title(f"$\\vartheta$={f}, $s$={sv:g}", fontsize=9)
            if j == 0:
                ax.set_ylabel("Erro desde a mudança")
            if i == len(freqs) - 1:
                ax.set_xlabel("Avaliações após a mudança")
    h, l = eixos[0, 0].get_legend_handles_labels()
    fig.legend(h, l, loc="lower center", ncol=len(ORDEM), frameon=False, bbox_to_anchor=(0.5, -0.01))
    fig.tight_layout(rect=(0, 0.05, 1, 1))
    fig.savefig(figs / "fig_curvas_erro.pdf"); fig.savefig(figs / "fig_curvas_erro.png")
    plt.close(fig)

    # Fig 3: trilha do tamanho populacional da política SAW (ilustração, semente 0)
    fig, eixos = plt.subplots(1, 2, figsize=(7.2, 2.4))
    for ax, f in zip(eixos, freqs):
        b = MovingPeaks(mudanca_a_cada=f, severidade_deslocamento=5.0, semente=0)
        ag = AlgoritmoGenetico(DenteDeSerra(20, 100, 10), b, semente=1000, config=ConfigAG())
        r = ag.executar(6)
        t = r.trilha_tamanho
        aval = np.cumsum(t)
        ax.step(aval, t, where="post", color=CORES["SAW"], linewidth=1.4)
        for k in range(1, 6):
            ax.axvline(k * f, color="#9a998f", linestyle=":", linewidth=0.8)
        ax.set_xlim(0, 5 * f)
        ax.set_title(f"$\\vartheta$={f}, $s$=5", fontsize=9)
        ax.set_xlabel("Avaliações (aprox.)")
        ax.set_ylabel("Tamanho da população")
        ax.set_ylim(0, 110)
    fig.tight_layout()
    fig.savefig(figs / "fig_trilha_tamanho.pdf"); fig.savefig(figs / "fig_trilha_tamanho.png")
    plt.close(fig)

    # Impressão de síntese
    pd.set_option("display.width", 200)
    print(resumo[["regime", "variante", "EO_m", "EO_dp", "BBC_m", "taxa_m", "rec_m", "tam_m", "disp_m", "ger_m"]]
          .round(2).to_string(index=False))
    print(pd.DataFrame(linhas_friedman).round(4).to_string(index=False))
    print(pares.round(4).to_string(index=False))


if __name__ == "__main__":
    principal()
