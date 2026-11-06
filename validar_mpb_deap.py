import argparse, random
import numpy as np
from scipy import stats

ap = argparse.ArgumentParser()
ap.add_argument("--sementes", type=int, default=100)
ap.add_argument("--ambientes", type=int, default=20)
ap.add_argument("--pontos", type=int, default=500)
ap.add_argument("--sem-deap", action="store_true")
args = ap.parse_args()
D, M, LO, HI = 5, 10, 0.0, 100.0

from agdin.benchmark import MovingPeaks
if not args.sem_deap:
    from deap.benchmarks import movingpeaks as mp


def meu(semente, lam, s):
    b = MovingPeaks(dimensao=D, n_picos=M, mudanca_a_cada=args.pontos, severidade_deslocamento=s,
                    correlacao=lam, refletir_altura_largura=True, altura_inicial_uniforme=50.0,
                    semente=semente)
    rng = np.random.default_rng(10_000 + semente)
    alt, desl, larg, eo = [], [], [], []
    for _ in range(args.ambientes):
        opt = b.otimo
        alt.append(opt); larg.append(float(np.mean(b.larguras)))
        antes = b.centros.copy()
        X = rng.uniform(LO, HI, size=(args.pontos, D))
        v = b.avaliar(X)                       # a mudança ocorre após as P avaliações
        eo.append(float(np.mean(opt - np.maximum.accumulate(v))))
        desl.append(float(np.mean(np.linalg.norm(b.centros - antes, axis=1))))
    return np.mean(alt), np.mean(desl), np.mean(larg), np.mean(eo)


def deap_(semente, lam, s):
    sc = dict(mp.SCENARIO_2)
    sc.update(lambda_=lam, move_severity=s, period=0)    # mudanças disparadas à mão
    b = mp.MovingPeaks(dim=D, random=random.Random(semente), **sc)
    rng = np.random.default_rng(20_000 + semente)
    alt, desl, larg, eo = [], [], [], []
    for _ in range(args.ambientes):
        opt = b.globalMaximum()[0]
        alt.append(opt); larg.append(float(np.mean(b.peaks_width)))
        antes = np.array(b.peaks_position, dtype=float)
        X = rng.uniform(LO, HI, size=(args.pontos, D))
        v = np.array([b(list(x), count=False)[0] for x in X])
        eo.append(float(np.mean(opt - np.maximum.accumulate(v))))
        b.changePeaks()
        desl.append(float(np.mean(np.linalg.norm(np.array(b.peaks_position, dtype=float) - antes, axis=1))))
    return np.mean(alt), np.mean(desl), np.mean(larg), np.mean(eo)


NOMES = ["altura do ótimo global", "deslocamento médio por mudança", "largura média", "erro off-line da busca aleatória"]
print(f"Sementes={args.sementes}, ambientes={args.ambientes}, pontos/ambiente={args.pontos}\n")
for lam in (0.0, 0.5):
    for s in (1.0, 5.0, 10.0):
        A = np.array([meu(k, lam, s) for k in range(args.sementes)])
        print(f"== lambda={lam}  s={s}")
        if args.sem_deap:
            for j, n in enumerate(NOMES):
                print(f"   {n:38s} meu: {A[:,j].mean():8.3f} ({A[:,j].std(ddof=1):.3f})")
            continue
        B = np.array([deap_(k, lam, s) for k in range(args.sementes)])
        for j, n in enumerate(NOMES):
            p = stats.mannwhitneyu(A[:, j], B[:, j]).pvalue
            print(f"   {n:38s} meu: {A[:,j].mean():8.3f} ({A[:,j].std(ddof=1):.3f})  "
                  f"DEAP: {B[:,j].mean():8.3f} ({B[:,j].std(ddof=1):.3f})  dif: {A[:,j].mean()-B[:,j].mean():+.3f}  p={p:.3f}")
