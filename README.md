# Tamanho populacional e imigrantes aleatórios em AGs para otimização dinâmica

O AG é testado em um benchmark de picos móveis (Moving Peaks), sob o mesmo orçamento de avaliações, comparando uma política de tamanho em dente de serra (SAW) com populações fixas e com controles que usam imigrantes aleatórios.

## Instalação

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## Estrutura

| Caminho | Conteúdo |
|---|---|
| `agdin/` | pacote: benchmark (`benchmark.py`), AG (`ag.py`), políticas de tamanho (`politicas.py`), execução em lote (`experimento.py`) |
| `executar_estudo.py` | estudo principal (6 regimes, 5 variantes) |
| `executar_extensoes.py` | extensões exploratórias, em blocos (ver abaixo) |
| `analisar.py`, `analisar_ext.py` | estatísticas, tabelas LaTeX e figuras |
| `validar_mpb_deap.py` | compara o benchmark com o do DEAP |
| `tests/` | 20 testes unitários |
| `resultados/`, `resultados_ext/` | dados brutos e resumos já gerados |
| `validacao_mpb_deap.txt` | saída da validação contra o DEAP |

## Como reproduzir

```bash
python -m unittest discover -s tests                 # testes

python executar_estudo.py --sementes 30 --ambientes 30 --saida resultados
python analisar.py                                   # tabelas e figuras do estudo principal

python executar_extensoes.py A B C D E F G A2 F2
python analisar_ext.py                               # tabelas e figuras das extensões
```

### Blocos das extensões

| Bloco | O que faz |
|---|---|
| A, A2 | SAW com crescimento por imigrantes (SAW-IMM), sementes 0–29 e 30–99 |
| F, F2 | controles de tamanho fixo com imigrantes (FIX-eq-IMM, FIX-20-IMM), idem |
| G | fração de imigrantes (20% e 50%) em FIX-20 |
| B | sensibilidade de SAW (N_max e decaimento, um fator por vez) |
| C | cenário alinhado ao DEAP (λ = 0,5, reflexão, altura inicial 50) |
| D | desenho principal com as sementes 30–99 (com 0–29, dá 100) |
| E | dimensão 10 e 20 picos |

Os blocos A–G e as extensões foram definidos depois de ver os resultados principais, e por isso são exploratórios.

## Validação do benchmark

```bash
python validar_mpb_deap.py | tee validacao_mpb_deap.txt
python validar_mpb_deap.py --sem-deap
```

## Observações

- A mudança do ambiente é sinalizada ao algoritmo pelo próprio benchmark (oráculo).
- O erro off-line é contado por avaliações, e o primeiro ambiente de cada execução é descartado.
- A sequência de ambientes depende só da semente do benchmark, o que permite comparação pareada.
