from dataclasses import dataclass, field

import numpy as np

# Limiares de erro (unidades de aptidão) usados para o tempo de recuperação.
LIMIARES_RECUPERACAO = (10.0, 5.0, 2.0, 1.0)
N_FAIXAS = 20  # resolução da curva de erro dentro de cada ambiente


@dataclass
class RegistroAmbiente:
    indice: int
    erro_offline: float          # média do erro corrente ao longo do ambiente
    erro_antes_mudanca: float    # erro do melhor encontrado ao fim do ambiente
    tempo_recuperacao: dict      # {limiar: avaliações até erro <= limiar, ou NaN}
    curva_erro: np.ndarray       # erro médio por faixa relativa do ambiente


class MovingPeaks:
    def __init__(
        self,
        dimensao: int = 5,
        n_picos: int = 10,
        mudanca_a_cada: int = 5000,
        severidade_deslocamento: float = 1.0,
        severidade_altura: float = 7.0,
        severidade_largura: float = 1.0,
        correlacao: float = 0.0,
        limite_inferior: float = 0.0,
        limite_superior: float = 100.0,
        faixa_altura: tuple[float, float] = (30.0, 70.0),
        faixa_largura: tuple[float, float] = (1.0, 12.0),
        semente: int = 0,
        refletir_altura_largura: bool = False,
        altura_inicial_uniforme: float | None = None,
    ) -> None:
        if dimensao < 1 or n_picos < 1:
            raise ValueError("dimensao e n_picos devem ser >= 1")
        if mudanca_a_cada < 1:
            raise ValueError("mudanca_a_cada deve ser >= 1")
        if severidade_deslocamento < 0:
            raise ValueError("severidade_deslocamento deve ser >= 0")
        if not 0.0 <= correlacao <= 1.0:
            raise ValueError("correlacao deve estar em [0, 1]")
        if limite_superior <= limite_inferior:
            raise ValueError("limite_superior deve ser maior que limite_inferior")

        self.d = dimensao
        self.m = n_picos
        self.mudanca_a_cada = mudanca_a_cada
        self.s = severidade_deslocamento
        self.h_sev = severidade_altura
        self.w_sev = severidade_largura
        self.lam = correlacao
        self.lo = limite_inferior
        self.hi = limite_superior
        self.h_min, self.h_max = faixa_altura
        self.w_min, self.w_max = faixa_largura
        self.refletir_hw = refletir_altura_largura
        self.rng = np.random.default_rng(semente)

        self.centros = self.rng.uniform(self.lo, self.hi, size=(self.m, self.d))
        self.alturas = self.rng.uniform(self.h_min, self.h_max, size=self.m)
        self.larguras = self.rng.uniform(self.w_min, self.w_max, size=self.m)
        if altura_inicial_uniforme is not None:  # como o cenário 2 do DEAP
            self.alturas = np.full(self.m, float(altura_inicial_uniforme))
        self.velocidades = self._direcao_aleatoria() * self.s

        self.indice_ambiente = 0
        self.avaliacoes_total = 0
        self._aval_no_ambiente = 0
        self.registros: list[RegistroAmbiente] = []
        self._reiniciar_estatisticas_ambiente()

    # ------------------------------------------------------------------ #
    # Função objetivo
    # ------------------------------------------------------------------ #
    @property
    def otimo(self) -> float:
        """Valor do ótimo global do ambiente corrente (conhecido pelo benchmark)."""
        return float(np.max(self.alturas))

    def _f(self, X: np.ndarray) -> np.ndarray:
        dist = np.linalg.norm(X[:, None, :] - self.centros[None, :, :], axis=2)
        return np.max(self.alturas[None, :] - self.larguras[None, :] * dist, axis=1)

    def avaliar(self, X: np.ndarray) -> np.ndarray:
        """Avalia um lote de indivíduos, contando avaliações e aplicando as
        mudanças ambientais exatamente quando o contador as exige.

        Se o lote atravessar uma fronteira de mudança, os indivíduos após a
        fronteira são avaliados no ambiente novo.
        """
        X = np.atleast_2d(np.asarray(X, dtype=float))
        n = len(X)
        saida = np.empty(n)
        i = 0
        while i < n:
            restante = self.mudanca_a_cada - self._aval_no_ambiente
            k = min(n - i, restante)
            valores = self._f(X[i : i + k])
            saida[i : i + k] = valores
            self._registrar(valores)
            self._aval_no_ambiente += k
            self.avaliacoes_total += k
            i += k
            if self._aval_no_ambiente == self.mudanca_a_cada:
                self._encerrar_ambiente()
                self._mudar()
        return saida

    # ------------------------------------------------------------------ #
    # Estatísticas internas (erro off-line, recuperação)
    # ------------------------------------------------------------------ #
    def _reiniciar_estatisticas_ambiente(self) -> None:
        self._melhor = -np.inf
        self._soma_erro = 0.0
        self._rec = {lim: np.nan for lim in LIMIARES_RECUPERACAO}
        self._soma_faixa = np.zeros(N_FAIXAS)
        self._cont_faixa = np.zeros(N_FAIXAS)
        self._ultimo_erro = np.nan

    def _registrar(self, valores: np.ndarray) -> None:
        opt = self.otimo
        acumulado = np.maximum.accumulate(np.concatenate(([self._melhor], valores)))[1:]
        erros = opt - acumulado
        self._melhor = float(acumulado[-1])
        self._soma_erro += float(erros.sum())
        self._ultimo_erro = float(erros[-1])
        posicoes = self._aval_no_ambiente + np.arange(1, len(valores) + 1)
        for lim in LIMIARES_RECUPERACAO:
            if np.isnan(self._rec[lim]):
                idx = np.flatnonzero(erros <= lim)
                if idx.size:
                    self._rec[lim] = float(posicoes[idx[0]])
        faixas = np.minimum(
            ((posicoes - 1) * N_FAIXAS) // self.mudanca_a_cada, N_FAIXAS - 1
        ).astype(int)
        np.add.at(self._soma_faixa, faixas, erros)
        np.add.at(self._cont_faixa, faixas, 1.0)

    def _encerrar_ambiente(self) -> None:
        curva = np.divide(
            self._soma_faixa,
            self._cont_faixa,
            out=np.full(N_FAIXAS, np.nan),
            where=self._cont_faixa > 0,
        )
        self.registros.append(
            RegistroAmbiente(
                indice=self.indice_ambiente,
                erro_offline=self._soma_erro / self.mudanca_a_cada,
                erro_antes_mudanca=self._ultimo_erro,
                tempo_recuperacao=dict(self._rec),
                curva_erro=curva,
            )
        )

    # ------------------------------------------------------------------ #
    # Dinâmica (Yazdani et al., 2021b, Eqs. PB-11 a PB-14)
    # ------------------------------------------------------------------ #
    def _direcao_aleatoria(self) -> np.ndarray:
        v = self.rng.uniform(-0.5, 0.5, size=(self.m, self.d))
        return self._normalizar(v)

    @staticmethod
    def _normalizar(v: np.ndarray) -> np.ndarray:
        norma = np.linalg.norm(v, axis=1, keepdims=True)
        norma[norma == 0.0] = 1.0
        return v / norma

    def _limitar(self, x: np.ndarray, lo: float, hi: float) -> np.ndarray:
        if not self.refletir_hw:
            return np.clip(x, lo, hi)
        x = np.where(x > hi, 2.0 * hi - x, x)
        x = np.where(x < lo, 2.0 * lo - x, x)
        return np.clip(x, lo, hi)

    def _mudar(self) -> None:
        # Alturas e larguras: passeio gaussiano, limitado às faixas do problema
        # (truncamento por padrão; reflexão simples, como no DEAP, se solicitado).
        self.alturas = self._limitar(
            self.alturas + self.h_sev * self.rng.normal(size=self.m), self.h_min, self.h_max
        )
        self.larguras = self._limitar(
            self.larguras + self.w_sev * self.rng.normal(size=self.m), self.w_min, self.w_max
        )
        # Centros: deslocamento de norma s, com correlação lambda entre mudanças.
        u = self.rng.uniform(-0.5, 0.5, size=(self.m, self.d))
        v = self._normalizar((1.0 - self.lam) * u + self.lam * self.velocidades) * self.s
        novos = self.centros + v
        acima = novos > self.hi
        abaixo = novos < self.lo
        novos[acima] = 2.0 * self.hi - novos[acima]
        novos[abaixo] = 2.0 * self.lo - novos[abaixo]
        v[acima | abaixo] *= -1.0  # reflexão nas bordas
        self.centros = np.clip(novos, self.lo, self.hi)
        self.velocidades = v

        self.indice_ambiente += 1
        self._aval_no_ambiente = 0
        self._reiniciar_estatisticas_ambiente()
