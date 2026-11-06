"""Políticas de tamanho populacional.

O tamanho populacional é o ÚNICO fator experimental deste estudo. Todas as
políticas compartilham o mesmo AG, os mesmos operadores e o mesmo tratamento
de mudança (reavaliação da população quando a mudança é sinalizada).
"""

from __future__ import annotations

from dataclasses import dataclass


class PoliticaTamanho:
    """Interface: decide quantos indivíduos terá a próxima geração."""

    nome: str = "base"

    def tamanho_inicial(self) -> int:
        raise NotImplementedError

    def proximo_tamanho(self, mudou: bool, tamanho_atual: int) -> int:
        raise NotImplementedError

    @property
    def limites(self) -> tuple[int, int]:
        raise NotImplementedError


@dataclass
class Fixa(PoliticaTamanho):
    n: int = 50
    nome: str = ""

    def __post_init__(self) -> None:
        if self.n < 4:
            raise ValueError("n deve ser >= 4")
        if not self.nome:
            self.nome = f"FIX-{self.n}"

    def tamanho_inicial(self) -> int:
        return self.n

    def proximo_tamanho(self, mudou: bool, tamanho_atual: int) -> int:
        return self.n

    @property
    def limites(self) -> tuple[int, int]:
        return (self.n, self.n)


@dataclass
class DenteDeSerra(PoliticaTamanho):
    """Política reativa e bidirecional, disparada pela sinalização de mudança.

    Ao ser sinalizada uma mudança, o tamanho vai a ``n_max``; nas gerações
    seguintes decai linearmente até ``n_min`` ao longo de ``geracoes_decaimento``
    gerações e permanece em ``n_min`` até a próxima mudança.

    A sinalização é um "oráculo" fornecido pelo benchmark: o estudo avalia a
    RESPOSTA à mudança, não a capacidade de detectá-la.
    """

    n_min: int = 20
    n_max: int = 100
    geracoes_decaimento: int = 10
    nome: str = ""

    def __post_init__(self) -> None:
        if self.n_min < 4 or self.n_max < self.n_min:
            raise ValueError("exige-se 4 <= n_min <= n_max")
        if self.geracoes_decaimento < 1:
            raise ValueError("geracoes_decaimento deve ser >= 1")
        if not self.nome:
            self.nome = f"SAW-{self.n_min}-{self.n_max}-D{self.geracoes_decaimento}"
        self._k = self.geracoes_decaimento  # começa já no mínimo; inicial é n_max

    def tamanho_inicial(self) -> int:
        self._k = 0
        return self.n_max

    def proximo_tamanho(self, mudou: bool, tamanho_atual: int) -> int:
        if mudou:
            self._k = 0
            return self.n_max
        self._k = min(self._k + 1, self.geracoes_decaimento)
        frac = self._k / self.geracoes_decaimento
        return int(round(self.n_max - (self.n_max - self.n_min) * frac))

    @property
    def limites(self) -> tuple[int, int]:
        return (self.n_min, self.n_max)
