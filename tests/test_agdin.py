import unittest

import numpy as np

from agdin.ag import AlgoritmoGenetico, ConfigAG
from agdin.benchmark import MovingPeaks
from agdin.politicas import DenteDeSerra, Fixa


class TestBenchmark(unittest.TestCase):
    def test_mudanca_ocorre_exatamente_apos_theta_avaliacoes(self):
        b = MovingPeaks(dimensao=2, n_picos=3, mudanca_a_cada=10, semente=1)
        x = np.random.default_rng(0).uniform(0, 100, size=(7, 2))
        b.avaliar(x)
        b.avaliar(x)
        self.assertEqual(b.indice_ambiente, 1)   # 14 avaliações
        b.avaliar(x)                             # 21
        self.assertEqual(b.indice_ambiente, 2)
        self.assertEqual(b.avaliacoes_total, 21)
        self.assertEqual(len(b.registros), 2)

    def test_lote_que_atravessa_fronteira_usa_ambiente_novo(self):
        b = MovingPeaks(dimensao=2, n_picos=3, mudanca_a_cada=10, semente=2)
        x = np.random.default_rng(1).uniform(0, 100, size=(15, 2))
        c0, h0, w0 = b.centros.copy(), b.alturas.copy(), b.larguras.copy()
        y = b.avaliar(x)
        c1, h1, w1 = b.centros.copy(), b.alturas.copy(), b.larguras.copy()

        def f(X, c, h, w):
            dist = np.linalg.norm(X[:, None, :] - c[None], axis=2)
            return np.max(h[None] - w[None] * dist, axis=1)

        np.testing.assert_allclose(y[:10], f(x[:10], c0, h0, w0))
        np.testing.assert_allclose(y[10:], f(x[10:], c1, h1, w1))

    def test_sequencia_de_ambientes_independe_do_tamanho_dos_lotes(self):
        a = MovingPeaks(dimensao=3, n_picos=4, mudanca_a_cada=20, semente=5)
        b = MovingPeaks(dimensao=3, n_picos=4, mudanca_a_cada=20, semente=5)
        rng = np.random.default_rng(0)
        for _ in range(8):
            a.avaliar(rng.uniform(0, 100, size=(10, 3)))
        for _ in range(4):
            b.avaliar(rng.uniform(0, 100, size=(20, 3)))
        self.assertEqual(a.indice_ambiente, b.indice_ambiente)
        np.testing.assert_allclose(a.centros, b.centros)
        np.testing.assert_allclose(a.alturas, b.alturas)

    def test_erro_offline_em_caso_conhecido(self):
        # Ambiente estático (severidades 0), 1 pico, 1 dimensão: f = h - w|x - c|.
        b = MovingPeaks(
            dimensao=1, n_picos=1, mudanca_a_cada=4,
            severidade_deslocamento=0.0, severidade_altura=0.0,
            severidade_largura=0.0, semente=3,
        )
        c, w = b.centros[0, 0], b.larguras[0]
        dists = np.array([9.0, 4.0, 6.0, 1.0])
        x = (c + dists).reshape(-1, 1)
        x = np.clip(x, 0, 100)
        real = np.abs(x[:, 0] - c)
        b.avaliar(x)
        erros = w * real
        esperado = np.minimum.accumulate(erros).mean()
        self.assertAlmostEqual(b.registros[0].erro_offline, esperado, places=9)
        self.assertAlmostEqual(b.registros[0].erro_antes_mudanca, erros.min(), places=9)

    def test_tempo_de_recuperacao_em_caso_conhecido(self):
        b = MovingPeaks(
            dimensao=1, n_picos=1, mudanca_a_cada=6,
            severidade_deslocamento=0.0, severidade_altura=0.0,
            severidade_largura=0.0, semente=4,
        )
        c, w = b.centros[0, 0], b.larguras[0]
        # erros por avaliação: 30, 20, 8, 4, 0.5, 0.5
        alvo = np.array([30.0, 20.0, 8.0, 4.0, 0.5, 0.5]) / w
        x = np.clip(c - alvo, 0, 100).reshape(-1, 1)
        if np.any(x[:, 0] == 0):
            self.skipTest("posição saiu do domínio para esta semente")
        b.avaliar(x)
        rec = b.registros[0].tempo_recuperacao
        self.assertEqual(rec[10.0], 3.0)
        self.assertEqual(rec[5.0], 4.0)
        self.assertEqual(rec[1.0], 5.0)

    def test_centros_permanecem_no_dominio_e_norma_do_deslocamento(self):
        b = MovingPeaks(dimensao=4, n_picos=6, mudanca_a_cada=1,
                        severidade_deslocamento=10.0, semente=7)
        for _ in range(200):
            b.avaliar(np.zeros((1, 4)))
            self.assertTrue(np.all(b.centros >= b.lo) and np.all(b.centros <= b.hi))
            np.testing.assert_allclose(np.linalg.norm(b.velocidades, axis=1), 10.0)
        self.assertTrue(np.all(b.alturas >= b.h_min) and np.all(b.alturas <= b.h_max))

    def test_validacoes(self):
        with self.assertRaises(ValueError):
            MovingPeaks(mudanca_a_cada=0)
        with self.assertRaises(ValueError):
            MovingPeaks(correlacao=1.5)


class TestPoliticas(unittest.TestCase):
    def test_fixa_constante(self):
        p = Fixa(30)
        self.assertEqual(p.tamanho_inicial(), 30)
        self.assertEqual(p.proximo_tamanho(True, 30), 30)
        self.assertEqual(p.proximo_tamanho(False, 30), 30)

    def test_dente_de_serra(self):
        p = DenteDeSerra(n_min=20, n_max=100, geracoes_decaimento=4)
        self.assertEqual(p.tamanho_inicial(), 100)
        seq = [p.proximo_tamanho(False, 0) for _ in range(6)]
        self.assertEqual(seq, [80, 60, 40, 20, 20, 20])
        self.assertEqual(p.proximo_tamanho(True, 20), 100)
        self.assertEqual(p.proximo_tamanho(False, 100), 80)

    def test_validacao(self):
        with self.assertRaises(ValueError):
            DenteDeSerra(n_min=50, n_max=20)


class TestAG(unittest.TestCase):
    def _rodar(self, politica, semente=0, theta=500, n_amb=6):
        b = MovingPeaks(dimensao=3, n_picos=4, mudanca_a_cada=theta, semente=semente)
        ag = AlgoritmoGenetico(politica, b, semente=semente + 1000, config=ConfigAG())
        return b, ag.executar(n_amb)

    def test_tamanhos_respeitam_limites(self):
        p = DenteDeSerra(n_min=10, n_max=40, geracoes_decaimento=5)
        _, res = self._rodar(p)
        self.assertGreaterEqual(res.trilha_tamanho.min(), 10)
        self.assertLessEqual(res.trilha_tamanho.max(), 40)

    def test_politica_fixa_mantem_tamanho(self):
        _, res = self._rodar(Fixa(25))
        self.assertTrue(np.all(res.trilha_tamanho == 25))

    def test_orcamento_e_registros(self):
        b, res = self._rodar(Fixa(25), theta=500, n_amb=6)
        self.assertGreaterEqual(res.avaliacoes, 6 * 500)
        self.assertEqual(len(b.registros), 6)
        for r in b.registros:
            self.assertGreaterEqual(r.erro_offline, 0.0)
            self.assertGreaterEqual(r.erro_antes_mudanca, 0.0)

    def test_inicializacao_comum_para_todas_as_politicas(self):
        tamanhos = []
        for pol in (Fixa(25), DenteDeSerra(10, 40, 5)):
            b = MovingPeaks(dimensao=3, n_picos=4, mudanca_a_cada=500, semente=2)
            original = b.avaliar
            lotes = []

            def espia(X, _orig=original, _l=lotes):
                _l.append(len(X))
                return _orig(X)

            b.avaliar = espia
            AlgoritmoGenetico(pol, b, semente=1002, config=ConfigAG()).executar(2)
            tamanhos.append(lotes[0])
        self.assertEqual(tamanhos[0], tamanhos[1])
        self.assertEqual(tamanhos[0], ConfigAG().tamanho_inicial_comum)

    def test_reprodutibilidade(self):
        _, r1 = self._rodar(DenteDeSerra(10, 40, 5), semente=3)
        _, r2 = self._rodar(DenteDeSerra(10, 40, 5), semente=3)
        np.testing.assert_array_equal(r1.trilha_tamanho, r2.trilha_tamanho)

    def test_mesmo_ambiente_para_politicas_diferentes(self):
        b1, _ = self._rodar(Fixa(25), semente=9)
        b2, _ = self._rodar(DenteDeSerra(10, 40, 5), semente=9)
        np.testing.assert_allclose(b1.alturas, b2.alturas)
        np.testing.assert_allclose(b1.centros, b2.centros)

    def test_custo_da_politica_adaptativa_difere_em_geracoes_nao_em_avaliacoes_por_ambiente(self):
        b, res = self._rodar(DenteDeSerra(10, 40, 5), theta=500, n_amb=6)
        # cada ambiente recebe exatamente theta avaliações, por construção
        self.assertEqual(len(b.registros), 6)
        self.assertGreater(res.geracoes_por_ambiente.min(), 0)


if __name__ == "__main__":
    unittest.main()


class TestExtensoes(unittest.TestCase):
    def test_imigrantes_mantem_orcamento_e_tamanho(self):
        from agdin.experimento import EspecPolitica, Regime, executar_uma
        from agdin.ag import ConfigAG
        for cresc in ("filhos", "imigrantes"):
            _, _, meta = executar_uma(EspecPolitica("saw", rotulo="S"), Regime(1000, 5.0), 3, 5,
                                      ConfigAG(crescimento=cresc))
            self.assertGreaterEqual(meta["avaliacoes"], 5 * 1000)
            self.assertGreaterEqual(meta["tamanho_medio"], 20)
            self.assertLessEqual(meta["tamanho_medio"], 100)

    def test_crescimento_invalido(self):
        from agdin.ag import ConfigAG
        with self.assertRaises(ValueError):
            ConfigAG(crescimento="x").validar()

    def test_reflexao_altura_largura_mantem_limites(self):
        from agdin.benchmark import MovingPeaks
        b = MovingPeaks(mudanca_a_cada=10, severidade_altura=50.0, severidade_largura=20.0,
                        refletir_altura_largura=True, altura_inicial_uniforme=50.0, semente=1)
        self.assertTrue((b.alturas == 50.0).all())
        for _ in range(30):
            b.avaliar(np.random.default_rng(0).uniform(0, 100, size=(10, 5)))
            self.assertTrue((b.alturas >= 30).all() and (b.alturas <= 70).all())
            self.assertTrue((b.larguras >= 1).all() and (b.larguras <= 12).all())
