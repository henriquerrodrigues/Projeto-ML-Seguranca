"""
Suíte de Testes Unitários e de Integração — MVP Segurança Viária SC
Compatível com unittest e pytest.
"""

import sys
import unittest
from pathlib import Path

# Adiciona diretórios ao path
ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(ROOT_DIR))

from utils import (
    classificar_periodo,
    eh_fim_de_semana,
    identificar_zona,
    calcular_taxa_fatalidade,
    sugerir_intervencao,
    MUNICIPIOS_LITORAL,
)

DATA_PATH = ROOT_DIR / "data_prf_sc" / "processed" / "sinistros_sc_todas_brs.csv"
MODEL_PATH = ROOT_DIR / "models" / "baseline_model.joblib"


class TestUnitariosRegrasNegocio(unittest.TestCase):
    """Testes unitários para regras de negócio e funções de domínio."""

    def test_classificar_periodo_madrugada(self):
        self.assertEqual(classificar_periodo(0), "Madrugada")
        self.assertEqual(classificar_periodo(5), "Madrugada")

    def test_classificar_periodo_manha(self):
        self.assertEqual(classificar_periodo(6), "Manhã")
        self.assertEqual(classificar_periodo(11), "Manhã")

    def test_classificar_periodo_tarde(self):
        self.assertEqual(classificar_periodo(12), "Tarde")
        self.assertEqual(classificar_periodo(17), "Tarde")

    def test_classificar_periodo_noite(self):
        self.assertEqual(classificar_periodo(18), "Noite")
        self.assertEqual(classificar_periodo(23), "Noite")

    def test_classificar_periodo_invalido(self):
        with self.assertRaises(ValueError):
            classificar_periodo(24)
        with self.assertRaises(ValueError):
            classificar_periodo(-1)

    def test_eh_fim_de_semana_dias_uteis(self):
        for dia in range(0, 5):  # Seg a Sex
            self.assertEqual(eh_fim_de_semana(dia), 0)

    def test_eh_fim_de_semana_fds(self):
        self.assertEqual(eh_fim_de_semana(5), 1)  # Sáb
        self.assertEqual(eh_fim_de_semana(6), 1)  # Dom

    def test_eh_fim_de_semana_invalido(self):
        with self.assertRaises(ValueError):
            eh_fim_de_semana(7)

    def test_identificar_zona_litoral(self):
        self.assertEqual(identificar_zona("FLORIANOPOLIS"), "🏖️ Litoral")
        self.assertEqual(identificar_zona("  balneario camboriu "), "🏖️ Litoral")
        self.assertEqual(identificar_zona("JOINVILLE"), "🏖️ Litoral")

    def test_identificar_zona_interior(self):
        self.assertEqual(identificar_zona("LAGES"), "🏔️ Interior")
        self.assertEqual(identificar_zona("CHAPECO"), "🏔️ Interior")
        self.assertEqual(identificar_zona(""), "🏔️ Interior")

    def test_calcular_taxa_fatalidade(self):
        self.assertAlmostEqual(calcular_taxa_fatalidade(100, 5), 5.0)
        self.assertAlmostEqual(calcular_taxa_fatalidade(0, 0), 0.0)

    def test_sugerir_intervencao_conhecida(self):
        emoji, cat, acao = sugerir_intervencao("Velocidade Incompatível")
        self.assertEqual(emoji, "🚨")
        self.assertEqual(cat, "Comportamento")
        self.assertIn("Radar", acao)

    def test_sugerir_intervencao_desconhecida(self):
        emoji, cat, acao = sugerir_intervencao("Causa Inexistente Teste")
        self.assertEqual(emoji, "❓")
        self.assertEqual(cat, "Outro")


class TestIntegracaoArquivosEModelo(unittest.TestCase):
    """Testes de integração para arquivos de dados e modelo de ML."""

    def test_arquivo_dados_csv_existe(self):
        self.assertTrue(DATA_PATH.exists(), f"Arquivo CSV não encontrado em {DATA_PATH}")
        self.assertGreater(DATA_PATH.stat().st_size, 1000)

    def test_modelo_joblib_existe(self):
        self.assertTrue(MODEL_PATH.exists(), f"Arquivo joblib não encontrado em {MODEL_PATH}")
        self.assertGreater(MODEL_PATH.stat().st_size, 1000)

    def test_municipios_litoral_integridade(self):
        self.assertGreater(len(MUNICIPIOS_LITORAL), 10)
        self.assertIn("FLORIANOPOLIS", MUNICIPIOS_LITORAL)
        self.assertIn("BALNEARIO CAMBORIU", MUNICIPIOS_LITORAL)

    def test_leitura_estrutura_csv_sem_pandas(self):
        import csv
        with open(DATA_PATH, mode="r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            first_row = next(reader)
            self.assertIn("br", first_row)
            self.assertIn("classificacao_acidente", first_row)
            self.assertIn("causa_acidente", first_row)


if __name__ == "__main__":
    unittest.main()

