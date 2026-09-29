"""
Módulo de utilitários e regras de negócio para a aplicação de ML e Segurança Viária.
Separado do streamlit_app.py para permitir testes unitários e de integração.
"""

from __future__ import annotations

# Mapeamento de municípios do Litoral Catarinense (corredor costeiro BR-101)
MUNICIPIOS_LITORAL = {
    "ARARANGUA", "PALHOCA", "SAO JOSE", "FLORIANOPOLIS", "BIGUACU", "TIJUCAS",
    "CAMBORIU", "BALNEARIO CAMBORIU", "ITAJAI", "NAVEGANTES", "BARRA VELHA",
    "BALNEARIO BARRA DO SUL", "ARAQUARI", "JOINVILLE", "GARUVA", "PASSO DE TORRES",
    "SANGAO", "ICARA", "CRICIUMA", "TUBARAO", "LAGUNA", "IMBITUBA", "GAROPABA",
    "PAULO LOPES", "PENHA", "BALNEARIO PICARRAS", "PORTO BELO", "GOVERNADOR CELSO RAMOS",
    "ITAPEMA", "SAO FRANCISCO DO SUL", "JAGUARUNA", "PESCARIA BRAVA", "MARACAJA",
    "SOMBRIO", "SANTA ROSA DO SUL", "SAO JOAO DO SUL", "TIMBE DO SUL", "CAPIVARI DE BAIXO",
}

EPOCAS_MAPA = {
    12: "🌞 Verão (Dez‑Fev)", 1: "🌞 Verão (Dez‑Fev)", 2: "🌞 Verão (Dez‑Fev)",
    3: "🍂 Outono (Mar‑Mai)", 4: "🍂 Outono (Mar‑Mai)", 5: "🍂 Outono (Mar‑Mai)",
    6: "❄️ Inverno (Jun‑Ago)", 7: "❄️ Inverno (Jun‑Ago)", 8: "❄️ Inverno (Jun‑Ago)",
    9: "🌸 Primavera (Set‑Nov)", 10: "🌸 Primavera (Set‑Nov)", 11: "🌸 Primavera (Set‑Nov)",
}

CAUSA_INTERVENCAO: dict[str, tuple[str, str, str]] = {
    "Velocidade Incompatível":
        ("🚨", "Comportamento", "Radar/lombada eletrônica + campanhas de conscientização"),
    "Falta de Atenção à Condução":
        ("📵", "Comportamento", "Campanha contra celular + blitz de atenção"),
    "Ingestão de álcool pelo condutor":
        ("🍺", "Comportamento", "Operação Balada Segura + blitz noturna"),
    "Condutor Dormindo":
        ("😴", "Comportamento", "Pontos de descanso + alertas de fadiga"),
    "Animais na Pista":
        ("🐄", "Via/Ambiente", "Cercas ao longo das rodovias + placas de alerta"),
    "Pista Escorregadia":
        ("🌧️", "Via/Ambiente", "Manutenção do pavimento + drenagem"),
    "Curva acentuada":
        ("🔄", "Engenharia", "Barreiras de contenção + redutor de velocidade"),
}


def classificar_periodo(hora: int) -> str:
    """Classifica a hora do dia no período correspondente."""
    if not (0 <= hora <= 23):
        raise ValueError("Hora deve estar entre 0 e 23.")
    if hora < 6:
        return "Madrugada"
    if hora < 12:
        return "Manhã"
    if hora < 18:
        return "Tarde"
    return "Noite"


def eh_fim_de_semana(dia_semana_num: int) -> int:
    """Retorna 1 para sábado (5) e domingo (6), 0 para dias úteis."""
    if not (0 <= dia_semana_num <= 6):
        raise ValueError("Dia da semana deve estar entre 0 (Seg) e 6 (Dom).")
    return 1 if dia_semana_num >= 5 else 0


def identificar_zona(municipio: str) -> str:
    """Identifica se o município pertence ao Litoral ou Interior de SC."""
    if not municipio:
        return "🏔️ Interior"
    mun_norm = municipio.strip().upper()
    return "🏖️ Litoral" if mun_norm in MUNICIPIOS_LITORAL else "🏔️ Interior"


def calcular_taxa_fatalidade(total_acidentes: int, total_fatais: int) -> float:
    """Calcula a taxa de fatalidade (%)."""
    if total_acidentes <= 0:
        return 0.0
    return float((total_fatais / total_acidentes) * 100)


def sugerir_intervencao(causa: str) -> tuple[str, str, str]:
    """Retorna a recomendação de intervenção para uma dada causa de acidente."""
    return CAUSA_INTERVENCAO.get(
        causa,
        ("❓", "Outro", "Investigar causa com agentes locais da PRF")
    )

