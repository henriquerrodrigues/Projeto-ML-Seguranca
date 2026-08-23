"""
Script de ingestão — Pontos Críticos e Gravidade de Sinistros em SC
=====================================================================================

O que este script faz:
  1) Baixa os CSVs anuais de acidentes da PRF (Dados Abertos, "agrupados por ocorrência").
    2) Concatena os anos e filtra UF == 'SC', mantendo todas as BRs do estado.
  3) Faz merge com a planilha do SNV (DNIT), casando cada acidente ao trecho
     rodoviário correspondente pelo (BR, UF, km), trazendo atributos de engenharia
     da via (superfície, extensão do trecho, jurisdição etc.).
  4) Salva o dataset final pronto para EDA em /mnt/user-data/outputs/.

IMPORTANTE — leia antes de rodar:
  - Este ambiente sandbox só tem saída de rede liberada para domínios de pacotes
    (pypi.org, github.com etc.). Ele NÃO alcança gov.br nem drive.google.com.
    Portanto, rode este script na SUA máquina (onde a rede é livre), não aqui no chat.
  - Os IDs do Google Drive abaixo foram coletados da página oficial em
    https://www.gov.br/prf/pt-br/acesso-a-informacao/dados-abertos/dados-abertos-da-prf
    em 2026-08-04. Esses links podem mudar — confira a página se algum ID falhar.
  - O SNV (malha rodoviária do DNIT) não tem uma URL fixa de download direto: a PRF
    disponibiliza a planilha via link dinâmico (site do DNIT/VGEO). Baixe manualmente
    a planilha "SNV - Base de Dados" (formato .xls/.csv) em
    https://servicos.dnit.gov.br/vgeo/ (ou no link "Rodovias" -> SNV) e aponte o
    caminho em SNV_LOCAL_PATH abaixo. O script assume as colunas típicas do SNV:
    BR, UF, Km Inicial, Km Final, Superfície, Jurisdição (ajuste os nomes conforme
    a versão baixada — eles mudam um pouco entre atualizações).

Dependências (rode localmente):
    pip install pandas gdown

Uso:
    python ingest_prf_sc.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import pandas as pd

# ----------------------------------------------------------------------------
# CONFIGURAÇÃO
# ----------------------------------------------------------------------------

# Pasta de trabalho local
DATA_DIR = Path("./data_prf_sc")
RAW_DIR = DATA_DIR / "raw"
OUT_DIR = DATA_DIR / "processed"

# Anos a baixar (agrupados por ocorrência). Ajuste conforme necessário —
# quanto mais anos, mais volume para o treino, mas atenção a mudanças de
# metodologia de coleta da PRF ao longo do tempo.
GDRIVE_FILE_IDS: dict[int, str] = {
    2026: "1A3IirNm0AzRaSosA1IS94DOVmvKsn0Ol",
    2025: "1-G3MdmHBt6CprDwcW99xxC4BZ2DU5ryR",
    2024: "14lB0vqMFkaZj8HZ44b0njYgxs9nAN8KO",
    2023: "1-WO3SfNrwwZ5_l7fRTiwBKRw7mi1-HUq",
    2022: "1PRQjuV5gOn_nn6UNvaJyVURDIfbSAK4-",
    2021: "12xH8LX9aN2gObR766YN3cMcuycwyCJDz",
    2020: "1esu6IiH5TVTxFoedv6DBGDd01Gvi8785",
    2019: "1pN3fn2wY34GH6cY-gKfbxRJJBFE0lb_l",
    2018: "1cM4IgGMIiR-u4gBIH5IEe3DcvBvUzedi",
    2017: "1HPLWt5f_l4RIX3tKjI4tUXyZOev52W0N",
}

UF_ALVO = "SC"
# Aponte aqui para a planilha do SNV baixada manualmente do DNIT/VGEO
SNV_LOCAL_PATH = DATA_DIR / "snv_base.csv"  # .xls também funciona, ver load_snv()

FINAL_OUTPUT_PATH = OUT_DIR / "sinistros_sc_todas_brs.csv"
BR_SUMMARY_PATH = OUT_DIR / "brs_sc_resumo.csv"


# ----------------------------------------------------------------------------
# 1) DOWNLOAD DOS CSVS DA PRF
# ----------------------------------------------------------------------------

def download_prf_year(year: int, file_id: str, dest_dir: Path) -> Path:
    """Baixa o CSV de um ano via gdown (lida com a tela de confirmação do Drive)."""
    import gdown

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / f"acidentes{year}.csv"
    if dest.exists():
        print(f"[{year}] já baixado, pulando.")
        return dest

    url = f"https://drive.google.com/uc?id={file_id}"
    print(f"[{year}] baixando...")
    gdown.download(url, str(dest), quiet=False)
    return dest


def read_prf_csv(path: Path) -> pd.DataFrame:
    """
    Lê um CSV da PRF. O arquivo baixado do Drive costuma ser, na verdade, um
    .zip contendo o CSV (ex.: datatran2023.csv) — detectamos isso pela
    assinatura PK e extraímos o CSV interno antes de parsear.
    Historicamente o CSV vem em latin-1, separador ';', decimal ','.
    Tenta algumas combinações comuns e cai para detecção automática se preciso.
    """
    import io
    import zipfile

    with open(path, "rb") as f:
        head = f.read(4)

    if head[:2] == b"PK":
        with zipfile.ZipFile(path) as zf:
            csv_names = [n for n in zf.namelist() if n.lower().endswith(".csv")]
            if not csv_names:
                raise RuntimeError(f"{path} é um zip sem nenhum .csv dentro.")
            raw_bytes = zf.read(csv_names[0])
        source = lambda: io.BytesIO(raw_bytes)  # noqa: E731
    else:
        source = lambda: path  # noqa: E731

    attempts = [
        dict(sep=";", encoding="latin-1", decimal=","),
        dict(sep=";", encoding="utf-8", decimal=","),
        dict(sep=",", encoding="latin-1", decimal="."),
    ]
    last_err = None
    for kwargs in attempts:
        try:
            df = pd.read_csv(source(), low_memory=False, **kwargs)
            if df.shape[1] > 3:  # sanity check: parseou colunas de verdade
                return df
        except Exception as e:  # noqa: BLE001
            last_err = e
    raise RuntimeError(f"Não consegui ler {path} com os parsers conhecidos: {last_err}")


def normalize_prf_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza nomes de colunas (a PRF varia capitalização/acentos entre anos)."""
    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.normalize("NFKD")
        .str.encode("ascii", errors="ignore")
        .str.decode("utf-8")
        .str.replace(r"\s+", "_", regex=True)
    )
    return df


def _to_nullable_int(series: pd.Series) -> pd.Series:
    """Converte identificadores para inteiro com suporte a nulos."""
    normalized = (
        series.astype(str)
        .str.strip()
        .replace({"": pd.NA, "nan": pd.NA, "None": pd.NA})
        .str.replace(r"\.0+$", "", regex=True)
        .str.extract(r"(-?\d+)", expand=False)
    )
    return pd.to_numeric(normalized, errors="coerce").astype("Int64")


def _clean_decimal_text(value: object) -> str:
    text = str(value).strip()
    text = re.sub(r"[^0-9,\.\-+]", "", text)
    if "," in text and "." in text:
        # Usa como decimal o último separador; os demais viram milhares.
        comma_pos = text.rfind(",")
        dot_pos = text.rfind(".")
        decimal_sep = "," if comma_pos > dot_pos else "."
        thousands_sep = "." if decimal_sep == "," else ","
        text = text.replace(thousands_sep, "")
        text = text.replace(decimal_sep, ".")
    elif "," in text:
        text = text.replace(",", ".")
    return text


def _coerce_coord_value(value: object, limit: float) -> float | None:
    if pd.isna(value):
        return None

    text = _clean_decimal_text(value)
    if text in {"", "+", "-", ".", "+.", "-."}:
        return None

    parsed = pd.to_numeric(text, errors="coerce")
    if pd.isna(parsed):
        return None

    number = float(parsed)
    if abs(number) <= limit:
        return number

    digits = re.sub(r"\D", "", text)
    if not digits:
        return None
    sign = -1 if text.startswith("-") else 1
    scaled = float(int(digits)) * sign
    while abs(scaled) > limit and scaled != 0:
        scaled /= 10.0
    return scaled if abs(scaled) <= limit else None


def normalize_prf_values(df: pd.DataFrame) -> pd.DataFrame:
    """Padroniza tipos sensíveis (id, km, br, latitude, longitude)."""
    df = df.copy()

    for id_col in ("id", "id_acidente"):
        if id_col in df.columns:
            df[id_col] = _to_nullable_int(df[id_col])

    if "km" in df.columns:
        df["km"] = pd.to_numeric(
            df["km"].astype(str).str.replace(",", ".", regex=False),
            errors="coerce",
        )

    if "br" in df.columns:
        df["br"] = pd.to_numeric(
            df["br"].astype(str).str.extract(r"(\d+)", expand=False),
            errors="coerce",
        ).astype("Int64")

    if "latitude" in df.columns:
        df["latitude"] = df["latitude"].map(lambda value: _coerce_coord_value(value, 90.0))
    if "longitude" in df.columns:
        df["longitude"] = df["longitude"].map(lambda value: _coerce_coord_value(value, 180.0))

    if "br" in df.columns:
        df["fora_br"] = df["br"].fillna(-1).eq(0)
        df["relevancia_rodovia"] = df["fora_br"].map(
            {True: "baixa (fora de BR)", False: "alta (em BR)"}
        )

    return df


# ----------------------------------------------------------------------------
# 2) FILTRO SC / TODAS AS BRs
# ----------------------------------------------------------------------------

def filter_sc_brs(df: pd.DataFrame) -> pd.DataFrame:
    df = normalize_prf_values(df)

    mask = pd.Series(True, index=df.index)
    if "uf" in df.columns:
        mask &= df["uf"].astype(str).str.upper().str.strip() == UF_ALVO
    return df[mask].reset_index(drop=True)


# ----------------------------------------------------------------------------
# 3) MERGE COM O SNV (DNIT) — atributos de engenharia do trecho
# ----------------------------------------------------------------------------

def load_snv(path: Path) -> pd.DataFrame:
    """
    Carrega a planilha do SNV. Ajuste os nomes de coluna abaixo conforme a
    versão baixada (eles variam um pouco entre releases do DNIT).
    Colunas esperadas após normalização: br, uf, km_inicial, km_final,
    superficie, jurisdicao (algumas dessas podem não existir — trate como opcional).
    """
    if not path.exists():
        print(
            f"[aviso] {path} não encontrado — pulando o merge com o SNV. "
            "Baixe a planilha do DNIT/VGEO e aponte SNV_LOCAL_PATH para rodar essa etapa."
        )
        return pd.DataFrame()

    if path.suffix.lower() in {".xls", ".xlsx"}:
        snv = pd.read_excel(path)
    else:
        snv = pd.read_csv(path, sep=None, engine="python")

    snv = normalize_prf_columns(snv)  # mesma limpeza de nomes serve aqui
    return snv


def merge_with_snv(acidentes: pd.DataFrame, snv: pd.DataFrame) -> pd.DataFrame:
    """
    Casa cada acidente ao trecho do SNV cujo intervalo [km_inicial, km_final]
    contém o km do acidente, para o mesmo BR/UF.
    """
    if snv.empty or "km" not in acidentes.columns:
        return acidentes

    required = {"br", "uf", "km_inicial", "km_final"}
    missing = required - set(snv.columns)
    if missing:
        print(f"[aviso] SNV sem as colunas {missing} — confira os nomes reais e ajuste load_snv(). Pulando merge.")
        return acidentes

    snv = snv.copy()
    snv["br"] = pd.to_numeric(snv["br"], errors="coerce")
    snv["km_inicial"] = pd.to_numeric(snv["km_inicial"], errors="coerce")
    snv["km_final"] = pd.to_numeric(snv["km_final"], errors="coerce")

    merged_rows = []
    for (uf, br), grp_acid in acidentes.groupby(["uf", "br"]):
        grp_snv = snv[(snv["uf"].astype(str).str.upper() == str(uf).upper()) & (snv["br"] == br)]
        if grp_snv.empty:
            merged_rows.append(grp_acid)
            continue

        grp_snv = grp_snv.sort_values("km_inicial")
        acids = grp_acid.copy()
        acids["_snv_idx"] = acids["km"].apply(
            lambda k: _find_segment(k, grp_snv) if pd.notnull(k) else None
        )
        acids = acids.merge(
            grp_snv.drop(columns=["br", "uf"]).add_prefix("snv_"),
            left_on="_snv_idx",
            right_index=True,
            how="left",
        ).drop(columns=["_snv_idx"])
        merged_rows.append(acids)

    return pd.concat(merged_rows, ignore_index=True) if merged_rows else acidentes


def _find_segment(km: float, snv_group: pd.DataFrame):
    match = snv_group[(snv_group["km_inicial"] <= km) & (km <= snv_group["km_final"])]
    return match.index[0] if not match.empty else None


# ----------------------------------------------------------------------------
# PIPELINE PRINCIPAL
# ----------------------------------------------------------------------------

def main() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    FINAL_OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    frames = []
    for year, file_id in sorted(GDRIVE_FILE_IDS.items()):
        try:
            path = download_prf_year(year, file_id, RAW_DIR)
            df = read_prf_csv(path)
            df = normalize_prf_columns(df)
            df["ano_arquivo"] = year
            df_sc = filter_sc_brs(df)
            print(f"[{year}] {len(df)} registros nacionais -> {len(df_sc)} em SC/todas as BRs")
            frames.append(df_sc)
        except Exception as e:  # noqa: BLE001
            print(f"[{year}] ERRO: {e}", file=sys.stderr)

    if not frames:
        print("Nenhum ano foi processado com sucesso. Abortando.")
        return

    acidentes = pd.concat(frames, ignore_index=True)
    interim_path = OUT_DIR / "acidentes_sc_todas_brs_bruto.csv"
    # Usa decimal "," para evitar que Excel/LibreOffice em pt-BR leia latitude/longitude
    # como texto ou aplique separação errada de milhares.
    acidentes.to_csv(interim_path, index=False, decimal=",")
    print(f"\nBase filtrada (sem merge SNV) salva em: {interim_path} ({len(acidentes)} linhas)")

    br_summary = (
        acidentes.groupby("br", dropna=False)
        .size()
        .rename("quantidade_acidentes")
        .reset_index()
        .sort_values("br", na_position="last")
    )
    br_summary.to_csv(BR_SUMMARY_PATH, index=False)
    print(f"Resumo das BRs em SC salvo em: {BR_SUMMARY_PATH}")

    snv = load_snv(SNV_LOCAL_PATH)
    final = merge_with_snv(acidentes, snv)

    final.to_csv(FINAL_OUTPUT_PATH, index=False, decimal=",")
    print(f"\nDataset final pronto para EDA: {FINAL_OUTPUT_PATH} ({len(final)} linhas, {final.shape[1]} colunas)")

    # resumo rápido, útil como primeiro check da EDA
    print("\nResumo rápido:")
    print(final.dtypes.value_counts())
    print("\nValores nulos por coluna (top 10):")
    print(final.isnull().sum().sort_values(ascending=False).head(10))


if __name__ == "__main__":
    main()
