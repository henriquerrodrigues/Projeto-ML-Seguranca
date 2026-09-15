"""Dashboard de sinistros nas rodovias federais de Santa Catarina (dados PRF)."""

import pandas as pd
import plotly.express as px
import streamlit as st

DATA_PATH = "data_prf_sc/processed/sinistros_sc_todas_brs.csv"

COLOR_SEQUENCE = ["#2a78d6", "#1b9e77", "#eb6834", "#6c5ce7", "#d03b3b", "#f2b84b"]

st.set_page_config(
    page_title="Sinistros nas Rodovias Federais de SC",
    page_icon="🚗",
    layout="wide",
)


@st.cache_data
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["data_inversa"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    df["ano"] = df["data_inversa"].dt.year
    df["mes"] = df["data_inversa"].dt.to_period("M").dt.to_timestamp()
    df["br"] = df["br"].astype(str)
    return df


df = load_data(DATA_PATH)

st.title("🚗 Sinistros nas Rodovias Federais de Santa Catarina")
st.caption("Fonte: Polícia Rodoviária Federal (PRF) — dados de 2020 a 2025.")

with st.sidebar:
    st.header("Filtros")
    anos = sorted(df["ano"].dropna().unique().astype(int))
    ano_sel = st.slider(
        "Período (ano)", min_value=min(anos), max_value=max(anos),
        value=(min(anos), max(anos)),
    )
    brs = sorted(df["br"].dropna().unique(), key=lambda x: int(x))
    br_sel = st.multiselect("BR", brs, default=brs)

df_f = df[
    df["ano"].between(ano_sel[0], ano_sel[1])
    & df["br"].isin(br_sel)
]

col1, col2, col3, col4 = st.columns(4)
col1.metric("Sinistros", f"{len(df_f):,}".replace(",", "."))
col2.metric("Mortos", f"{int(df_f['mortos'].sum()):,}".replace(",", "."))
col3.metric("Feridos", f"{int(df_f['feridos'].sum()):,}".replace(",", "."))
col4.metric("Veículos envolvidos", f"{int(df_f['veiculos'].sum()):,}".replace(",", "."))

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Sinistros por BR")
    por_br = (
        df_f.groupby("br").size().reset_index(name="sinistros")
        .sort_values("sinistros", ascending=False)
    )
    fig_br = px.bar(
        por_br, x="br", y="sinistros", color="br",
        color_discrete_sequence=COLOR_SEQUENCE,
    )
    fig_br.update_layout(showlegend=False, xaxis_title="BR", yaxis_title="Sinistros")
    st.plotly_chart(fig_br, use_container_width=True)

with right:
    st.subheader("Evolução mensal")
    por_mes = df_f.groupby("mes").size().reset_index(name="sinistros")
    fig_mes = px.line(por_mes, x="mes", y="sinistros", markers=True)
    fig_mes.update_traces(line_color=COLOR_SEQUENCE[0])
    fig_mes.update_layout(xaxis_title="Mês", yaxis_title="Sinistros")
    st.plotly_chart(fig_mes, use_container_width=True)

left2, right2 = st.columns(2)

with left2:
    st.subheader("Principais causas")
    top_causas = (
        df_f["causa_acidente"].value_counts().head(10)
        .reset_index()
    )
    top_causas.columns = ["causa_acidente", "sinistros"]
    fig_causas = px.bar(
        top_causas.sort_values("sinistros"), x="sinistros", y="causa_acidente",
        orientation="h", color_discrete_sequence=[COLOR_SEQUENCE[2]],
    )
    fig_causas.update_layout(yaxis_title="", xaxis_title="Sinistros")
    st.plotly_chart(fig_causas, use_container_width=True)

with right2:
    st.subheader("Gravidade do acidente")
    grav = df_f["classificacao_acidente"].value_counts().reset_index()
    grav.columns = ["classificacao_acidente", "sinistros"]
    fig_grav = px.pie(
        grav, names="classificacao_acidente", values="sinistros",
        color_discrete_sequence=COLOR_SEQUENCE, hole=0.4,
    )
    st.plotly_chart(fig_grav, use_container_width=True)

st.divider()

st.subheader("Localização dos sinistros")
mapa_df = df_f.dropna(subset=["latitude", "longitude"])
if len(mapa_df) > 8000:
    mapa_df = mapa_df.sample(8000, random_state=42)
st.map(mapa_df.rename(columns={"latitude": "lat", "longitude": "lon"})[["lat", "lon"]])
