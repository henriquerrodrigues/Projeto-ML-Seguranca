"""
Dashboard + ML · Sinistros nas Rodovias Federais de SC — PRF
=======================================================================
"""

from __future__ import annotations
import sys
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from PIL import Image

# Importação de funções de domínio para modularização e testabilidade
ROOT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT_DIR / "src"))

from utils import (
    classificar_periodo,
    eh_fim_de_semana,
    identificar_zona,
    sugerir_intervencao,
    EPOCAS_MAPA,
    MUNICIPIOS_LITORAL,
    CAUSA_INTERVENCAO,
)

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────────────────────────
# CONSTANTES
# ──────────────────────────────────────────────────────────────────────────────

DATA_PATH   = ROOT_DIR / "data_prf_sc/processed/sinistros_sc_todas_brs.csv"
MODEL_PATH  = ROOT_DIR / "models/baseline_model.joblib"
FAVICON     = ROOT_DIR / "icons/cc-logo-icon-bg-transparent@512w.png"
LOGO        = str(ROOT_DIR / "icons/cc-logo-bg-transparent.png")

CORES = ["#2a78d6","#1b9e77","#eb6834","#6c5ce7","#d03b3b","#f2b84b","#00b4d8","#e76f51"]
COR_GRAVIDADE = {
    "Sem Vítimas":        "#1b9e77",
    "Com Vítimas Feridas": "#f2b84b",
    "Com Vítimas Fatais":  "#d03b3b",
}
FONT = "Roboto, sans-serif"

MESES_NOMES = {1:"Jan",2:"Fev",3:"Mar",4:"Abr",5:"Mai",6:"Jun",
               7:"Jul",8:"Ago",9:"Set",10:"Out",11:"Nov",12:"Dez"}

DIAS_NOMES = {0:"Seg",1:"Ter",2:"Qua",3:"Qui",4:"Sex",5:"Sáb",6:"Dom"}

EPOCAS_ORDEM = ["🌞 Verão (Dez‑Fev)","🍂 Outono (Mar‑Mai)",
                "❄️ Inverno (Jun‑Ago)","🌸 Primavera (Set‑Nov)"]

# ──────────────────────────────────────────────────────────────────────────────
# LAYOUT BASE
# ──────────────────────────────────────────────────────────────────────────────

st.set_page_config(
    page_title="Sinistros Rodovias Federais SC",
    page_icon=Image.open(FAVICON),
    layout="wide",
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Roboto:wght@400;500;700&display=swap');
html,body,[class*="css"]{font-family:'Roboto',sans-serif;}

.kpi-card{
    background:rgba(42,120,214,.07);border-left:4px solid #2a78d6;
    border-radius:6px;padding:.75rem 1rem;margin:.4rem 0;font-size:.9rem;
}
.kpi-red{border-color:#d03b3b;background:rgba(208,59,59,.07);}
.kpi-green{border-color:#1b9e77;background:rgba(27,158,119,.07);}
.kpi-orange{border-color:#eb6834;background:rgba(235,104,52,.07);}

.tag{display:inline-block;padding:2px 10px;border-radius:12px;font-size:.8rem;font-weight:600;margin:2px;}
.tag-red{background:#d03b3b22;color:#d03b3b;}
.tag-orange{background:#eb683422;color:#eb6834;}
.tag-blue{background:#2a78d622;color:#2a78d6;}

.app-footer{
    margin-top:3rem;padding-top:1rem;
    border-top:1px solid rgba(128,128,128,.3);
    text-align:center;font-size:.82rem;color:rgba(128,128,128,.85);
}

@keyframes fadeUp{from{opacity:0;transform:translateY(16px)}to{opacity:1;transform:translateY(0)}}
[data-testid="stPlotlyChart"],[data-testid="stMetric"]{animation:fadeUp .5s ease-out both;}
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────────────────────────────────────
# CARREGAMENTO DE DADOS
# ──────────────────────────────────────────────────────────────────────────────

@st.cache_data(show_spinner="⏳ Carregando dados PRF…")
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path, low_memory=False)

    df["data_inversa"] = pd.to_datetime(df["data_inversa"], errors="coerce")
    df["ano"]     = df["data_inversa"].dt.year.astype("Int64")
    df["mes_num"] = df["data_inversa"].dt.month.astype("Int64")
    df["mes_nome"]= df["mes_num"].map(MESES_NOMES)
    df["dia_semana_num"] = df["data_inversa"].dt.dayofweek.astype("Int64")
    df["dia_semana_nome"]= df["dia_semana_num"].map(DIAS_NOMES)

    hora_raw = df["horario"].astype(str).str.extract(r"^(\d{1,2})")[0]
    df["hora"] = pd.to_numeric(hora_raw, errors="coerce").fillna(12).astype(int)

    def _periodo(h):
        return f"🌙 {classificar_periodo(h)}" if h<6 else (f"🌅 {classificar_periodo(h)}" if h<12 else (f"☀️ {classificar_periodo(h)}" if h<18 else f"🌆 {classificar_periodo(h)}"))
    df["periodo_do_dia"] = df["hora"].apply(_periodo)
    df["periodo_simples"] = df["hora"].apply(classificar_periodo)

    df["br"] = df["br"].astype(str).str.strip()

    FERIADOS = {(1,1),(4,21),(5,1),(9,7),(10,12),(11,2),(11,15),(11,20),(12,25)}
    df["eh_feriado"] = df["data_inversa"].apply(
        lambda d: 1 if pd.notnull(d) and (d.month,d.day) in FERIADOS else 0
    )
    df["eh_fim_de_semana"] = df["dia_semana_num"].apply(lambda d: eh_fim_de_semana(int(d)) if pd.notnull(d) else 0)

    def _tipo_dia(row):
        if row["eh_feriado"]: return "🎉 Feriado"
        if row["eh_fim_de_semana"]: return "🏖️ Fim de semana"
        return "💼 Dia útil"
    df["tipo_dia"] = df.apply(_tipo_dia, axis=1)

    df["epoca"] = df["mes_num"].map(EPOCAS_MAPA)
    df["zona"] = df["municipio"].apply(identificar_zona)
    df["is_litoral"] = df["zona"] == "🏖️ Litoral"

    MAPA_TARGET = {"Sem Vítimas":0,"Com Vítimas Feridas":1,"Com Vítimas Fatais":2}
    df["target"] = df["classificacao_acidente"].map(MAPA_TARGET)

    df["dia_da_semana"] = df["dia_semana_num"]
    df["mes"] = df["mes_num"]

    for c in ("latitude","longitude"):
        df[c] = pd.to_numeric(
            df[c].astype(str).str.replace(",","."), errors="coerce"
        )

    return df


@st.cache_resource(show_spinner="🤖 Carregando modelo ML…")
def load_model(path: str):
    return joblib.load(path)


df_full = load_data(DATA_PATH)

try:
    artefato = load_model(MODEL_PATH)
    pipeline_ml     = artefato["pipeline"]
    features_cat_ml = artefato["features_categoricas"]
    features_num_ml = artefato["features_numericas"]
    classes_ml      = artefato["classes_target"]
    model_name      = artefato["model_name"]
    model_score     = artefato["best_score"]
    modelo_ok       = True
except Exception as e_model:
    modelo_ok  = False
    e_model_str = str(e_model)


# ──────────────────────────────────────────────────────────────────────────────
# SIDEBAR — FILTROS GLOBAIS
# ──────────────────────────────────────────────────────────────────────────────

with st.sidebar:
    try:
        st.image(LOGO, width=170)
    except Exception:
        pass

    st.markdown("## 🔎 Filtros Globais")
    st.caption("Todos os painéis refletem os filtros abaixo.")

    with st.expander("📅 Período", expanded=True):
        anos_disp = sorted(df_full["ano"].dropna().unique().astype(int))
        ano_sel = st.slider(
            "Anos", min_value=min(anos_disp), max_value=max(anos_disp),
            value=(min(anos_disp), max(anos_disp))
        )

        meses_disp = list(MESES_NOMES.values())
        meses_sel = st.multiselect(
            "Mês(es)", meses_disp, default=meses_disp,
            help="Filtre por mês do ano"
        )

        epocas_disp = EPOCAS_ORDEM
        epocas_sel = st.multiselect(
            "Época do ano", epocas_disp, default=epocas_disp
        )

    with st.expander("📆 Tipo de Dia", expanded=False):
        tipos_dia_disp = ["💼 Dia útil","🏖️ Fim de semana","🎉 Feriado"]
        tipos_dia_sel = st.multiselect(
            "Tipo de dia", tipos_dia_disp, default=tipos_dia_disp
        )

        dias_sem_disp = list(DIAS_NOMES.values())
        dias_sem_sel = st.multiselect(
            "Dias da semana", dias_sem_disp, default=dias_sem_disp
        )

        periodos_disp = ["🌙 Madrugada","🌅 Manhã","☀️ Tarde","🌆 Noite"]
        periodos_sel = st.multiselect(
            "Período do dia", periodos_disp, default=periodos_disp
        )

    with st.expander("🛣️ Via e Condições", expanded=False):
        brs_disp = sorted(df_full["br"].dropna().unique(), key=int)
        br_sel = st.multiselect("BR(s)", brs_disp, default=brs_disp)

        pistas_disp = sorted(df_full["tipo_pista"].dropna().unique())
        pistas_sel = st.multiselect("Tipo de Pista", pistas_disp, default=pistas_disp)

        climas_disp = sorted(df_full["condicao_metereologica"].dropna().unique())
        climas_sel = st.multiselect("Condição Climática", climas_disp, default=climas_disp)

        zona_sel = st.multiselect(
            "Zona", ["🏖️ Litoral","🏔️ Interior"],
            default=["🏖️ Litoral","🏔️ Interior"]
        )

    with st.expander("⚠️ Gravidade", expanded=False):
        grav_disp = ["Sem Vítimas","Com Vítimas Feridas","Com Vítimas Fatais"]
        grav_sel = st.multiselect("Classificação", grav_disp, default=grav_disp)

    st.divider()
    st.caption("Fonte: PRF · SC ")
    if st.button("🔄 Limpar filtros"):
        st.rerun()


# ──────────────────────────────────────────────────────────────────────────────
# APLICAR FILTROS
# ──────────────────────────────────────────────────────────────────────────────

meses_num_sel = [k for k,v in MESES_NOMES.items() if v in meses_sel]

df = df_full[
    df_full["ano"].between(ano_sel[0], ano_sel[1])
    & df_full["mes_num"].isin(meses_num_sel)
    & df_full["epoca"].isin(epocas_sel)
    & df_full["tipo_dia"].isin(tipos_dia_sel)
    & df_full["dia_semana_nome"].isin(dias_sem_sel)
    & df_full["periodo_do_dia"].isin(periodos_sel)
    & df_full["br"].isin(br_sel)
    & df_full["tipo_pista"].isin(pistas_sel)
    & df_full["condicao_metereologica"].isin(climas_sel)
    & df_full["zona"].isin(zona_sel)
    & df_full["classificacao_acidente"].isin(grav_sel)
].copy()


# ──────────────────────────────────────────────────────────────────────────────
# CABEÇALHO E ABAS
# ──────────────────────────────────────────────────────────────────────────────

st.title("🚦 Sinistros nas Rodovias Federais de Santa Catarina")
st.caption(
    f"Exibindo **{len(df):,}** registros com os filtros atuais "
    f"(total histórico: {len(df_full):,}) · PRF · SC "
    .replace(",",".")
)

if df.empty:
    st.warning("⚠️ Nenhum dado encontrado com os filtros selecionados. Ajuste os filtros na barra lateral.")
    st.stop()

tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Visão Geral",
    "🌊 Sazonal & Litoral",
    "🔎 Diagnóstico por BR",
    "🤖 Previsão de Risco",
    "🗺️ Mapa de Risco",
])


# ══════════════════════════════════════════════════════════════════════════════
# ABA 1 · VISÃO GERAL
# ══════════════════════════════════════════════════════════════════════════════
with tab1:
    total     = len(df)
    mortos    = int(df["mortos"].sum())
    feridos   = int(df["feridos"].sum())
    veiculos  = int(df["veiculos"].sum())
    n_fatais  = int((df["target"]==2).sum())
    taxa_fat  = n_fatais/total*100 if total else 0

    c1,c2,c3,c4,c5 = st.columns(5)
    c1.metric("🚗 Sinistros",   f"{total:,}".replace(",","."))
    c2.metric("💀 Mortos",      f"{mortos:,}".replace(",","."))
    c3.metric("🤕 Feridos",     f"{feridos:,}".replace(",","."))
    c4.metric("🚙 Veículos",    f"{veiculos:,}".replace(",","."))
    c5.metric("☠️ Taxa Fatal",  f"{taxa_fat:.2f}%")

    st.markdown(
        f'<div class="kpi-card kpi-red">⚠️ <b>{n_fatais:,} acidentes com vítimas fatais</b> · '
        f'taxa de fatalidade de <b>{taxa_fat:.2f}%</b> sobre os sinistros filtrados.</div>'
        .replace(",","."),
        unsafe_allow_html=True
    )

    st.divider()

    l1,r1 = st.columns(2)

    with l1:
        st.subheader("Sinistros por BR")
        por_br = (
            df.groupby("br").size().reset_index(name="n")
            .assign(br_label=lambda d: "BR-" + d["br"])
            .sort_values("n", ascending=False)
        )
        fig = px.bar(por_br, x="br_label", y="n", color="br_label",
                     color_discrete_sequence=CORES, text="n")
        fig.update_traces(textposition="outside")
        fig.update_layout(
            showlegend=False, xaxis_title="BR", yaxis_title="Sinistros",
            xaxis=dict(type="category", categoryorder="total descending"),
            font_family=FONT, margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    with r1:
        st.subheader("Gravidade por BR (%)")
        grav_br = (
            df.dropna(subset=["classificacao_acidente", "br"])
            .groupby(["br", "classificacao_acidente"]).size().reset_index(name="n")
        )
        total_br = grav_br.groupby("br")["n"].transform("sum")
        grav_br["pct"] = grav_br["n"] / total_br * 100
        grav_br["br_label"] = "BR-" + grav_br["br"]
        fig = px.bar(
            grav_br, x="br_label", y="pct", color="classificacao_acidente",
            barmode="stack", color_discrete_map=COR_GRAVIDADE,
            text=grav_br["pct"].map(lambda v: f"{v:.0f}%"),
        )
        fig.update_traces(textposition="inside")
        fig.update_layout(
            xaxis_title="BR", yaxis_title="%", legend_title="Gravidade",
            xaxis=dict(type="category", categoryorder="total descending"),
            font_family=FONT, margin=dict(t=10),
        )
        st.plotly_chart(fig, use_container_width=True)

    l2,r2 = st.columns(2)

    with l2:
        st.subheader("Evolução de Mortes por BR (ano)")
        ev = df.groupby(["ano","br"])["mortos"].sum().reset_index()
        fig = px.line(ev, x="ano", y="mortos", color="br",
                      markers=True, color_discrete_sequence=CORES)
        fig.update_layout(xaxis_title="Ano",yaxis_title="Mortos",
                          legend_title="BR",font_family=FONT,margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with r2:
        st.subheader("Taxa de Fatalidade por Período do Dia")
        tp = (
            df.dropna(subset=["target","periodo_do_dia"])
            .groupby("periodo_do_dia")["target"]
            .apply(lambda s: (s==2).mean()*100).reset_index()
        )
        tp.columns = ["periodo","taxa"]
        ordem = ["🌙 Madrugada","🌅 Manhã","☀️ Tarde","🌆 Noite"]
        tp["periodo"] = pd.Categorical(tp["periodo"],categories=ordem,ordered=True)
        tp = tp.sort_values("periodo")
        fig = px.bar(tp, x="periodo", y="taxa", color="periodo",
                     color_discrete_sequence=["#6c5ce7","#2a78d6","#1b9e77","#d03b3b"],
                     text=tp["taxa"].map(lambda v: f"{v:.2f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,xaxis_title="Período",
                          yaxis_title="% Fatais",font_family=FONT,margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    l3,r3 = st.columns(2)

    with l3:
        st.subheader("Top 12 Causas de Acidentes")
        causas = df["causa_acidente"].value_counts().head(12).reset_index()
        causas.columns = ["causa","n"]
        fig = px.bar(causas.sort_values("n"), x="n", y="causa",
                     orientation="h", color_discrete_sequence=[CORES[2]],
                     text="n")
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="",xaxis_title="Sinistros",
                          font_family=FONT,margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with r3:
        st.subheader("Taxa de Fatalidade por Tipo de Acidente")
        taxa_tipo = (
            df.dropna(subset=["target","tipo_acidente"])
            .groupby("tipo_acidente")["target"]
            .agg(taxa=lambda s:(s==2).mean()*100, total="count")
            .reset_index().sort_values("taxa",ascending=True)
        )
        fig = px.bar(taxa_tipo, x="taxa", y="tipo_acidente",
                     orientation="h",
                     color="taxa",
                     color_continuous_scale=["#1b9e77","#f2b84b","#d03b3b"],
                     text=taxa_tipo["taxa"].map(lambda v: f"{v:.1f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(yaxis_title="",xaxis_title="% Acidentes Fatais",
                          coloraxis_showscale=False,font_family=FONT,margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    l4,r4 = st.columns(2)

    with l4:
        st.subheader("Sinistros por Tipo de Dia")
        td = df.groupby("tipo_dia").agg(
            n=("tipo_dia","count"),
            mortos=("mortos","sum")
        ).reset_index()
        fig = px.bar(td, x="tipo_dia", y="n", color="tipo_dia",
                     color_discrete_sequence=CORES, text="n")
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,xaxis_title="",yaxis_title="Sinistros",
                          font_family=FONT,margin=dict(t=10))
        st.plotly_chart(fig, use_container_width=True)

    with r4:
        st.subheader("Sinistros e Mortos por Dia da Semana")
        dsem = df.groupby("dia_semana_nome").agg(
            n=("dia_semana_nome","count"),
            mortos=("mortos","sum")
        ).reset_index()
        ordem_dias = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
        dsem["dia_semana_nome"] = pd.Categorical(dsem["dia_semana_nome"],categories=ordem_dias,ordered=True)
        dsem = dsem.sort_values("dia_semana_nome")
        fig = go.Figure()
        fig.add_trace(go.Bar(x=dsem["dia_semana_nome"],y=dsem["n"],
                             name="Sinistros",marker_color=CORES[0]))
        fig.add_trace(go.Scatter(x=dsem["dia_semana_nome"],y=dsem["mortos"],
                                 name="Mortos",mode="lines+markers",
                                 marker_color="#d03b3b",yaxis="y2"))
        fig.update_layout(
            yaxis=dict(title="Sinistros"),
            yaxis2=dict(title=dict(text="Mortos", font=dict(color="#d03b3b")),
                        overlaying="y", side="right"),
            legend=dict(x=.01,y=.99),font_family=FONT,margin=dict(t=10)
        )
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 2 · SAZONAL & LITORAL
# ══════════════════════════════════════════════════════════════════════════════
with tab2:
    st.header("🌊 Sazonalidade e Efeito Turístico no Litoral de SC")
    st.markdown("""
    O litoral catarinense recebe **milhões de turistas** entre dezembro e fevereiro,
    sobrecarregando a **BR‑101** e rodovias de acesso às praias.
    Aqui investigamos se esse fluxo piora a gravidade dos acidentes e onde
    a fiscalização faz mais diferença.
    """)

    st.subheader("📅 Volume e Gravidade por Época do Ano")
    ep = (
        df.dropna(subset=["epoca","classificacao_acidente"])
        .groupby(["epoca","classificacao_acidente"]).size().reset_index(name="n")
    )
    ep["epoca"] = pd.Categorical(ep["epoca"],categories=EPOCAS_ORDEM,ordered=True)
    ep = ep.sort_values("epoca")
    fig = px.bar(ep, x="epoca", y="n", color="classificacao_acidente",
                 barmode="group", color_discrete_map=COR_GRAVIDADE, text="n")
    fig.update_traces(textposition="outside")
    fig.update_layout(xaxis_title="Época",yaxis_title="Sinistros",
                      legend_title="Gravidade",font_family=FONT)
    st.plotly_chart(fig, use_container_width=True)

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("☠️ Taxa de Fatalidade por Época")
        tf_ep = (
            df.dropna(subset=["target","epoca"])
            .groupby("epoca")["target"]
            .apply(lambda s:(s==2).mean()*100).reset_index()
        )
        tf_ep.columns = ["epoca","taxa"]
        tf_ep["epoca"] = pd.Categorical(tf_ep["epoca"],categories=EPOCAS_ORDEM,ordered=True)
        tf_ep = tf_ep.sort_values("epoca")
        fig = px.bar(tf_ep, x="epoca", y="taxa", color="epoca",
                     color_discrete_sequence=CORES,
                     text=tf_ep["taxa"].map(lambda v: f"{v:.2f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,xaxis_title="",
                          yaxis_title="% Fatais",font_family=FONT)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.subheader("🏖️ Litoral vs Interior por Época")
        lz = (
            df.dropna(subset=["target","epoca","zona"])
            .groupby(["zona","epoca"])["target"]
            .apply(lambda s:(s==2).mean()*100).reset_index()
        )
        lz.columns = ["zona","epoca","taxa"]
        lz["epoca"] = pd.Categorical(lz["epoca"],categories=EPOCAS_ORDEM,ordered=True)
        lz = lz.sort_values("epoca")
        fig = px.bar(lz, x="epoca", y="taxa", color="zona",
                     barmode="group",
                     color_discrete_map={"🏖️ Litoral":"#2a78d6","🏔️ Interior":"#1b9e77"},
                     text=lz["taxa"].map(lambda v: f"{v:.2f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(xaxis_title="",yaxis_title="% Fatais",
                          legend_title="Zona",font_family=FONT)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("📅 Sazonalidade Mensal — BR‑101 no Litoral (todos os anos)")
    df101 = df[(df["br"]=="101") & (df["is_litoral"])].dropna(subset=["mes_num"])
    mensal = df101.groupby("mes_num").agg(
        sinistros=("br","count"), mortos=("mortos","sum")
    ).reset_index()
    mensal["mes_nome"] = mensal["mes_num"].map(MESES_NOMES)
    mensal["mes_nome"] = pd.Categorical(
        mensal["mes_nome"], categories=list(MESES_NOMES.values()), ordered=True
    )
    mensal = mensal.sort_values("mes_num")

    fig = go.Figure()
    fig.add_trace(go.Bar(x=mensal["mes_nome"],y=mensal["sinistros"],
                         name="Sinistros",marker_color="#2a78d6",yaxis="y1"))
    fig.add_trace(go.Scatter(x=mensal["mes_nome"],y=mensal["mortos"],
                              name="Mortos",mode="lines+markers",
                              marker=dict(color="#d03b3b",size=8),
                              line=dict(color="#d03b3b",width=2),yaxis="y2"))
    fig.add_vrect(x0="Nov", x1="Mar",fillcolor="rgba(235,104,52,.10)",
                  line_width=0,annotation_text="Alta temporada",
                  annotation_position="top left")
    fig.update_layout(
        yaxis=dict(title=dict(text="Sinistros", font=dict(color="#2a78d6"))),
        yaxis2=dict(title=dict(text="Mortos", font=dict(color="#d03b3b")),
                    overlaying="y", side="right"),
        legend=dict(x=.01,y=.99),font_family=FONT
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("⚠️ Causas Predominantes: Verão vs Resto do Ano")
    df_causa = df.dropna(subset=["causa_acidente","mes_num"]).copy()
    df_causa["periodo_turismo"] = df_causa["mes_num"].apply(
        lambda m: "🌞 Verão (Dez‑Fev)" if m in [12,1,2] else "📅 Restante"
    )
    top12_causas = df_causa["causa_acidente"].value_counts().head(12).index
    comp = (
        df_causa[df_causa["causa_acidente"].isin(top12_causas)]
        .groupby(["causa_acidente","periodo_turismo"]).size().reset_index(name="n")
    )
    fig = px.bar(
        comp.sort_values("n",ascending=True),
        x="n", y="causa_acidente", color="periodo_turismo",
        barmode="group", orientation="h",
        color_discrete_map={"🌞 Verão (Dez‑Fev)":"#eb6834","📅 Restante":"#2a78d6"},
    )
    fig.update_layout(yaxis_title="",xaxis_title="Sinistros",
                      legend_title="Período",font_family=FONT)
    st.plotly_chart(fig, use_container_width=True)

    df_lit_ver = df[df["mes_num"].isin([12,1,2]) & df["is_litoral"]]
    df_int_ver = df[df["mes_num"].isin([12,1,2]) & ~df["is_litoral"]]
    df_rest    = df[~df["mes_num"].isin([12,1,2])]

    def taxa(d): return (d["target"]==2).sum()/len(d)*100 if len(d) else 0

    t_lv = taxa(df_lit_ver)
    t_iv = taxa(df_int_ver)
    t_re = taxa(df_rest)

    ci1,ci2,ci3 = st.columns(3)
    ci1.metric("🏖️ Litoral · Verão", f"{t_lv:.2f}%", help="Taxa de fatalidade Dez‑Fev no litoral")
    ci2.metric("🏔️ Interior · Verão",f"{t_iv:.2f}%", help="Taxa de fatalidade Dez‑Fev no interior")
    ci3.metric("📅 Restante do Ano", f"{t_re:.2f}%", help="Taxa de fatalidade fora do verão")

    delta = t_lv - t_re
    cor = "kpi-red" if delta > 0 else "kpi-green"
    sinal = "+" if delta > 0 else ""
    st.markdown(
        f'<div class="kpi-card {cor}">📊 O litoral no verão apresenta taxa de fatalidade '
        f'<b>{sinal}{delta:.2f}p.p.</b> em relação ao restante do ano.</div>',
        unsafe_allow_html=True
    )

    with st.expander("📋 Recomendações de Fiscalização — Litoral SC", expanded=True):
        st.markdown("""
| Ação | Época Alvo | Rodovia | Impacto Estimado |
|------|-----------|---------|----------------|
| 🚨 **Radares e blitz de velocidade** | Dez‑Fev | BR‑101 litoral | ⭐⭐⭐⭐⭐ |
| 🍺 **Operação Balada Segura** (etilômetro) | Sex/Sáb noite · Dez‑Fev | BR‑101 | ⭐⭐⭐⭐⭐ |
| 📵 **Blitz anti-celular** | Todo o ano (pico: verão) | BR‑101, BR‑282 | ⭐⭐⭐⭐ |
| 😴 **Pontos de descanso obrigatórios** | Feriados e verão | BR‑101, BR‑116 | ⭐⭐⭐⭐ |
| 🐄 **Cercamento de pistas** (animais) | Todo o ano (pico: inverno) | BR‑282, BR‑158 | ⭐⭐⭐ |
| 🌫️ **Painéis de velocidade variável** (neblina) | Inverno | BR‑101 Serra, BR‑282 | ⭐⭐⭐⭐ |
| 🚧 **Sinalização reforçada em obras** | Jun‑Ago | BR‑282, BR‑470 | ⭐⭐⭐ |
| 💡 **Iluminação de pontos críticos** | Todo o ano (noite) | Toda malha | ⭐⭐⭐⭐ |
| ⚖️ **Postos de pesagem ativos** (caminhões) | Todo o ano | BR‑116, BR‑163 | ⭐⭐⭐ |
        """)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 3 · DIAGNÓSTICO POR BR
# ══════════════════════════════════════════════════════════════════════════════
with tab3:
    st.header("🔎 Diagnóstico por BR — Causa Principal e Sugestões")
    st.markdown(
        "Selecione uma BR para ver a análise detalhada: causa principal, "
        "fatores de risco, municípios mais críticos e recomendações específicas."
    )

    br_diag = st.selectbox("Selecione a BR para diagnóstico",
                           sorted(df["br"].unique(), key=int),
                           index=0,
                           key="br_diag")

    df_br = df[df["br"] == br_diag].copy()

    if df_br.empty:
        st.warning("Sem dados para esta BR com os filtros atuais.")
    else:
        total_br   = len(df_br)
        mortos_br  = int(df_br["mortos"].sum())
        feridos_br = int(df_br["feridos"].sum())
        fatais_br  = int((df_br["target"]==2).sum())
        taxa_br    = fatais_br/total_br*100

        k1,k2,k3,k4 = st.columns(4)
        k1.metric(f"BR‑{br_diag} · Sinistros", f"{total_br:,}".replace(",","."))
        k2.metric("Mortos", f"{mortos_br:,}".replace(",","."))
        k3.metric("Feridos", f"{feridos_br:,}".replace(",","."))
        k4.metric("Taxa de Fatalidade", f"{taxa_br:.2f}%")

        causas_br = df_br["causa_acidente"].value_counts()
        causa_principal = causas_br.index[0] if len(causas_br) else "N/D"
        n_causa_princ = causas_br.iloc[0] if len(causas_br) else 0
        pct_causa = n_causa_princ/total_br*100

        emoji_int, categoria_int, acao_int = sugerir_intervencao(causa_principal)

        st.markdown(f"""
<div class="kpi-card kpi-orange">
{emoji_int} <b>Causa principal na BR‑{br_diag}:</b>
<span class="tag tag-red">{causa_principal}</span>
— aparece em <b>{n_causa_princ:,} sinistros ({pct_causa:.1f}%)</b><br>
<b>Categoria:</b> {categoria_int} &nbsp;|&nbsp;
<b>Ação recomendada:</b> {acao_int}
</div>
""".replace(",","."), unsafe_allow_html=True)

        col_d1, col_d2 = st.columns(2)

        with col_d1:
            st.subheader(f"Top 10 Causas — BR‑{br_diag}")
            top10 = causas_br.head(10).reset_index()
            top10.columns = ["causa","n"]
            fig = px.bar(top10.sort_values("n"), x="n", y="causa",
                         orientation="h", color_discrete_sequence=[CORES[2]],
                         text="n")
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="",xaxis_title="Sinistros",
                               font_family=FONT,margin=dict(t=10))
            st.plotly_chart(fig, use_container_width=True)

        with col_d2:
            st.subheader(f"Gravidade por Época — BR‑{br_diag}")
            ep_br = (
                df_br.dropna(subset=["epoca","classificacao_acidente"])
                .groupby(["epoca","classificacao_acidente"]).size().reset_index(name="n")
            )
            ep_br["epoca"] = pd.Categorical(ep_br["epoca"],categories=EPOCAS_ORDEM,ordered=True)
            ep_br = ep_br.sort_values("epoca")
            fig = px.bar(ep_br, x="epoca", y="n", color="classificacao_acidente",
                         barmode="stack", color_discrete_map=COR_GRAVIDADE)
            fig.update_layout(xaxis_title="",yaxis_title="Sinistros",
                               legend_title="Gravidade",font_family=FONT)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"🏙️ Municípios com Mais Acidentes Graves — BR‑{br_diag}")
        mun_criticos = (
            df_br[df_br["target"]==2]
            .groupby("municipio")
            .agg(fatais=("target","count"), mortos=("mortos","sum"),
                 feridos=("feridos","sum"))
            .reset_index().sort_values("fatais",ascending=False).head(15)
        )
        if mun_criticos.empty:
            st.info("Nenhum acidente fatal com esses filtros.")
        else:
            fig = px.bar(mun_criticos, x="fatais", y="municipio",
                         orientation="h",
                         color="mortos",
                         color_continuous_scale=["#f2b84b","#d03b3b"],
                         text="fatais")
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="",xaxis_title="Acidentes Fatais",
                               coloraxis_colorbar_title="Mortos",
                               font_family=FONT)
            st.plotly_chart(fig, use_container_width=True)

        col_p1, col_p2 = st.columns(2)
        with col_p1:
            st.subheader(f"⏰ Horário mais crítico — BR‑{br_diag}")
            hour_df = (
                df_br.dropna(subset=["hora","target"])
                .groupby("hora")["target"]
                .apply(lambda s:(s==2).mean()*100).reset_index()
            )
            hour_df.columns = ["hora","taxa"]
            fig = px.area(hour_df, x="hora", y="taxa",
                          color_discrete_sequence=["#d03b3b"])
            fig.update_layout(xaxis_title="Hora do Dia",yaxis_title="% Fatais",
                               font_family=FONT)
            st.plotly_chart(fig, use_container_width=True)

        with col_p2:
            st.subheader(f"🌧️ Condição Climática vs. Fatalidade — BR‑{br_diag}")
            clim_df = (
                df_br.dropna(subset=["condicao_metereologica","target"])
                .groupby("condicao_metereologica")["target"]
                .apply(lambda s:(s==2).mean()*100).reset_index()
                .sort_values("target",ascending=False)
            )
            clim_df.columns = ["clima","taxa"]
            fig = px.bar(clim_df, x="taxa", y="clima", orientation="h",
                         color="taxa",
                         color_continuous_scale=["#1b9e77","#f2b84b","#d03b3b"],
                         text=clim_df["taxa"].map(lambda v:f"{v:.1f}%"))
            fig.update_traces(textposition="outside")
            fig.update_layout(yaxis_title="",xaxis_title="% Fatais",
                               coloraxis_showscale=False,font_family=FONT)
            st.plotly_chart(fig, use_container_width=True)

        st.subheader(f"💡 Sugestões de Melhoria para as Top Causas — BR‑{br_diag}")
        top_causas_list = causas_br.head(10).index.tolist()
        rows_sug = []
        for c in top_causas_list:
            em, cat, acao = sugerir_intervencao(c)
            rows_sug.append({"Causa": c,"Categoria": cat,"Ação Recomendada": acao,"Qtd": causas_br[c]})
        df_sug = pd.DataFrame(rows_sug)
        st.dataframe(
            df_sug,
            column_config={
                "Causa": st.column_config.TextColumn("Causa", width="large"),
                "Categoria": st.column_config.TextColumn("Categoria"),
                "Ação Recomendada": st.column_config.TextColumn("Ação Recomendada", width="large"),
                "Qtd": st.column_config.NumberColumn("Sinistros"),
            },
            hide_index=True, use_container_width=True
        )

        st.subheader(f"📆 Taxa de Fatalidade por Dia da Semana — BR‑{br_diag}")
        dsem_br = (
            df_br.dropna(subset=["dia_semana_nome","target"])
            .groupby("dia_semana_nome")["target"]
            .apply(lambda s:(s==2).mean()*100).reset_index()
        )
        dsem_br.columns = ["dia","taxa"]
        ordem_dias = ["Seg","Ter","Qua","Qui","Sex","Sáb","Dom"]
        dsem_br["dia"] = pd.Categorical(dsem_br["dia"],categories=ordem_dias,ordered=True)
        dsem_br = dsem_br.sort_values("dia")
        fig = px.bar(dsem_br, x="dia", y="taxa", color="dia",
                     color_discrete_sequence=CORES,
                     text=dsem_br["taxa"].map(lambda v:f"{v:.2f}%"))
        fig.update_traces(textposition="outside")
        fig.update_layout(showlegend=False,xaxis_title="",yaxis_title="% Fatais",
                          font_family=FONT)
        st.plotly_chart(fig, use_container_width=True)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 4 · PREVISÃO DE RISCO
# ══════════════════════════════════════════════════════════════════════════════
with tab4:
    st.header("🤖 Simulador de Previsão de Gravidade")

    if not modelo_ok:
        st.error(
            f"❌ Modelo não carregado: `{e_model_str}`\n\n"
            "Reexecute o notebook `02_preprocessing_modeling.ipynb` no seu ambiente "
            "local para regravar o `baseline_model.joblib` com as versões de libs instaladas."
        )
    else:
        st.markdown(f"""
Preencha o cenário abaixo. O modelo **{model_name}** (F1‑Score ponderado: **{model_score:.4f}**)
prevê a probabilidade de gravidade *antes* do evento, sem usar dados pós‑acidente.

> 🟢 **Sem Vítimas** · 🟡 **Com Feridos** · 🔴 **Fatal**
        """)

        st.divider()

        ci1, ci2, ci3 = st.columns(3)

        with ci1:
            st.subheader("🛣️ Rodovia e Local")
            br_sim = st.selectbox(
                "BR", sorted(df_full["br"].unique(), key=int), key="sim_br"
            )
            km_sim = st.number_input("KM", 0.0, 999.0, 200.0, 0.5, key="sim_km")
            zona_sim = st.selectbox(
                "Zona", ["🏖️ Litoral","🏔️ Interior"], key="sim_zona"
            )
            lat_sim = -27.5 if zona_sim.startswith("🏖️") else -27.3
            lon_sim = -48.5 if zona_sim.startswith("🏖️") else -50.0
            tipo_pista_sim = st.selectbox(
                "Tipo de Pista",
                sorted(df_full["tipo_pista"].dropna().unique()), key="sim_pista"
            )
            tracado_sim = st.selectbox(
                "Traçado da Via",
                ["Reta","Curva","Declive","Aclive","Viaduto;Reta",
                 "Curva;Declive","Rotatória;Curva","Ponte;Reta"],
                key="sim_tracado"
            )
            uso_solo_sim = st.selectbox("Área Urbana?",["Não","Sim"], key="sim_uso")

        with ci2:
            st.subheader("🕐 Tempo e Condições")
            mes_sim = st.selectbox(
                "Mês", list(range(1,13)),
                format_func=lambda m: f"{MESES_NOMES[m]} ({m})",
                index=0, key="sim_mes"
            )
            hora_sim = st.slider("Hora do Dia", 0, 23, 14, key="sim_hora")
            dia_sem_sim = st.selectbox(
                "Dia da Semana",
                ["Seg (0)","Ter (1)","Qua (2)","Qui (3)","Sex (4)","Sáb (5)","Dom (6)"],
                index=4, key="sim_dia"
            )
            dia_num_sim = int(dia_sem_sim.split("(")[1].replace(")",""))
            fds_sim = eh_fim_de_semana(dia_num_sim)
            feriado_sim = st.checkbox("É feriado nacional?", key="sim_feriado")
            clima_sim = st.selectbox(
                "Condição Climática",
                sorted(df_full["condicao_metereologica"].dropna().unique()),
                key="sim_clima"
            )

            periodo_sim = classificar_periodo(hora_sim)

            if hora_sim in [5,6,7]:       fase_sim = "Amanhecer"
            elif hora_sim in [17,18,19]:  fase_sim = "Anoitecer"
            elif hora_sim < 6 or hora_sim >= 20: fase_sim = "Plena Noite"
            else:                          fase_sim = "Pleno dia"

        with ci3:
            st.subheader("🚗 Acidente")
            tipo_ac_sim = st.selectbox(
                "Tipo de Acidente",
                sorted(df_full["tipo_acidente"].dropna().unique()),
                key="sim_tipo_ac"
            )
            sentido_sim = st.selectbox(
                "Sentido da Via",["Crescente","Decrescente"], key="sim_sentido"
            )

        st.divider()
        prever = st.button("🔮 Prever Gravidade", type="primary", use_container_width=True)

        if prever:
            entrada = {
                "br":                     br_sim,
                "fase_dia":               fase_sim,
                "sentido_via":            sentido_sim,
                "condicao_metereologica": clima_sim,
                "tipo_pista":             tipo_pista_sim,
                "tracado_via":            tracado_sim,
                "uso_solo":               uso_solo_sim,
                "periodo_do_dia":         periodo_sim,
                "tipo_acidente":          tipo_ac_sim,
                "km":                     km_sim,
                "latitude":               lat_sim,
                "longitude":              lon_sim,
                "mes":                    mes_sim,
                "dia_da_semana":          dia_num_sim,
                "eh_fim_de_semana":       fds_sim,
                "eh_feriado":             int(feriado_sim),
            }
            df_in = pd.DataFrame([entrada])
            cols_cat = [c for c in features_cat_ml if c in df_in.columns]
            cols_num = [c for c in features_num_ml if c in df_in.columns]
            df_model_in = df_in[cols_cat + cols_num]

            try:
                probs    = pipeline_ml.predict_proba(df_model_in)[0]
                pred_cls = int(np.argmax(probs))
                pred_lbl = classes_ml[pred_cls]

                emojis = {0:"🟢",1:"🟡",2:"🔴"}
                cores_cls = {0:"#1b9e77",1:"#f2b84b",2:"#d03b3b"}
                st.markdown(
                    f'<div class="kpi-card" style="border-color:{cores_cls[pred_cls]};'
                    f'background:rgba(0,0,0,.04);font-size:1.15rem">'
                    f'{emojis[pred_cls]} <b>Previsão: {pred_lbl}</b> — '
                    f'confiança <b>{probs[pred_cls]*100:.1f}%</b></div>',
                    unsafe_allow_html=True
                )

                res1, res2 = st.columns([1,1])

                with res1:
                    labels_cls = [classes_ml[i] for i in range(3)]
                    fig = go.Figure(go.Bar(
                        x=labels_cls,
                        y=[p*100 for p in probs],
                        marker_color=["#1b9e77","#f2b84b","#d03b3b"],
                        text=[f"{p*100:.1f}%" for p in probs],
                        textposition="outside",
                    ))
                    fig.update_layout(
                        title="Probabilidade por Classe",
                        yaxis=dict(title="%",range=[0,110]),
                        xaxis_title="Classe",
                        font_family=FONT,
                    )
                    st.plotly_chart(fig, use_container_width=True)

                with res2:
                    risk = {
                        "Noturno/Madrugada": 1.0 if hora_sim<6 or hora_sim>=22 else
                                             (0.6 if hora_sim>=19 else 0.2),
                        "Fim de Semana":     0.8 if fds_sim else 0.2,
                        "Chuva/Neblina":     0.9 if clima_sim in [
                            "Chuva","Nevoeiro/Neblina","Garoa/Chuvisco","Granizo"] else 0.15,
                        "Pista Simples":     0.85 if tipo_pista_sim=="Simples" else
                                             (0.4 if tipo_pista_sim=="Dupla" else 0.1),
                        "Alta Temporada":    0.9 if mes_sim in [12,1,2] else
                                             (0.5 if mes_sim in [6,7] else 0.2),
                        "Tipo Alto Risco":   0.95 if tipo_ac_sim in [
                            "Colisão frontal","Tombamento","Atropelamento de Pedestre",
                            "Queda de ocupante de veículo","Capotamento"] else 0.25,
                        "Feriado":           0.75 if feriado_sim else 0.1,
                        "Curva/Declive":     0.8 if any(t in tracado_sim
                            for t in ["Curva","Declive","Aclive"]) else 0.2,
                    }
                    fig_r = go.Figure(go.Scatterpolar(
                        r=list(risk.values()), theta=list(risk.keys()),
                        fill="toself",
                        fillcolor=f"rgba({','.join(str(int(v)) for v in (208,59,59))},.2)",
                        line_color="#d03b3b", name="Nível de Risco",
                    ))
                    fig_r.update_layout(
                        title="Radar de Fatores de Risco",
                        polar=dict(radialaxis=dict(visible=True,range=[0,1])),
                        showlegend=False, font_family=FONT,
                    )
                    st.plotly_chart(fig_r, use_container_width=True)

                if pred_cls == 2:
                    st.error("🚨 **Alto risco de acidente fatal.** Reveja velocidade, nível de cansaço e condição da via antes de prosseguir.")
                elif pred_cls == 1:
                    st.warning("⚠️ **Risco de acidente com feridos.** Atenção redobrada é recomendada nestas condições.")
                else:
                    st.success("✅ **Menor risco de vítimas.** Condições relativamente seguras — mas atenção nunca é demais.")

                st.subheader("💡 O que pode melhorar neste cenário?")
                sugestoes = []
                if hora_sim < 6 or hora_sim >= 22:
                    sugestoes.append(("💡","Iluminação de pontos críticos e câmeras noturnas na rodovia"))
                if fds_sim or feriado_sim:
                    sugestoes.append(("🚔","Reforço de equipes PRF nos finais de semana e feriados"))
                if clima_sim in ["Chuva","Nevoeiro/Neblina","Garoa/Chuvisco"]:
                    sugestoes.append(("🌧️","Painel de mensagem variável com redução de velocidade automática"))
                if tipo_pista_sim == "Simples":
                    sugestoes.append(("🛣️","Duplicação da pista ou implantação de faixa adicional em trechos críticos"))
                if mes_sim in [12,1,2]:
                    sugestoes.append(("🌞","Operação Verão com postos fixos de abordagem a cada 50 km na BR"))
                if tipo_ac_sim in ["Colisão frontal","Tombamento","Capotamento"]:
                    sugestoes.append(("🛡️","Guard-rails, barreiras central e cables de aço em trechos com histórico"))
                if any(t in tracado_sim for t in ["Curva","Declive"]):
                    sugestoes.append(("⬇️","Redutor de velocidade + alerta sonoro em curvas e descidas acentuadas"))
                if fds_sim or mes_sim in [12,1,2]:
                    sugestoes.append(("😴","Área de descanso com parada obrigatória a cada 2h de viagem"))

                if sugestoes:
                    for em, txt in sugestoes:
                        st.markdown(
                            f'<div class="kpi-card kpi-blue">{em} {txt}</div>',
                            unsafe_allow_html=True
                        )
                else:
                    st.info("Cenário sem fatores de alto risco identificados nas regras heurísticas.")

            except Exception as exc:
                st.error(f"Erro ao rodar o modelo: {exc}")

        st.divider()
        st.subheader("📊 Gravidade Histórica por BR × Época (dados filtrados)")
        st.caption("Use os filtros globais para refinar. Células em % de acidentes fatais.")

        pivot = (
            df.dropna(subset=["br","epoca","target"])
            .groupby(["br","epoca"])["target"]
            .apply(lambda s:(s==2).mean()*100)
            .reset_index()
        )
        pivot.columns = ["br","epoca","taxa_fatal"]
        pivot_wide = pivot.pivot(index="br",columns="epoca",values="taxa_fatal").fillna(0)
        pivot_wide = pivot_wide.reindex(columns=[c for c in EPOCAS_ORDEM if c in pivot_wide.columns])

        fig_heat = px.imshow(
            pivot_wide.values,
            x=list(pivot_wide.columns),
            y=list(pivot_wide.index),
            color_continuous_scale=["#1b9e77","#f2b84b","#d03b3b"],
            text_auto=".1f",
            labels=dict(x="Época",y="BR",color="% Fatais"),
            aspect="auto",
        )
        fig_heat.update_layout(font_family=FONT,coloraxis_colorbar_title="% Fatais")
        st.plotly_chart(fig_heat, use_container_width=True)
        st.caption("Leitura: quanto mais vermelho, maior a taxa de acidentes fatais naquela BR × época.")

        with st.expander("ℹ️ Sobre o Modelo e Limitações"):
            st.markdown(f"""
**Modelo:** `{model_name}` · **F1‑Score ponderado:** `{model_score:.4f}`  
**Features categóricas:** `{', '.join(features_cat_ml)}`  
**Features numéricas:** `{', '.join(features_num_ml)}`

#### ⚠️ Limitações
- Treinado com dados históricos de SC.
- `tipo_acidente` é informado *após* o evento na fonte PRF — no simulador representa o **tipo provável esperado** (ex: "colisão frontal em pista molhada com neblina").
- Dados desbalanceados: fatais são ~5‑8% dos casos. Compensado com `class_weight="balanced"`.
- **Não substitui** julgamento operacional de agentes da PRF.

#### 🔄 Como melhorar o modelo
- Incluir `hora` e `municipio` como features diretas.
- Adicionar dados de fluxo veicular (DNIT) e histórico de obras.
- Retreinar com feature de densidade turística (mês × litoral).
- Avaliar SHAP values para explicabilidade por predição.
            """)


# ══════════════════════════════════════════════════════════════════════════════
# ABA 5 · MAPA DE RISCO
# ══════════════════════════════════════════════════════════════════════════════
with tab5:
    st.header("🗺️ Mapa de Risco Interativo")

    cm1, cm2 = st.columns([1,3])

    with cm1:
        st.subheader("Filtros do Mapa")
        grav_mapa = st.multiselect(
            "Gravidade", grav_disp, default=["Com Vítimas Fatais","Com Vítimas Feridas"],
            key="mapa_grav"
        )
        epocas_mapa = st.multiselect(
            "Época", EPOCAS_ORDEM, default=EPOCAS_ORDEM, key="mapa_epoca"
        )
        max_pts = st.slider("Máx. pontos", 500, 12000, 6000, 500, key="mapa_pts")
        cor_por = st.radio(
            "Colorir por",
            ["Gravidade","Época","Tipo de Pista","Condição Climática"],
            key="mapa_cor"
        )

    mapa_df = df.dropna(subset=["latitude","longitude","classificacao_acidente"])
    if grav_mapa:
        mapa_df = mapa_df[mapa_df["classificacao_acidente"].isin(grav_mapa)]
    if epocas_mapa:
        mapa_df = mapa_df[mapa_df["epoca"].isin(epocas_mapa)]
    mapa_df = mapa_df[
        mapa_df["latitude"].between(-30,-25) &
        mapa_df["longitude"].between(-54,-47)
    ]
    if len(mapa_df) > max_pts:
        mapa_df = mapa_df.sample(max_pts, random_state=42)

    with cm2:
        if mapa_df.empty:
            st.warning("Nenhum ponto com esses filtros. Ajuste na barra lateral ou nos filtros do mapa.")
        else:
            cor_col_map = {
                "Gravidade":          ("classificacao_acidente", COR_GRAVIDADE),
                "Época":              ("epoca", None),
                "Tipo de Pista":      ("tipo_pista", None),
                "Condição Climática": ("condicao_metereologica", None),
            }
            color_col, color_map = cor_col_map[cor_por]

            kwargs_mapa = dict(
                lat="latitude", lon="longitude",
                color=color_col,
                hover_data={
                    "municipio":True,"br":True,"km":True,
                    "causa_acidente":True,"tipo_acidente":True,
                    "data_inversa":True,"classificacao_acidente":True,
                },
                zoom=7, center={"lat":-27.5,"lon":-50.0},
                opacity=0.65,
                title=f"Mapa de Sinistros — {len(mapa_df):,} pontos".replace(",","."),
            )
            if color_map:
                kwargs_mapa["color_discrete_map"] = color_map

            if hasattr(px, "scatter_map"):
                fig_mapa = px.scatter_map(mapa_df, **kwargs_mapa)
                fig_mapa.update_layout(
                    map_style="carto-positron",
                    margin=dict(l=0,r=0,t=40,b=0),
                    legend_title=cor_por,
                    font_family=FONT,
                )
            else:
                fig_mapa = px.scatter_mapbox(mapa_df, **kwargs_mapa)
                fig_mapa.update_layout(
                    mapbox_style="carto-positron",
                    margin=dict(l=0,r=0,t=40,b=0),
                    legend_title=cor_por,
                    font_family=FONT,
                )
            st.plotly_chart(fig_mapa, use_container_width=True)

    st.subheader("🔥 Hotspots — Municípios e KMs Mais Críticos")
    ht1, ht2 = st.columns(2)

    with ht1:
        st.markdown("**Por município**")
        hot_mun = (
            mapa_df[mapa_df["classificacao_acidente"].isin(
                ["Com Vítimas Fatais","Com Vítimas Feridas"]
            )]
            .groupby(["municipio","br"])
            .agg(graves=("target","count"), mortos=("mortos","sum"), feridos=("feridos","sum"))
            .reset_index().sort_values("graves",ascending=False).head(20)
        )
        if hot_mun.empty:
            st.info("Sem dados graves para esses filtros.")
        else:
            st.dataframe(
                hot_mun.rename(columns={
                    "municipio":"Município","br":"BR",
                    "graves":"Acidentes Graves","mortos":"Mortos","feridos":"Feridos"
                }),
                hide_index=True, use_container_width=True
            )

    with ht2:
        st.markdown("**Por KM (trechos críticos)**")
        if "km" in mapa_df.columns:
            mapa_df_km = mapa_df.copy()
            mapa_df_km["km_num"] = pd.to_numeric(
                mapa_df_km["km"].astype(str).str.replace(",","."), errors="coerce"
            )
            mapa_df_km["km_bloco"] = (mapa_df_km["km_num"]//5*5).astype("Int64")
            hot_km = (
                mapa_df_km[mapa_df_km["classificacao_acidente"]=="Com Vítimas Fatais"]
                .groupby(["br","km_bloco"])
                .agg(fatais=("target","count"), mortos=("mortos","sum"))
                .reset_index().sort_values("fatais",ascending=False).head(20)
            )
            hot_km["trecho"] = "BR-" + hot_km["br"] + " · KM " + hot_km["km_bloco"].astype(str) + "‑" + (hot_km["km_bloco"]+5).astype(str)
            st.dataframe(
                hot_km[["trecho","fatais","mortos"]].rename(columns={
                    "trecho":"Trecho","fatais":"Acidentes Fatais","mortos":"Mortos"
                }),
                hide_index=True, use_container_width=True
            )


# ──────────────────────────────────────────────────────────────────────────────
# FOOTER
# ──────────────────────────────────────────────────────────────────────────────

st.markdown("""
<div class="app-footer">
    Desenvolvido por Henrique Ribeiro Rodrigues e Jean Gondorek — Planejamento e Gestão de Projetos<br>
    &copy; 2026 Todos os direitos reservados · Dados: Polícia Rodoviária Federal (PRF) · SC
</div>
""", unsafe_allow_html=True)

