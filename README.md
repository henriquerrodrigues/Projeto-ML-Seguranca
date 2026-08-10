# Plataforma Inteligente para Apoio à Decisão em Segurança Viária

Projeto acadêmico desenvolvido na Universidade Federal da Fronteira Sul (UFFS) para a disciplina de PGP, focado no mapeamento de manchas críticas (*hotspots*) e na predição da severidade de sinistros de trânsito em rodovias federais.

---

## 1. Problema e Objetivo

### Problema
Apesar da grande disponibilidade de microdados abertos disponibilizados pela Polícia Rodoviária Federal (PRF), a gestão da segurança viária e a alocação de recursos operacionais de fiscalização ocorrem predominantemente de forma reativa e baseada em estatísticas descritivas consolidadas *a posteriori*. Análises estáticas convencionais falham em capturar padrões complexos e não lineares entre o histórico de infrações e a severidade real dos acidentes em trechos e horários específicos.

### Objetivo
Desenvolver uma plataforma de apoio à decisão em segurança viária baseada em modelos de *Machine Learning* e dashboards interativos. A solução utiliza os microdados abertos de acidentes e infrações da PRF para delimitar trechos críticos e classificar a gravidade de sinistros em todas as rodovias federais do estado de Santa Catarina.

---

## 2. Escopo do Projeto (Sprint 0)

* **Recorte Geográfico:** Rodovias federais que cortam o estado de Santa Catarina (`uf == 'SC'`).
* **Período dos Dados:** Ocorrências e autuações de 2020 a 2026.
* **Fontes de Dados:** Portal de Dados Abertos da Polícia Rodoviária Federal (Bases do Boletim de Acidente de Trânsito - BAT e Sistema de Infrações - SISCOM/AUTOPRF).
* **Variável-Alvo (Target do ML):** `classificacao_acidente` (*Sem Vítimas*, *Com Vítimas Feridas*, *Com Vítimas Fatais*).
* **Modelos Previstos:**
  * **Supervisionado (Classificação):** XGBoost, Random Forest e LightGBM para severidade.
  * **Não Supervisionado (Clustering):** DBSCAN / HDBSCAN para manchas geoespaciais de risco.
* **Entregável Final:** Pipeline de ETL + Modelos de ML + Dashboard Interativo (Mapas de Calor, Filtros Espaço-Temporais e Simuladores de Risco) + Artigo Científico SBC.

---

## 3. Equipe

* **Jean C. C. Gondorek** – `<jean.gondorek@estudante.uffs.edu.br>`
* **Henrique R. Rodrigues** – `<henrique.ribeiro@uffs.edu.br>`
* **Instituição:** Universidade Federal da Fronteira Sul (UFFS) – Campus Chapecó, SC

---

## 4. Estrutura do Repositório

Pastas principais do projeto:

- `article/`: Artigo científico no formato SBC construído incrementalmente a cada sprint.
- `app/`: Aplicação web do dashboard interativo.
- `data/`: Dados brutos, processados e artefatos intermediários (ignorados no git via `.gitignore`).
- `docs/`: Documentação de apoio, incluindo PM Canvas, Backlog, Kanban e atas de reuniões.
- `models/`: Modelos de ML treinados e relatórios de avaliação de desempenho.
- `notebooks/`: Notebooks Jupyter de Análise Exploratória (EDA) e experimentação de algoritmos.
- `src/`: Código-fonte modularizado para ETL, pipelines de ML, algoritmos geoespaciais e backend do dashboard.
- `tests/`: Testes unitários e de integração das rotinas de processamento e dados.
- `.github/`: Automações, templates de *Issues* e configurações do GitHub Projects.

---

## 5. Fluxo de Trabalho (Sprints)

1. **Sprint 0 - Planejamento:** Definição do escopo, Canvas, Backlog inicial e Seção 1 da Introdução/Problema do artigo.
2. **Sprint 1 - ETL e EDA:** Coleta, limpeza, padronização geoespacial dos CSVs da PRF e análise exploratória de dados.
3. **Sprint 2 - Eng. de Features e Modelagem:** Construção do dataset final, treinamento dos algoritmos de classificação e agrupamento (*hotspots*).
4. **Sprint 3 - Dashboard e Artigo Final:** Desenvolvimento da interface visual interativa e escrita final dos resultados do artigo.

---

## 6. Licença

Este projeto está licenciado sob a licença MIT. Veja o arquivo `LICENSE` para mais detalhes.