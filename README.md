# Previsão de Retornos e Recomendação de Portfólio — S&P 500

Trabalho final da disciplina de Aprendizagem de Máquina (Doutorado — UFC).

O objetivo é prever o retorno de ativos do S&P 500 no horizonte de 21 dias úteis (~1 mês) e, com base nas previsões, agrupar os ativos por perfil de risco para suporte à recomendação de portfólio.

---

## Estrutura do Projeto

```
notebooks/
├── 01_download_data.ipynb          # Coleta de dados via Wikipedia + yfinance
├── 02_feature_engineering.ipynb    # Indicadores técnicos e target de retorno
├── 03_forecasting_models.ipynb     # Modelos de previsão (abordagem tabular)
├── 04_clustering_portfolio.ipynb   # Clusterização de ativos e avaliação de portfólio
└── 05_timeseries_rnn.ipynb         # Modelos de séries temporais (LSTM via DARTS)

data/
├── raw/                            # Dados brutos baixados pelo NB01
└── processed/                      # Features, previsões e recomendações
```

---

## Pipeline

### NB01 — Coleta de Dados
Download dos ~500 ativos do S&P 500 via Wikipedia e yfinance, cobrindo o período de 2018 a 2026.

### NB02 — Feature Engineering
Construção de 11 indicadores técnicos por ativo:
retornos em múltiplos horizontes (1d, 5d, 21d), volatilidade (21d, 63d), razão com médias móveis (SMA20, SMA50), volume relativo, drawdown, posição no range e high-low range. Target: retorno simples e log-retorno dos próximos 21 dias.

### NB03 — Modelos de Previsão (Abordagem Tabular)
Cada observação é um par `(ativo, data)`. Um único modelo é treinado em todos os ativos simultaneamente.

**Split temporal:** treino 2018–2022 / validação 2023 / teste 2024–2026

**Modelos:** Regressão Linear · Random Forest · LightGBM

**Métrica principal:** IC (Information Coefficient) — correlação de Spearman entre retorno previsto e retorno real, calculada cross-sectionalmente por data. IC > 0,02 indica sinal utilizável (Grinold & Kahn, 2000).

| Modelo | IC | ICIR |
|---|---|---|
| Regressão Linear | **0,0382** | 0,199 |
| LightGBM | 0,0352 | 0,210 |
| Random Forest | 0,0279 | 0,177 |

### NB04 — Clusterização de Ativos
Agrupamento dos ativos em 3 clusters com K-Means usando volatilidade média, retorno previsto médio e drawdown. Os clusters são mapeados para perfis de investidor (**Conservador**, **Moderado**, **Arrojado**) pela volatilidade crescente. HDBSCAN é utilizado como validação do número natural de clusters nos dados.

O perfil do usuário determina de qual cluster os ativos são selecionados. A avaliação simula um rebalanceamento mensal selecionando os 20 ativos com maior retorno previsto dentro do cluster correspondente.

### NB05 — Abordagem de Séries Temporais (LSTM)
LSTM global treinado em todos os ativos simultaneamente via DARTS (`BlockRNNModel`). O modelo recebe uma janela de 63 dias de log-retornos e 11 covariáveis passadas e prevê os próximos 21 retornos diários. O IC é calculado da mesma forma que no NB03 para comparação direta entre as abordagens.

---

## Como Executar

```bash
# 1. Criar e ativar o ambiente
conda create -n financial_market python=3.10
conda activate financial_market

# 2. Instalar dependências
pip install -r requirements.txt

# 3. Executar os notebooks em ordem
jupyter notebook
```

> **Atenção:** o NB05 requer PyTorch (instalado automaticamente com `darts`) e pode levar 20–40 minutos em CPU.

---

## Dependências Principais

| Pacote | Uso |
|---|---|
| `yfinance` | Download de dados históricos |
| `scikit-learn` | Modelos, clustering, pré-processamento |
| `lightgbm` | Gradient Boosting |
| `darts` | LSTM global (séries temporais) |
| `scipy` | Cálculo do IC (Spearman) |
