## Context

A detecção e prevenção de ataques sintéticos (deepfakes) requer não apenas a identificação de ataques isolados, mas também o acompanhamento do volume de ataques ao longo do tempo. Atualmente, temos dados históricos de volumes ("Volume_Deepfakes"), mas carecemos de uma projeção futura. A implementação de um modelo ARIMA nos permitirá entender matematicamente a carga e a tendência linear desses ataques, possibilitando ações preventivas embasadas.

## Goals / Non-Goals

**Goals:**
- Ajustar um modelo estatístico confiável (ARIMA) para prever o volume de deepfakes nos próximos 30 dias.
- Implementar as rotinas matemáticas necessárias para o diagnóstico da série temporal (estacionariedade e resíduos).
- Fornecer suporte visual (gráfico sobreposto) que compare historicamente os volumes reais com a previsão.

**Non-Goals:**
- Não se objetiva implementar modelos de Machine Learning complexos (ex: LSTMs, Transformers) neste primeiro momento; o foco é em métodos estatísticos tradicionais.
- A previsão não considerará neste escopo correlações com variáveis exógenas (ARIMAX).

## Decisions

- **Uso do statsmodels**: Utilizaremos a biblioteca `statsmodels.tsa.arima.model.ARIMA`. Racional: É a implementação padrão e mais madura para modelos ARIMA em Python.
- **Teste de Estacionariedade (Dickey-Fuller)**: Necessário para garantir que os dados de entrada no ARIMA não possuam raiz unitária. Isso ditará o valor de `d` na ordem do modelo.
- **Ajuste Manual e Visual**: Os parâmetros `(p, d, q)` serão ajustados possivelmente baseados em análises prévias de gráficos ACF e PACF.
- **Diagnóstico de Resíduos**: Garantir que os resíduos se comportem como ruído branco, validando o fit do modelo.

## Risks / Trade-offs

- **Série Temporal não linear ou muito ruidosa** -> Mitigação: O modelo ARIMA pode não capturar toda a complexidade, mas serve como baseline sólida. Caso falhe, diferenciações adicionais ou transformações logarítmicas podem ser aplicadas.
- **Overfitting dos parâmetros (p,q)** -> Mitigação: Manter a análise de resíduos e parcimônia na escolha da ordem do modelo.
