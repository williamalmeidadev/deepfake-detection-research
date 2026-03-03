## Why

Precisamos estabelecer matematicamente qual será a carga e tendência linear de futuros ataques sintéticos (deepfakes). O modelo ARIMA mapeia o passado imediato para apontar o futuro próximo de forma precisa, permitindo antecipar os volumes de ataques para os próximos 30 dias.

## What Changes

- Implementação de um modelo ARIMA (AutoRegressive Integrated Moving Average) para previsão de séries temporais.
- Realização de testes de estacionariedade (Teste Dickey-Fuller) e definição da ordem de diferenciação (d).
- Ajuste de hiperparâmetros (p, d, q) do modelo para encontrar o melhor "fit" da curva.
- Validação do modelo instanciado com o método `.fit()`, verificação de resíduos (garantir ruído branco) e forecasting para passos futuros (30 dias).
- Criação de um gráfico sobrepondo os dados reais de volume de deepfakes ("Volume_Deepfakes") contra a previsão gerada.

## Capabilities

### New Capabilities
- `arima-time-series-forecast`: Capacidade de modelar e prever o volume futuro de deepfakes nos próximos 30 dias utilizando métodos estatísticos clássicos (ARIMA), incluindo diagnóstico de estacionariedade e análise de resíduos.

### Modified Capabilities

## Impact

- Necessidade de importar `ARIMA` do módulo `statsmodels.tsa.arima.model`.
- Adição de trechos de código de diagnóstico estatístico e visualizações no pipeline analítico de séries temporais de deepfakes.
