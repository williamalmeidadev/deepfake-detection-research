## 1. Setup e Diagnóstico Inicial

- [x] 1.1 Importar `ARIMA` do módulo `statsmodels.tsa.arima.model`.
- [x] 1.2 Testar a estacionariedade da série temporal `Volume_Deepfakes` usando o Teste Dickey-Fuller.
- [x] 1.3 Definir a ordem de diferenciação (d) com base no teste de estacionariedade.

## 2. Modelagem e Previsão

- [x] 2.1 Instanciar e ajustar (`fit()`) o modelo ARIMA com os hiperparâmetros (p, d, q) apropriados.
- [x] 2.2 Verificar os resíduos do modelo para garantir que se comportam como ruído branco.
- [x] 2.3 Executar a previsão (`forecast`) para 30 dias futuros (`steps=30`).
- [x] 2.4 Gerar gráfico sobrepondo os dados reais históricos de volume contra a curva de previsão gerada.
