## ADDED Requirements

### Requirement: Teste de Estacionariedade
O sistema SHALL fornecer capacidade de testar a série temporal `Volume_Deepfakes` quanto à estacionariedade usando o Teste Dickey-Fuller.

#### Scenario: Série requer diferenciação
- **WHEN** o p-value do teste Dickey-Fuller é maior que 0.05
- **THEN** o hiperparâmetro `d` do ARIMA deve ser ajustado para tornar a série estacionária (d >= 1)

### Requirement: Ajuste do Modelo ARIMA e Previsão
O sistema SHALL permitir o instanciamento e ajuste de um modelo ARIMA utilizando os parâmetros `(p, d, q)` definidos, gerando previsões para 30 passos no futuro.

#### Scenario: Previsão bem sucedida
- **WHEN** o modelo é ajustado (`fit()`) com dados válidos de volume de deepfakes
- **THEN** o método `forecast(steps=30)` deve retornar 30 valores futuros projetados

### Requirement: Diagnóstico de Resíduos
O sistema SHALL incluir a verificação dos resíduos do modelo ajustado para assegurar que se assemelham a um ruído branco.

#### Scenario: Resíduos validados
- **WHEN** a análise de resíduos é executada
- **THEN** não deve haver autocorrelação significativa remanescente nos lags do modelo

### Requirement: Visualização de Dados e Previsão
O sistema SHALL gerar um gráfico que sobreponha os dados reais históricos de volume com a curva de previsão gerada.

#### Scenario: Geração de gráfico
- **WHEN** o modelo finaliza a previsão de 30 dias
- **THEN** um gráfico será exibido mostrando a série original concatenada/sobreposta com a linha de projeção futura
