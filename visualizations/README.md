# Visualizações

Esta pasta contém gráficos e visualizações geradas pela análise do modelo.

## Arquivos Gerados

- **model_analysis.png** - Painel completo com 6 visualizações do modelo:
  - Matriz de Confusão
  - Métricas por Classe (Precision, Recall, F1-Score)
  - Acurácia Geral
  - Top 15 Features Positivas
  - Top 15 Features Negativas
  - Distribuição de Predições

## Como Gerar

As visualizações são geradas automaticamente ao executar:

```bash
python src/sentiment_classifier.py
```

## Tecnologias

- **matplotlib** - Biblioteca principal de plotagem
- **seaborn** - Visualizações estatísticas aprimoradas

> **Nota:** As imagens são salvas em alta resolução (300 DPI) para melhor qualidade.
