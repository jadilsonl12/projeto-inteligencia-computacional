# Modelos Treinados

Esta pasta contém os modelos treinados e vetorizadores salvos durante a execução do projeto.

## Arquivos Gerados

- **sentiment_model.pkl** - Modelo de classificação treinado (Logistic Regression)
- **tfidf_vectorizer.pkl** - Vetorizador TF-IDF configurado com os parâmetros do treinamento

## Como Usar

Os modelos são gerados automaticamente ao executar `src/sentiment_classifier.py` e podem ser carregados posteriormente para fazer predições sem precisar treinar novamente.

```python
classifier = SentimentClassifier()
classifier.load_model()
sentiment, confidence = classifier.predict_sentiment("This movie is amazing!")
```

> **Nota:** Os arquivos `.pkl` são ignorados pelo git conforme configurado no `.gitignore`.
