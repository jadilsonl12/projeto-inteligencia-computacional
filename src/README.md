# Documentação do Código - sentiment_classifier.py

Este documento explica detalhadamente cada função e classe do código de classificação de sentimentos.

---

## 📋 Índice

1. [Classe SentimentClassifier](#classe-sentimentclassifier)
2. [Métodos da Classe](#métodos-da-classe)
3. [Função generate_visualizations](#função-generate_visualizations)
4. [Função main](#função-main)

---

## Classe SentimentClassifier

Classe principal responsável por todo o pipeline de classificação de sentimentos em avaliações de filmes do IMDb.

### `__init__(self, model_type='logistic_regression', max_features=5000)`

**Descrição:** Construtor da classe que inicializa o classificador de sentimentos.

**Parâmetros:**
- `model_type` (str): Tipo de modelo a ser usado. Opções:
  - `'logistic_regression'` - Regressão Logística (padrão, recomendado)
  - `'naive_bayes'` - Naive Bayes Multinomial
- `max_features` (int): Número máximo de palavras/features que o TF-IDF vai considerar (padrão: 5000)

**O que faz:**
1. Armazena o tipo de modelo escolhido
2. Define o número máximo de features
3. Inicializa os atributos `vectorizer` e `model` como `None`
4. Cria uma instância do `WordNetLemmatizer` do NLTK para lematização
5. Chama `_download_nltk_resources()` para garantir que os recursos do NLTK estejam disponíveis

**Exemplo de uso:**
```python
# Criar classificador padrão (Regressão Logística, 5000 features)
classifier = SentimentClassifier()

# Criar com Naive Bayes e 10000 features
classifier = SentimentClassifier(model_type='naive_bayes', max_features=10000)
```

---

### `_get_project_dirs()` (método estático)

**Descrição:** Retorna os caminhos absolutos dos diretórios principais do projeto.

**Parâmetros:** Nenhum (método estático)

**Retorna:** Dictionary com os caminhos:
- `'models'` - Pasta onde modelos treinados são salvos
- `'data'` - Pasta onde dados processados são salvos
- `'visualizations'` - Pasta onde gráficos são salvos

**O que faz:**
1. Usa `Path(__file__)` para obter o caminho do arquivo atual
2. Navega para o diretório pai (raiz do projeto)
3. Constrói caminhos para as três pastas principais
4. Retorna um dicionário com os caminhos

**Exemplo de uso:**
```python
dirs = SentimentClassifier._get_project_dirs()
print(dirs['models'])  # D:\projeto-inteligencia-computacional\models
```

---

### `_download_nltk_resources(self)`

**Descrição:** Garante que todos os recursos necessários do NLTK estejam baixados e disponíveis.

**Parâmetros:** Nenhum

**Retorna:** Nenhum

**O que faz:**
1. Define uma lista de recursos NLTK necessários:
   - `punkt` - Tokenizador de sentenças
   - `punkt_tab` - Tabelas do tokenizador (NLTK 3.9+)
   - `stopwords` - Lista de palavras comuns (a, the, is, etc.)
   - `wordnet` - Base de dados lexical para lematização
   - `omw-1.4` - Open Multilingual Wordnet
2. Para cada recurso, tenta encontrá-lo no sistema
3. Se não encontrar, faz o download automaticamente
4. Mostra mensagem no console quando está baixando

**Por que é importante:** Sem esses recursos, o pré-processamento de texto falharia.

---

### `preprocess_text(self, text)`

**Descrição:** Realiza todo o pré-processamento de texto necessário para análise de sentimentos.

**Parâmetros:**
- `text` (str): Texto bruto a ser processado (review do filme)

**Retorna:** String com o texto limpo e processado

**O que faz (passo a passo):**

1. **Decodificar entidades HTML:**
   ```python
   text = unescape(text)  # &amp; → &, &lt; → <
   ```

2. **Remover tags HTML:**
   ```python
   text = re.sub(r'<.*?>', '', text)  # <br>, <p>, etc.
   ```

3. **Converter para minúsculas:**
   ```python
   text = text.lower()  # "GREAT Movie" → "great movie"
   ```

4. **Remover URLs:**
   ```python
   text = re.sub(r'http\S+|www\S+', '', text)
   ```

5. **Remover caracteres especiais e números:**
   ```python
   text = re.sub(r'[^a-z\s]', '', text)  # Mantém apenas letras e espaços
   ```

6. **Tokenização (dividir em palavras):**
   ```python
   tokens = word_tokenize(text)  # "great movie" → ["great", "movie"]
   ```

7. **Remover stopwords (palavras comuns):**
   ```python
   # Remove: a, the, is, was, etc.
   # Remove palavras com menos de 3 caracteres
   tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
   ```

8. **Lematização (reduzir à forma base):**
   ```python
   tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
   # "movies" → "movie", "running" → "run"
   ```

9. **Juntar tokens em texto:**
   ```python
   return ' '.join(tokens)
   ```

**Exemplo:**
```python
# Entrada
text = "This movie was AMAZING! Best film I've seen in years!! 😍"

# Saída após processamento
"movie amazing best film seen year"
```

---

### `load_data(self, sample_size=None, save_processed=True)`

**Descrição:** Carrega o dataset IMDb, pré-processa os textos e opcionalmente salva em CSV.

**Parâmetros:**
- `sample_size` (int, opcional): 
  - Se `None`: usa todo o dataset (50.000 exemplos)
  - Se número: usa amostra aleatória (ex: 5000)
- `save_processed` (bool): Se `True`, salva dados processados em `/data/`

**Retorna:** Tupla com 4 elementos:
```python
(X_train, y_train, X_test, y_test)
# X_train: textos de treino processados
# y_train: labels de treino (0=negativo, 1=positivo)
# X_test: textos de teste processados
# y_test: labels de teste
```

**O que faz:**

1. **Carrega dataset da Hugging Face:**
   ```python
   dataset = load_dataset('imdb')  # 25k treino + 25k teste
   ```

2. **Converte para DataFrame do pandas:**
   - Facilita manipulação dos dados
   - Permite uso de métodos como `.sample()` e `.apply()`

3. **Aplica amostragem (se solicitado):**
   ```python
   train_df = train_df.sample(n=sample_size, random_state=42)
   # random_state=42 garante reprodutibilidade
   ```

4. **Pré-processa todos os textos:**
   ```python
   train_df['processed_text'] = train_df['text'].apply(self.preprocess_text)
   # Aplica preprocess_text() em cada review
   ```

5. **Salva dados processados (se solicitado):**
   - Chama `_save_processed_data()` que cria 3 arquivos CSV

6. **Retorna dados prontos para treinamento**

**Por que usar sample_size?**
- Dataset completo demora ~10-15 minutos
- Amostra de 5000 exemplos: ~1-2 minutos
- Útil para testes rápidos e desenvolvimento

---

### `_save_processed_data(self, train_df, test_df)`

**Descrição:** Salva os dados pré-processados e estatísticas em arquivos CSV na pasta `/data/`.

**Parâmetros:**
- `train_df` (DataFrame): DataFrame com dados de treino
- `test_df` (DataFrame): DataFrame com dados de teste

**Retorna:** Nenhum

**O que faz:**

1. **Obtém diretório de dados:**
   ```python
   data_dir = self._get_project_dirs()['data']
   data_dir.mkdir(exist_ok=True)  # Cria pasta se não existir
   ```

2. **Salva dados processados em CSV:**
   ```python
   # processed_train.csv - textos limpos + labels de treino
   # processed_test.csv - textos limpos + labels de teste
   ```

3. **Calcula estatísticas do dataset:**
   - Total de exemplos (treino/teste/total)
   - Distribuição de classes (positivos/negativos)
   - Tamanho médio dos textos (número de palavras)

4. **Salva estatísticas em CSV:**
   ```python
   # dataset_statistics.csv
   ```

5. **Exibe mensagens de confirmação**

**Arquivos gerados:**
- `processed_train.csv` - 10.000 linhas (se sample_size=10000)
- `processed_test.csv` - 10.000 linhas
- `dataset_statistics.csv` - 3 linhas (treino, teste, total)

---

### `train(self, X_train, y_train)`

**Descrição:** Treina o modelo de classificação usando os dados de treino.

**Parâmetros:**
- `X_train`: Textos de treino (processados)
- `y_train`: Labels correspondentes (0 ou 1)

**Retorna:** Nenhum (atualiza `self.model` e `self.vectorizer`)

**O que faz:**

1. **Cria e treina o TF-IDF Vectorizer:**
   ```python
   self.vectorizer = TfidfVectorizer(
       max_features=5000,      # Top 5000 palavras mais importantes
       min_df=2,               # Palavra deve aparecer em pelo menos 2 documentos
       max_df=0.8,             # Ignora palavras em mais de 80% dos documentos
       ngram_range=(1, 2)      # Considera palavras individuais e pares
   )
   ```

   **TF-IDF (Term Frequency-Inverse Document Frequency):**
   - Converte texto em números
   - Palavras mais raras e importantes recebem pesos maiores
   - Palavras muito comuns recebem pesos menores

   **ngram_range=(1,2) significa:**
   - Unigrams: "great", "movie"
   - Bigrams: "great movie", "bad acting"

2. **Transforma textos em vetores:**
   ```python
   X_train_tfidf = self.vectorizer.fit_transform(X_train)
   # Cada texto vira um vetor de 5000 números
   ```

3. **Cria e treina o modelo escolhido:**
   
   **Se Logistic Regression:**
   ```python
   self.model = LogisticRegression(
       max_iter=1000,    # Máximo 1000 iterações
       random_state=42,  # Reprodutibilidade
       n_jobs=-1         # Usa todos os CPUs disponíveis
   )
   ```
   
   **Se Naive Bayes:**
   ```python
   self.model = MultinomialNB()
   ```

4. **Ajusta o modelo aos dados:**
   ```python
   self.model.fit(X_train_tfidf, y_train)
   # Aprende os padrões de palavras positivas/negativas
   ```

**Por que Logistic Regression é padrão?**
- Melhor performance em textos
- Fornece probabilidades calibradas
- Permite ver quais palavras são mais importantes

---

### `evaluate(self, X_test, y_test)`

**Descrição:** Avalia o desempenho do modelo treinado usando dados de teste.

**Parâmetros:**
- `X_test`: Textos de teste (processados)
- `y_test`: Labels verdadeiros

**Retorna:** Dictionary com:
```python
{
    'accuracy': 0.8680,              # Acurácia geral
    'predictions': array([1,0,1...]), # Predições do modelo
    'confusion_matrix': array([[...]])  # Matriz de confusão
}
```

**O que faz:**

1. **Transforma textos de teste em vetores:**
   ```python
   X_test_tfidf = self.vectorizer.transform(X_test)
   # Usa o mesmo vetorizador do treino
   ```

2. **Faz predições:**
   ```python
   y_pred = self.model.predict(X_test_tfidf)
   ```

3. **Calcula acurácia:**
   ```python
   accuracy = accuracy_score(y_test, y_pred)
   # Percentual de acertos
   ```

4. **Gera relatório de classificação:**
   - **Precision (Precisão):** De todas as predições positivas, quantas estavam corretas?
   - **Recall (Revocação):** De todos os casos positivos reais, quantos foram identificados?
   - **F1-Score:** Média harmônica entre Precision e Recall

5. **Exibe resultados formatados:**
   ```
   RESULTADOS DA AVALIAÇÃO
   Acurácia: 0.8680 (86.80%)
   
                 precision  recall  f1-score
   Negativo       0.87      0.86      0.87
   Positivo       0.86      0.87      0.87
   ```

6. **Cria matriz de confusão:**
   ```
   [[VP  FN]    VP = Verdadeiros Positivos
    [FP  VN]]    VN = Verdadeiros Negativos
                 FP = Falsos Positivos
                 FN = Falsos Negativos
   ```

---

### `predict_sentiment(self, text)`

**Descrição:** Prediz o sentimento de um novo texto (review).

**Parâmetros:**
- `text` (str): Texto/review a ser analisado

**Retorna:** Tupla com:
```python
(sentiment, confidence)
# sentiment: "POSITIVO" ou "NEGATIVO"
# confidence: 0-100 (percentual de confiança)
```

**O que faz:**

1. **Verifica se modelo está treinado:**
   ```python
   if self.model is None or self.vectorizer is None:
       raise ValueError("Modelo não treinado")
   ```

2. **Pré-processa o texto:**
   ```python
   processed = self.preprocess_text(text)
   ```

3. **Vetoriza o texto:**
   ```python
   vectorized = self.vectorizer.transform([processed])
   ```

4. **Faz predição:**
   ```python
   prediction = self.model.predict(vectorized)[0]  # 0 ou 1
   ```

5. **Calcula probabilidades:**
   ```python
   probability = self.model.predict_proba(vectorized)[0]
   # Retorna [prob_negativo, prob_positivo]
   ```

6. **Formata resultado:**
   ```python
   sentiment = "POSITIVO" if prediction == 1 else "NEGATIVO"
   confidence = probability[prediction] * 100
   ```

**Exemplo de uso:**
```python
text = "This movie was absolutely amazing!"
sentiment, confidence = classifier.predict_sentiment(text)
print(f"{sentiment} ({confidence:.2f}%)")
# Saída: POSITIVO (95.23%)
```

---

### `save_model(self, model_path=None, vectorizer_path=None)`

**Descrição:** Salva o modelo treinado e o vetorizador em arquivos .pkl para uso futuro.

**Parâmetros:**
- `model_path` (str, opcional): Caminho customizado para salvar o modelo
- `vectorizer_path` (str, opcional): Caminho customizado para salvar o vetorizador

**Retorna:** Nenhum

**O que faz:**

1. **Obtém diretório de modelos:**
   ```python
   models_dir = self._get_project_dirs()['models']
   models_dir.mkdir(exist_ok=True)
   ```

2. **Define caminhos padrão (se não fornecidos):**
   ```python
   model_path = models_dir / 'sentiment_model.pkl'
   vectorizer_path = models_dir / 'tfidf_vectorizer.pkl'
   ```

3. **Serializa e salva o modelo:**
   ```python
   with open(model_path, 'wb') as f:
       pickle.dump(self.model, f)
   ```

4. **Serializa e salva o vetorizador:**
   ```python
   with open(vectorizer_path, 'wb') as f:
       pickle.dump(self.vectorizer, f)
   ```

5. **Exibe confirmação**

**Por que salvar o modelo?**
- Evita re-treinar (economiza tempo)
- Permite usar o modelo em outros scripts
- Útil para deploy em produção

**Formato .pkl:**
- Pickle = serialização Python
- Guarda objetos Python completos
- Pode ser carregado com `pickle.load()`

---

### `load_model(self, model_path=None, vectorizer_path=None)`

**Descrição:** Carrega um modelo e vetorizador previamente salvos.

**Parâmetros:**
- `model_path` (str, opcional): Caminho do modelo salvo
- `vectorizer_path` (str, opcional): Caminho do vetorizador salvo

**Retorna:** Nenhum (atualiza `self.model` e `self.vectorizer`)

**O que faz:**

1. **Obtém diretório de modelos:**
   ```python
   models_dir = self._get_project_dirs()['models']
   ```

2. **Define caminhos padrão (se não fornecidos):**
   ```python
   model_path = models_dir / 'sentiment_model.pkl'
   vectorizer_path = models_dir / 'tfidf_vectorizer.pkl'
   ```

3. **Carrega o modelo:**
   ```python
   with open(model_path, 'rb') as f:
       self.model = pickle.load(f)
   ```

4. **Carrega o vetorizador:**
   ```python
   with open(vectorizer_path, 'rb') as f:
       self.vectorizer = pickle.load(f)
   ```

5. **Exibe confirmação**

**Uso típico:**
```python
# Criar instância
classifier = SentimentClassifier()

# Carregar modelo salvo (sem treinar)
classifier.load_model()

# Usar imediatamente
sentiment, conf = classifier.predict_sentiment("Great movie!")
```

---

## Função generate_visualizations

### `generate_visualizations(classifier, results, X_test, y_test)`

**Descrição:** Gera uma visualização completa com 6 gráficos mostrando o desempenho do modelo.

**Parâmetros:**
- `classifier`: Instância do SentimentClassifier treinado
- `results`: Dictionary retornado por `evaluate()` com métricas
- `X_test`: Textos de teste
- `y_test`: Labels de teste verdadeiros

**Retorna:** Nenhum (salva imagem em `/visualizations/model_analysis.png`)

**O que faz:**

### 1. Setup Inicial
```python
# Obtém diretório e cria se não existir
viz_dir = SentimentClassifier._get_project_dirs()['visualizations']
viz_dir.mkdir(exist_ok=True)

# Configura estilo dos gráficos
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Cria figura com 6 subplots (2 linhas x 3 colunas)
fig = plt.figure(figsize=(16, 10))
```

### 2. Gráfico 1: Matriz de Confusão (Canto Superior Esquerdo)
```python
ax1 = plt.subplot(2, 3, 1)
```

**O que mostra:**
- Heatmap 2x2 com cores azuis
- Linha = Valor Real, Coluna = Valor Predito
- Células mostram quantidades:
  - [0,0] = Negativos classificados como Negativos ✓
  - [0,1] = Negativos classificados como Positivos ✗
  - [1,0] = Positivos classificados como Negativos ✗
  - [1,1] = Positivos classificados como Positivos ✓

**Interpretação:**
- Diagonal principal alta = bom modelo
- Células fora da diagonal = erros

### 3. Gráfico 2: Métricas por Classe (Centro Superior)
```python
ax2 = plt.subplot(2, 3, 2)
```

**O que mostra:**
- Gráfico de barras agrupadas
- 3 métricas × 2 classes = 6 barras
- Vermelho (#ff6b6b) = Classe Negativa
- Azul-verde (#4ecdc4) = Classe Positiva
- Valores de 0 a 1 no eixo Y

**Métricas exibidas:**
- **Precision:** Acertos / (Acertos + Falsos Positivos)
- **Recall:** Acertos / (Acertos + Falsos Negativos)
- **F1-Score:** 2 × (Precision × Recall) / (Precision + Recall)

**Valores aparecem acima de cada barra**

### 4. Gráfico 3: Acurácia Geral (Canto Superior Direito)
```python
ax3 = plt.subplot(2, 3, 3)
```

**O que mostra:**
- Número gigante verde (#2ecc71) centralizado
- Mostra acurácia em percentual (ex: 86.80%)
- Texto "Acurácia Geral" abaixo
- Sem eixos (tipo "card")

**Por que destacar?**
- Métrica mais importante
- Fácil visualização rápida

### 5. Gráfico 4: Top 15 Palavras Positivas (Canto Inferior Esquerdo)
```python
ax4 = plt.subplot(2, 3, 4)
```

**O que mostra:**
- Barras horizontais em tons de verde
- Palavras que mais indicam sentimento POSITIVO
- Eixo X = Peso do coeficiente do modelo
- Quanto maior o peso, mais positiva a palavra

**Exemplos típicos:**
- "excellent", "great", "best"
- "amazing", "wonderful", "perfect"
- "loved", "favorite", "recommend"

**Como funciona:**
```python
coef = classifier.model.coef_[0]  # Pesos de todas as palavras
top_indices = np.argsort(coef)[-15:]  # 15 maiores pesos
```

### 6. Gráfico 5: Top 15 Palavras Negativas (Centro Inferior)
```python
ax5 = plt.subplot(2, 3, 5)
```

**O que mostra:**
- Barras horizontais em tons de vermelho
- Palavras que mais indicam sentimento NEGATIVO
- Eixo X = Peso do coeficiente (valores negativos)
- Quanto mais negativo o peso, mais negativa a palavra

**Exemplos típicos:**
- "worst", "terrible", "awful"
- "boring", "waste", "disappointing"
- "bad", "poor", "horrible"

**Como funciona:**
```python
top_indices = np.argsort(coef)[:15]  # 15 menores pesos
```

### 7. Gráfico 6: Distribuição de Predições (Canto Inferior Direito)
```python
ax6 = plt.subplot(2, 3, 6)
```

**O que mostra:**
- Gráfico de barras simples
- 2 barras: quantidade de predições Negativas e Positivas
- Cores: vermelho (negativo) e azul-verde (positivo)
- Valores aparecem no topo de cada barra

**Por que é útil:**
- Verifica se modelo está balanceado
- Detecta viés (ex: prediz tudo como positivo)
- Ideal: ~50/50 se dataset é balanceado

### 8. Finalização
```python
plt.tight_layout()  # Ajusta espaçamento automático

# Salva em alta resolução
output_path = viz_dir / 'model_analysis.png'
plt.savefig(str(output_path), dpi=300, bbox_inches='tight')

plt.close()  # Libera memória
```

**Parâmetros de salvamento:**
- `dpi=300` = Alta qualidade (300 pontos por polegada)
- `bbox_inches='tight'` = Remove espaços em branco extras

---

## Função main

### `main()`

**Descrição:** Função principal que orquestra todo o pipeline de treinamento, avaliação e teste.

**Parâmetros:** Nenhum

**Retorna:** Nenhum

**O que faz (fluxo completo):**

### 1. Exibe Header
```python
print("="*70)
print("CLASSIFICAÇÃO DE SENTIMENTOS EM AVALIAÇÕES DE FILMES")
print("="*70)
```

### 2. Cria Instância do Classificador
```python
classifier = SentimentClassifier(
    model_type='logistic_regression',  # Pode alterar para 'naive_bayes'
    max_features=5000                   # Pode aumentar para 10000+
)
```

### 3. Carrega e Processa Dados
```python
X_train, y_train, X_test, y_test = classifier.load_data(sample_size=1000)
```

**Configurações atuais:**
- `sample_size=1000` = Usa 1000 exemplos de treino e teste
- `save_processed=True` (padrão) = Salva CSVs em `/data/`

**Para dataset completo:**
```python
classifier.load_data(sample_size=None)  # 25.000 treino + 25.000 teste
```

### 4. Treina o Modelo
```python
classifier.train(X_train, y_train)
```

**Processos internos:**
- Cria vetorizador TF-IDF
- Transforma textos em números
- Treina modelo de Machine Learning
- Aprende padrões de palavras positivas/negativas

### 5. Avalia Desempenho
```python
results = classifier.evaluate(X_test, y_test)
```

**Exibe no console:**
- Acurácia geral
- Precision, Recall, F1-Score por classe
- Relatório completo

### 6. Salva Modelo
```python
classifier.save_model()
```

**Arquivos criados:**
- `/models/sentiment_model.pkl` (~5 MB)
- `/models/tfidf_vectorizer.pkl` (~15 MB)

### 7. Gera Visualizações
```python
generate_visualizations(classifier, results, X_test, y_test)
```

**Arquivo criado:**
- `/visualizations/model_analysis.png` (imagem com 6 gráficos)

### 8. Testa com Exemplos
```python
test_reviews = [
    "This movie was absolutely amazing! The acting was superb and the plot was engaging.",
    "Terrible film. Waste of time and money. I couldn't even finish watching it.",
    "It was okay, nothing special but not terrible either.",
    "Best movie I've seen in years! Highly recommended!",
    "Boring and predictable. The worst movie of the year."
]

for review in test_reviews:
    sentiment, confidence = classifier.predict_sentiment(review)
    print(f"Sentimento: {sentiment} (Confiança: {confidence:.2f}%)")
```

**Saída esperada:**
```
Review 1: This movie was absolutely amazing!...
Sentimento: POSITIVO (Confiança: 95.23%)
----------------------------------------------------------------------

Review 2: Terrible film. Waste of time and money...
Sentimento: NEGATIVO (Confiança: 98.54%)
----------------------------------------------------------------------
...
```

### 9. Finaliza
```python
print("PROCESSO CONCLUÍDO COM SUCESSO!")
```

---

## 🔧 Configurações e Customizações

### Como ajustar o tamanho do dataset?

No método `main()`, linha:
```python
X_train, y_train, X_test, y_test = classifier.load_data(sample_size=1000)
```

**Opções:**
- `sample_size=1000` - Rápido (1-2 min), ~78-82% acurácia
- `sample_size=5000` - Médio (3-5 min), ~84-86% acurácia
- `sample_size=10000` - Bom (5-8 min), ~86-88% acurácia
- `sample_size=None` - Completo (10-15 min), ~88-90% acurácia

### Como mudar o número de features?

No construtor:
```python
classifier = SentimentClassifier(
    model_type='logistic_regression',
    max_features=10000  # Aumentar para capturar mais palavras
)
```

**Impacto:**
- Mais features = Modelo mais preciso (até certo ponto)
- Mais features = Mais lento e usa mais memória
- Recomendado: 5000-15000 para este dataset

### Como trocar o algoritmo?

```python
# Opção 1: Logistic Regression (padrão, recomendado)
classifier = SentimentClassifier(model_type='logistic_regression')

# Opção 2: Naive Bayes (mais rápido, menos preciso)
classifier = SentimentClassifier(model_type='naive_bayes')
```

**Comparação:**
| Algoritmo | Velocidade | Acurácia | Interpretabilidade |
|-----------|------------|----------|-------------------|
| Logistic Regression | Média | Alta (86-88%) | Alta (pesos das palavras) |
| Naive Bayes | Rápida | Média (82-85%) | Média |

---

## 📊 Métricas de Avaliação Explicadas

### Acurácia (Accuracy)
```
Acurácia = (Acertos) / (Total)
```
- Percentual geral de acertos
- **Problema:** Pode enganar se classes desbalanceadas

### Precision (Precisão)
```
Precision = VP / (VP + FP)
```
- De todas as predições POSITIVAS, quantas estavam corretas?
- **Alta precision:** Poucas classificações erradas como positivo

### Recall (Revocação)
```
Recall = VP / (VP + FN)
```
- De todos os casos POSITIVOS reais, quantos foram identificados?
- **Alto recall:** Pegou a maioria dos positivos

### F1-Score
```
F1 = 2 × (Precision × Recall) / (Precision + Recall)
```
- Média harmônica entre Precision e Recall
- **Melhor métrica** quando classes desbalanceadas

### Matriz de Confusão

```
                 Predito
              Neg    Pos
Real  Neg  [  VP  |  FN  ]
      Pos  [  FP  |  VN  ]
```

- **VP (Verdadeiros Positivos):** Acertou o positivo ✓
- **VN (Verdadeiros Negativos):** Acertou o negativo ✓
- **FP (Falsos Positivos):** Erro - disse positivo mas era negativo ✗
- **FN (Falsos Negativos):** Erro - disse negativo mas era positivo ✗

---

## 🎯 Fluxo Completo de Execução

```
1. Importar bibliotecas
   ↓
2. Criar SentimentClassifier
   ↓
3. Download recursos NLTK (automático)
   ↓
4. Carregar dataset IMDb (Hugging Face)
   ↓
5. Pré-processar textos
   ├─ Limpar HTML
   ├─ Lowercase
   ├─ Tokenizar
   ├─ Remover stopwords
   └─ Lematizar
   ↓
6. Salvar dados processados (.csv)
   ↓
7. Criar vetores TF-IDF
   ↓
8. Treinar modelo (Logistic Regression)
   ↓
9. Avaliar no conjunto de teste
   ↓
10. Salvar modelo (.pkl)
    ↓
11. Gerar visualizações (.png)
    ↓
12. Testar com novos exemplos
    ↓
13. Concluir ✓
```

---

## 🚀 Exemplos de Uso

### Uso Básico (Treinar e Salvar)
```python
# Criar e treinar
classifier = SentimentClassifier()
X_train, y_train, X_test, y_test = classifier.load_data(sample_size=5000)
classifier.train(X_train, y_train)
results = classifier.evaluate(X_test, y_test)
classifier.save_model()
```

### Carregar Modelo Existente
```python
# Criar instância
classifier = SentimentClassifier()

# Carregar modelo salvo (sem treinar)
classifier.load_model()

# Usar imediatamente
text = "This movie is incredible!"
sentiment, confidence = classifier.predict_sentiment(text)
print(f"{sentiment}: {confidence:.2f}%")
```

### Processar Múltiplas Reviews
```python
reviews = [
    "Amazing cinematography and acting!",
    "Worst movie ever, don't waste your time",
    "It's okay, nothing special"
]

for review in reviews:
    sentiment, conf = classifier.predict_sentiment(review)
    print(f"{review[:30]}... → {sentiment} ({conf:.1f}%)")
```

### Análise de Performance Customizada
```python
classifier = SentimentClassifier(max_features=10000)
X_train, y_train, X_test, y_test = classifier.load_data(sample_size=None)
classifier.train(X_train, y_train)

results = classifier.evaluate(X_test, y_test)
print(f"Acurácia: {results['accuracy']*100:.2f}%")
print(f"Total erros: {(results['predictions'] != y_test).sum()}")
```

---

## 📝 Notas Importantes

### Performance Esperada
- **Sample 1000:** ~78-82% acurácia, 1-2 minutos
- **Sample 5000:** ~84-86% acurácia, 3-5 minutos
- **Sample 10000:** ~86-88% acurácia, 5-8 minutos
- **Dataset completo:** ~88-90% acurácia, 10-15 minutos

### Limitações
1. **Apenas inglês:** Modelo treinado em reviews em inglês
2. **Binário:** Apenas positivo/negativo (sem neutro)
3. **Contexto:** Não entende sarcasmo ou ironia complexa
4. **Domínio:** Otimizado para reviews de filmes

### Possíveis Melhorias
1. **Aumentar max_features** para 10000-15000
2. **Usar dataset completo** (sample_size=None)
3. **Adicionar bigramas** (já implementado com ngram_range=(1,2))
4. **Experimentar outros modelos** (SVM, Random Forest, Deep Learning)
5. **Ajustar hiperparâmetros** do TF-IDF

---

## 🔍 Troubleshooting

### Erro: "Modelo não treinado"
**Solução:** Execute `train()` antes de `predict_sentiment()`

### Baixa acurácia (<75%)
**Possíveis causas:**
- sample_size muito pequeno
- max_features muito baixo
- Dados de teste diferentes do treino

### Script muito lento
**Soluções:**
- Reduzir sample_size
- Reduzir max_features
- Usar Naive Bayes (mais rápido)

### Erro ao salvar/carregar modelo
**Verificar:**
- Pasta `/models/` existe?
- Permissões de escrita
- Espaço em disco

---

**Documentação criada para:** Projeto de Classificação de Sentimentos  
**Versão:** 1.0  
**Data:** 18 de Novembro de 2025
