# Dados

Esta pasta armazena os dados processados e estatísticas do dataset IMDb.

## Fonte dos Dados

O projeto utiliza o **IMDb Movie Reviews Dataset** disponível através da biblioteca `datasets` da Hugging Face:
- 50.000 reviews de filmes (25.000 positivas + 25.000 negativas)
- Dataset baixado automaticamente ao executar o script

## Arquivos Gerados

Ao executar `src/sentiment_classifier.py`, os seguintes arquivos são criados automaticamente:

### 📊 Dados Processados
- **processed_train.csv** - Dados de treino pré-processados (textos limpos + labels)
- **processed_test.csv** - Dados de teste pré-processados (textos limpos + labels)

### 📈 Estatísticas
- **dataset_statistics.csv** - Estatísticas do dataset incluindo:
  - Total de exemplos (treino/teste)
  - Distribuição de classes (positivos/negativos)
  - Tamanho médio dos textos processados

## Pré-processamento Aplicado

Os textos passam pelas seguintes etapas:
1. Remoção de HTML tags
2. Conversão para minúsculas
3. Tokenização
4. Remoção de stopwords (inglês)
5. Lematização (WordNet)

## Cache Original

O dataset original é automaticamente cacheado pelo Hugging Face em:
- Windows: `C:\Users\<usuario>\.cache\huggingface\datasets`

## Uso dos Dados Salvos

Os arquivos CSV podem ser utilizados para:
- Análise exploratória adicional
- Treinamento offline sem reprocessamento
- Experimentação com outros modelos
- Validação do pré-processamento

> **Nota:** Arquivos de dados (*.csv, *.txt, *.json) são ignorados pelo git conforme configurado no `.gitignore`.
