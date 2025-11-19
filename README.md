# Classificação de Sentimentos em Avaliações de Filmes

## Descrição do Projeto

Este projeto implementa um modelo de Processamento de Linguagem Natural (NLP) para classificação de sentimentos em avaliações de filmes do IMDb. Utilizando técnicas de aprendizado supervisionado, o modelo analisa textos de reviews e determina se a avaliação é positiva ou negativa. O projeto emprega algoritmos como Regressão Logística e Naive Bayes, com vetorização TF-IDF para representação textual. A base de dados IMDb contém 50.000 avaliações balanceadas, permitindo treinamento robusto e validação eficaz do modelo de classificação binária.

## Estrutura do Projeto

```
projeto-inteligencia-computacional/
│
├── data/                    # Diretório para datasets
├── models/                  # Modelos treinados salvos
├── notebooks/               # Jupyter notebooks
│   └── sentiment_analysis.ipynb
├── src/                     # Código fonte
│   └── sentiment_classifier.py
├── requirements.txt         # Dependências do projeto
├── README.md               # Este arquivo
└── LICENSE                 # Licença do projeto
```

## Como Executar o Projeto

### Pré-requisitos

- Python 3.8 ou superior
- pip (gerenciador de pacotes Python)

### Instalação

1. Clone o repositório:
```bash
git clone https://github.com/jadilsonl12/projeto-inteligencia-computacional.git
cd projeto-inteligencia-computacional
```

2. Crie um ambiente virtual (recomendado):
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. Instale as dependências:
```bash
pip install -r requirements.txt
```

### Executando o Notebook

```bash
jupyter notebook notebooks/sentiment_analysis.ipynb
```

### Executando o Script Python

```bash
python src/sentiment_classifier.py
```

## Tecnologias Utilizadas

- **Python 3.8+**: Linguagem principal
- **scikit-learn**: Algoritmos de machine learning
- **pandas**: Manipulação de dados
- **numpy**: Operações numéricas
- **nltk**: Processamento de linguagem natural
- **matplotlib/seaborn**: Visualização de dados

## Resultados

O modelo alcança aproximadamente 85-90% de acurácia na classificação de sentimentos, demonstrando boa performance na distinção entre avaliações positivas e negativas.

## Dataset

O projeto utiliza o IMDb Movie Reviews Dataset, disponível publicamente através da biblioteca datasets do Hugging Face ou diretamente do Kaggle.

## Licença

Este projeto está licenciado sob a Licença MIT - veja o arquivo [LICENSE](LICENSE) para detalhes.

## Autor

Desenvolvido por Jadilson Lucio dos Santos, Willy Lourenço da Silva e João Vinícuis Lima de Oliveira como projeto da disciplina de Inteligência Computacional.

## Contribuições

Contribuições são bem-vindas! Sinta-se à vontade para abrir issues ou enviar pull requests.
