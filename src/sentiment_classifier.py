"""
Classificação de Sentimentos em Avaliações de Filmes
Modelo de NLP para identificar se uma avaliação é positiva ou negativa
Dataset: IMDb Movie Reviews
"""

import numpy as np
import pandas as pd
import pickle
import re
from html import unescape
import warnings
warnings.filterwarnings('ignore')

# Visualização
import matplotlib
matplotlib.use('Agg')  # Usar backend sem interface gráfica
import matplotlib.pyplot as plt
import seaborn as sns

# Scikit-learn
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# NLTK
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer

# Datasets
from datasets import load_dataset


class SentimentClassifier:
    """
    Classificador de sentimentos para avaliações de filmes
    """
    
    @staticmethod
    def _get_project_dirs():
        """Retorna os diretórios do projeto"""
        from pathlib import Path
        script_dir = Path(__file__).parent.absolute()
        project_dir = script_dir.parent
        return {
            'models': project_dir / 'models',
            'data': project_dir / 'data',
            'visualizations': project_dir / 'visualizations'
        }
    
    def __init__(self, model_type='logistic_regression', max_features=5000):
        """
        Inicializa o classificador
        
        Args:
            model_type (str): 'logistic_regression' ou 'naive_bayes'
            max_features (int): Número máximo de features para TF-IDF
        """
        self.model_type = model_type
        self.max_features = max_features
        self.vectorizer = None
        self.model = None
        self.lemmatizer = WordNetLemmatizer()
        self._download_nltk_resources()
        
    def _download_nltk_resources(self):
        """Download de recursos necessários do NLTK"""
        resources = ['punkt', 'punkt_tab', 'stopwords', 'wordnet', 'omw-1.4']
        for resource in resources:
            try:
                # Tentar encontrar o recurso
                if resource == 'punkt_tab':
                    nltk.data.find(f'tokenizers/{resource}/')
                elif resource in ['punkt', 'stopwords']:
                    nltk.data.find(f'tokenizers/{resource}')
                elif resource in ['wordnet', 'omw-1.4']:
                    nltk.data.find(f'corpora/{resource}')
            except LookupError:
                print(f"Baixando recurso NLTK: {resource}...")
                nltk.download(resource, quiet=True)
    
    def preprocess_text(self, text):
        """
        Pré-processa o texto
        
        Args:
            text (str): Texto a ser processado
            
        Returns:
            str: Texto processado
        """
        # Decodificar entidades HTML
        text = unescape(text)
        
        # Remover HTML tags
        text = re.sub(r'<.*?>', '', text)
        
        # Converter para minúsculas
        text = text.lower()
        
        # Remover URLs
        text = re.sub(r'http\S+|www\S+', '', text)
        
        # Remover caracteres especiais e números
        text = re.sub(r'[^a-z\s]', '', text)
        
        # Tokenização
        tokens = word_tokenize(text)
        
        # Remover stopwords
        stop_words = set(stopwords.words('english'))
        tokens = [word for word in tokens if word not in stop_words and len(word) > 2]
        
        # Lematização
        tokens = [self.lemmatizer.lemmatize(word) for word in tokens]
        
        return ' '.join(tokens)
    
    def load_data(self, sample_size=None, save_processed=True):
        """
        Carrega o dataset IMDb
        
        Args:
            sample_size (int): Tamanho da amostra (None para dataset completo)
            save_processed (bool): Se True, salva os dados processados em CSV
            
        Returns:
            tuple: (X_train, y_train, X_test, y_test)
        """
        print("Carregando dataset IMDb...")
        dataset = load_dataset('imdb')
        
        # Converter para DataFrame
        train_df = pd.DataFrame(dataset['train'])
        test_df = pd.DataFrame(dataset['test'])
        
        if sample_size:
            train_df = train_df.sample(n=sample_size, random_state=42)
            test_df = test_df.sample(n=sample_size, random_state=42)
            print(f"Usando amostra de {sample_size} exemplos")
        
        print(f"Dataset carregado: {len(train_df)} treino, {len(test_df)} teste")
        
        # Pré-processar textos
        print("Pré-processando textos...")
        train_df['processed_text'] = train_df['text'].apply(self.preprocess_text)
        test_df['processed_text'] = test_df['text'].apply(self.preprocess_text)
        
        # Salvar dados processados se solicitado
        if save_processed:
            self._save_processed_data(train_df, test_df)
        
        return (
            train_df['processed_text'],
            train_df['label'],
            test_df['processed_text'],
            test_df['label']
        )
    
    def _save_processed_data(self, train_df, test_df):
        """
        Salva os dados processados e estatísticas em arquivos CSV
        
        Args:
            train_df: DataFrame de treino
            test_df: DataFrame de teste
        """
        dirs = self._get_project_dirs()
        data_dir = dirs['data']
        data_dir.mkdir(exist_ok=True)
        
        # Salvar dados processados
        train_path = data_dir / 'processed_train.csv'
        test_path = data_dir / 'processed_test.csv'
        
        train_df[['processed_text', 'label']].to_csv(train_path, index=False)
        test_df[['processed_text', 'label']].to_csv(test_path, index=False)
        
        # Criar estatísticas do dataset
        stats = {
            'Dataset': ['Treino', 'Teste', 'Total'],
            'Total_Exemplos': [len(train_df), len(test_df), len(train_df) + len(test_df)],
            'Positivos': [
                (train_df['label'] == 1).sum(),
                (test_df['label'] == 1).sum(),
                (train_df['label'] == 1).sum() + (test_df['label'] == 1).sum()
            ],
            'Negativos': [
                (train_df['label'] == 0).sum(),
                (test_df['label'] == 0).sum(),
                (train_df['label'] == 0).sum() + (test_df['label'] == 0).sum()
            ],
            'Tamanho_Medio_Texto': [
                train_df['processed_text'].str.split().str.len().mean().round(2),
                test_df['processed_text'].str.split().str.len().mean().round(2),
                pd.concat([train_df['processed_text'], test_df['processed_text']]).str.split().str.len().mean().round(2)
            ]
        }
        
        stats_df = pd.DataFrame(stats)
        stats_path = data_dir / 'dataset_statistics.csv'
        stats_df.to_csv(stats_path, index=False)
        
        print(f"\n✓ Dados processados salvos em: {data_dir}")
        print(f"  - {train_path.name}")
        print(f"  - {test_path.name}")
        print(f"  - {stats_path.name}")
    
    def train(self, X_train, y_train):
        """
        Treina o modelo
        
        Args:
            X_train: Textos de treino
            y_train: Labels de treino
        """
        print("\nCriando vetores TF-IDF...")
        self.vectorizer = TfidfVectorizer(
            max_features=self.max_features,
            min_df=2,
            max_df=0.8,
            ngram_range=(1, 2)
        )
        
        X_train_tfidf = self.vectorizer.fit_transform(X_train)
        print(f"Shape dos dados de treino: {X_train_tfidf.shape}")
        
        print(f"\nTreinando modelo ({self.model_type})...")
        if self.model_type == 'logistic_regression':
            self.model = LogisticRegression(max_iter=1000, random_state=42, n_jobs=-1)
        elif self.model_type == 'naive_bayes':
            self.model = MultinomialNB()
        else:
            raise ValueError("model_type deve ser 'logistic_regression' ou 'naive_bayes'")
        
        self.model.fit(X_train_tfidf, y_train)
        print("✓ Modelo treinado com sucesso!")
    
    def evaluate(self, X_test, y_test):
        """
        Avalia o modelo
        
        Args:
            X_test: Textos de teste
            y_test: Labels de teste
            
        Returns:
            dict: Métricas de avaliação
        """
        print("\nAvaliando modelo...")
        X_test_tfidf = self.vectorizer.transform(X_test)
        y_pred = self.model.predict(X_test_tfidf)
        
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n{'='*60}")
        print(f"RESULTADOS DA AVALIAÇÃO")
        print(f"{'='*60}")
        print(f"Acurácia: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Negativo', 'Positivo'])}")
        
        return {
            'accuracy': accuracy,
            'predictions': y_pred,
            'confusion_matrix': confusion_matrix(y_test, y_pred)
        }
    
    def predict_sentiment(self, text):
        """
        Prediz o sentimento de um texto
        
        Args:
            text (str): Texto a ser analisado
            
        Returns:
            tuple: (sentimento, confiança)
        """
        if self.model is None or self.vectorizer is None:
            raise ValueError("Modelo não treinado. Execute train() primeiro.")
        
        # Pré-processar
        processed = self.preprocess_text(text)
        
        # Vetorizar
        vectorized = self.vectorizer.transform([processed])
        
        # Predizer
        prediction = self.model.predict(vectorized)[0]
        probability = self.model.predict_proba(vectorized)[0]
        
        sentiment = "POSITIVO" if prediction == 1 else "NEGATIVO"
        confidence = probability[prediction] * 100
        
        return sentiment, confidence
    
    def save_model(self, model_path=None, vectorizer_path=None):
        """
        Salva o modelo e o vetorizador
        
        Args:
            model_path (str): Caminho para salvar o modelo
            vectorizer_path (str): Caminho para salvar o vetorizador
        """
        dirs = self._get_project_dirs()
        models_dir = dirs['models']
        models_dir.mkdir(exist_ok=True)
        
        if model_path is None:
            model_path = models_dir / 'sentiment_model.pkl'
        if vectorizer_path is None:
            vectorizer_path = models_dir / 'tfidf_vectorizer.pkl'
        
        with open(model_path, 'wb') as f:
            pickle.dump(self.model, f)
        
        with open(vectorizer_path, 'wb') as f:
            pickle.dump(self.vectorizer, f)
        
        print(f"\n✓ Modelo salvo em: {model_path}")
        print(f"✓ Vetorizador salvo em: {vectorizer_path}")
    
    def load_model(self, model_path=None, vectorizer_path=None):
        """
        Carrega um modelo e vetorizador salvos
        
        Args:
            model_path (str): Caminho do modelo salvo
            vectorizer_path (str): Caminho do vetorizador salvo
        """
        dirs = self._get_project_dirs()
        models_dir = dirs['models']
        
        if model_path is None:
            model_path = models_dir / 'sentiment_model.pkl'
        if vectorizer_path is None:
            vectorizer_path = models_dir / 'tfidf_vectorizer.pkl'
        
        with open(model_path, 'rb') as f:
            self.model = pickle.load(f)
        
        with open(vectorizer_path, 'rb') as f:
            self.vectorizer = pickle.load(f)
        
        print(f"\n✓ Modelo carregado de: {model_path}")
        print(f"✓ Vetorizador carregado de: {vectorizer_path}")


def generate_visualizations(classifier, results, X_test, y_test):
    """
    Gera visualizações dos resultados do modelo
    """
    # Obter diretório de visualizações
    dirs = SentimentClassifier._get_project_dirs()
    viz_dir = dirs['visualizations']
    viz_dir.mkdir(exist_ok=True)
    
    # Configurar estilo
    plt.style.use('seaborn-v0_8-darkgrid')
    sns.set_palette("husl")
    
    # Criar figura com subplots
    fig = plt.figure(figsize=(16, 10))
    
    # 1. Matriz de Confusão
    ax1 = plt.subplot(2, 3, 1)
    cm = results['confusion_matrix']
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Negativo', 'Positivo'],
                yticklabels=['Negativo', 'Positivo'])
    ax1.set_title('Matriz de Confusão', fontsize=14, fontweight='bold')
    ax1.set_ylabel('Valor Real', fontsize=12)
    ax1.set_xlabel('Valor Predito', fontsize=12)
    
    # 2. Métricas de Performance
    ax2 = plt.subplot(2, 3, 2)
    accuracy = results['accuracy']
    
    # Calcular precision, recall, f1-score para cada classe
    from sklearn.metrics import precision_recall_fscore_support
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, results['predictions'], average=None)
    
    metrics_data = {
        'Negativo': [precision[0], recall[0], f1[0]],
        'Positivo': [precision[1], recall[1], f1[1]]
    }
    
    x = np.arange(3)
    width = 0.35
    
    bars1 = ax2.bar(x - width/2, metrics_data['Negativo'], width, label='Negativo', color='#ff6b6b', alpha=0.8)
    bars2 = ax2.bar(x + width/2, metrics_data['Positivo'], width, label='Positivo', color='#4ecdc4', alpha=0.8)
    
    ax2.set_ylabel('Score', fontsize=12)
    ax2.set_title('Métricas por Classe', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(['Precision', 'Recall', 'F1-Score'])
    ax2.legend()
    ax2.set_ylim(0, 1)
    ax2.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width()/2., height,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    # 3. Acurácia Geral
    ax3 = plt.subplot(2, 3, 3)
    ax3.text(0.5, 0.5, f'{accuracy*100:.2f}%', 
             ha='center', va='center', fontsize=60, fontweight='bold',
             color='#2ecc71')
    ax3.text(0.5, 0.2, 'Acurácia Geral', 
             ha='center', va='center', fontsize=16, color='gray')
    ax3.axis('off')
    ax3.set_xlim(0, 1)
    ax3.set_ylim(0, 1)
    
    # 4. Top 15 Features Positivas
    ax4 = plt.subplot(2, 3, 4)
    if hasattr(classifier.model, 'coef_'):
        feature_names = np.array(classifier.vectorizer.get_feature_names_out())
        coef = classifier.model.coef_[0]
        
        top_positive_indices = np.argsort(coef)[-15:]
        top_positive_words = feature_names[top_positive_indices]
        top_positive_scores = coef[top_positive_indices]
        
        colors_pos = plt.cm.Greens(np.linspace(0.4, 0.8, len(top_positive_words)))
        ax4.barh(range(len(top_positive_words)), top_positive_scores, color=colors_pos)
        ax4.set_yticks(range(len(top_positive_words)))
        ax4.set_yticklabels(top_positive_words)
        ax4.set_xlabel('Peso (Coeficiente)', fontsize=11)
        ax4.set_title('Top 15 Palavras POSITIVAS', fontsize=13, fontweight='bold')
        ax4.grid(axis='x', alpha=0.3)
    
    # 5. Top 15 Features Negativas
    ax5 = plt.subplot(2, 3, 5)
    if hasattr(classifier.model, 'coef_'):
        top_negative_indices = np.argsort(coef)[:15]
        top_negative_words = feature_names[top_negative_indices]
        top_negative_scores = coef[top_negative_indices]
        
        colors_neg = plt.cm.Reds(np.linspace(0.4, 0.8, len(top_negative_words)))
        ax5.barh(range(len(top_negative_words)), top_negative_scores, color=colors_neg)
        ax5.set_yticks(range(len(top_negative_words)))
        ax5.set_yticklabels(top_negative_words)
        ax5.set_xlabel('Peso (Coeficiente)', fontsize=11)
        ax5.set_title('Top 15 Palavras NEGATIVAS', fontsize=13, fontweight='bold')
        ax5.grid(axis='x', alpha=0.3)
    
    # 6. Distribuição de Predições
    ax6 = plt.subplot(2, 3, 6)
    predictions_counts = pd.Series(results['predictions']).value_counts().sort_index()
    colors_dist = ['#ff6b6b', '#4ecdc4']
    bars = ax6.bar(['Negativo', 'Positivo'], predictions_counts.values, 
                   color=colors_dist, alpha=0.8, edgecolor='black', linewidth=2)
    ax6.set_ylabel('Quantidade', fontsize=12)
    ax6.set_title('Distribuição das Predições', fontsize=14, fontweight='bold')
    ax6.grid(axis='y', alpha=0.3)
    
    # Adicionar valores nas barras
    for bar in bars:
        height = bar.get_height()
        ax6.text(bar.get_x() + bar.get_width()/2., height,
                f'{int(height)}', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    
    # Salvar figura
    output_path = viz_dir / 'model_analysis.png'
    plt.savefig(str(output_path), dpi=300, bbox_inches='tight')
    plt.close()  # Fechar a figura para liberar memória
    print(f"✓ Gráficos salvos em: {output_path}")
    print("✓ Visualizações geradas com sucesso!")


def main():
    """
    Função principal para treinar e avaliar o modelo
    """
    print("="*70)
    print("CLASSIFICAÇÃO DE SENTIMENTOS EM AVALIAÇÕES DE FILMES")
    print("="*70)
    
    # Criar classificador
    classifier = SentimentClassifier(
        model_type='logistic_regression',
        max_features=5000
    )
    
    # Carregar dados (usando amostra de 5000 para demonstração)
    # Para usar o dataset completo, use sample_size=None
    X_train, y_train, X_test, y_test = classifier.load_data(sample_size=1000)
    
    # Treinar modelo
    classifier.train(X_train, y_train)
    
    # Avaliar modelo
    results = classifier.evaluate(X_test, y_test)
    
    # Salvar modelo
    classifier.save_model()
    
    # Gerar visualizações
    print(f"\n{'='*70}")
    print("GERANDO VISUALIZAÇÕES")
    print(f"{'='*70}")
    generate_visualizations(classifier, results, X_test, y_test)
    
    # Testar com exemplos
    print(f"\n{'='*70}")
    print("TESTE COM NOVOS TEXTOS")
    print(f"{'='*70}")
    
    test_reviews = [
        "This movie was absolutely amazing! The acting was superb and the plot was engaging.",
        "Terrible film. Waste of time and money. I couldn't even finish watching it.",
        "It was okay, nothing special but not terrible either.",
        "Best movie I've seen in years! Highly recommended!",
        "Boring and predictable. The worst movie of the year."
    ]
    
    for i, review in enumerate(test_reviews, 1):
        sentiment, confidence = classifier.predict_sentiment(review)
        print(f"\nReview {i}: {review[:70]}...")
        print(f"Sentimento: {sentiment} (Confiança: {confidence:.2f}%)")
        print("-" * 70)
    
    print(f"\n{'='*70}")
    print("PROCESSO CONCLUÍDO COM SUCESSO!")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()
