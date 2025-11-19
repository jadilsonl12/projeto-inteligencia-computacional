# 🚀 Guia de Execução do Projeto

## Ambiente Virtual Criado com Sucesso! ✅

O ambiente virtual `venv` foi criado e todas as dependências foram instaladas.

## Como Ativar o Ambiente Virtual

### No PowerShell (Windows):
```powershell
.\venv\Scripts\Activate.ps1
```

### Se houver erro de política de execução, execute primeiro:
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### No CMD (Windows):
```cmd
venv\Scripts\activate.bat
```

## Como Executar o Projeto

### 1. Ativar o ambiente virtual
```powershell
.\venv\Scripts\Activate.ps1
```

### 2. Executar o script Python
```powershell
python src\sentiment_classifier.py
```

**Nota:** O script usa uma amostra de 5000 exemplos por padrão para demonstração rápida. 
Para usar o dataset completo (50.000 exemplos), edite o arquivo `src/sentiment_classifier.py` 
e altere `sample_size=5000` para `sample_size=None` na linha 309.

### 3. Executar o Notebook Jupyter
```powershell
jupyter notebook notebooks\sentiment_analysis.ipynb
```

Ou para abrir o Jupyter Lab:
```powershell
jupyter lab
```

## Verificar Instalações

Para verificar se todas as bibliotecas estão instaladas corretamente:
```powershell
python test_imports.py
```

## Estrutura de Diretórios

```
projeto-inteligencia-computacional/
│
├── venv/                        # Ambiente virtual (NÃO commitar no Git)
├── data/                        # Dados serão baixados aqui automaticamente
├── models/                      # Modelos treinados salvos
├── notebooks/
│   └── sentiment_analysis.ipynb # Notebook completo com análises
├── src/
│   └── sentiment_classifier.py  # Script Python executável
├── .gitignore                   # Ignora venv, cache, etc.
├── LICENSE                      # Licença MIT
├── README.md                    # Documentação principal
├── requirements.txt             # Dependências do projeto
└── test_imports.py             # Script de teste de importações
```

## Tempo de Execução Estimado

### Script Python (sample_size=5000):
- Download do dataset: ~1-2 minutos (primeira vez)
- Pré-processamento: ~2-3 minutos
- Treinamento: ~1-2 minutos
- **Total: ~5-7 minutos**

### Notebook Jupyter:
- Dependendo de quantas células você executar
- Análise exploratória completa: ~10-15 minutos

## Comandos Úteis

### Ver pacotes instalados:
```powershell
pip list
```

### Atualizar um pacote:
```powershell
pip install --upgrade nome-do-pacote
```

### Desativar o ambiente virtual:
```powershell
deactivate
```

### Limpar cache do Python:
```powershell
Get-ChildItem -Path . -Include __pycache__,*.pyc -Recurse | Remove-Item -Force -Recurse
```

## Problemas Comuns e Soluções

### 1. Erro ao ativar o ambiente virtual
**Problema:** Script não pode ser carregado devido à política de execução
**Solução:** 
```powershell
Set-ExecutionPolicy -ExecutionPolicy RemoteSigned -Scope CurrentUser
```

### 2. Download lento do dataset
**Problema:** Dataset IMDb demora para baixar
**Solução:** Seja paciente na primeira execução. O dataset será cacheado localmente.

### 3. Erro de memória
**Problema:** Memória insuficiente para processar o dataset completo
**Solução:** Use uma amostra menor ajustando `sample_size` no código.

### 4. Jupyter não abre
**Problema:** Jupyter não inicia
**Solução:** 
```powershell
python -m jupyter notebook
```

## Primeiros Passos Recomendados

1. ✅ Ativar o ambiente virtual
2. ✅ Executar `test_imports.py` para verificar as instalações
3. ✅ Executar o script Python para treinar o modelo
4. ✅ Abrir o notebook para análise exploratória detalhada
5. ✅ Experimentar com seus próprios textos de reviews

## Publicar no GitHub

```powershell
# Inicializar repositório Git
git init

# Adicionar arquivos
git add .

# Fazer commit inicial
git commit -m "Initial commit: Sentiment Analysis project"

# Renomear branch para main
git branch -M main

# Adicionar repositório remoto (substitua com seu URL)
git remote add origin https://github.com/seu-usuario/projeto-inteligencia-computacional.git

# Fazer push
git push -u origin main
```

**Nota:** O arquivo `.gitignore` já está configurado para não enviar:
- Ambiente virtual (`venv/`)
- Cache Python (`__pycache__/`)
- Modelos treinados (`.pkl`, `.h5`)
- Notebooks checkpoints
- Dados locais

## Recursos Adicionais

- **Documentação NLTK:** https://www.nltk.org/
- **Documentação Scikit-learn:** https://scikit-learn.org/
- **Dataset IMDb:** https://huggingface.co/datasets/imdb
- **Jupyter Notebook:** https://jupyter.org/

---

**Desenvolvido para o curso de Inteligência Computacional**

Boa sorte com seu projeto! 🎉
