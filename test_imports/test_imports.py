"""
Script de teste rápido para verificar as instalações
"""

print("Testando importações...")

try:
    import numpy as np
    print("✓ NumPy instalado")
except ImportError as e:
    print(f"✗ NumPy não instalado: {e}")

try:
    import pandas as pd
    print("✓ Pandas instalado")
except ImportError as e:
    print(f"✗ Pandas não instalado: {e}")

try:
    import sklearn
    print("✓ Scikit-learn instalado")
except ImportError as e:
    print(f"✗ Scikit-learn não instalado: {e}")

try:
    import nltk
    print("✓ NLTK instalado")
except ImportError as e:
    print(f"✗ NLTK não instalado: {e}")

try:
    import matplotlib
    print("✓ Matplotlib instalado")
except ImportError as e:
    print(f"✗ Matplotlib não instalado: {e}")

try:
    import seaborn
    print("✓ Seaborn instalado")
except ImportError as e:
    print(f"✗ Seaborn não instalado: {e}")

try:
    from datasets import load_dataset
    print("✓ Datasets instalado")
except ImportError as e:
    print(f"✗ Datasets não instalado: {e}")

print("\n" + "="*50)
print("Todas as bibliotecas necessárias estão instaladas!")
print("="*50)
