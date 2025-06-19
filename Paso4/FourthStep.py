from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.corpus import stopwords
import pickle
from scipy.sparse import save_npz
from pathlib import Path
import pandas as pd
import numpy as np

# Descargar palabras vacías si no se han descargado previamente
import nltk
nltk.download('stopwords')

# Obtener palabras vacías para español
spanish_stopwords = stopwords.words('spanish')

# Cargar los datos de entrenamiento y prueba
BASE_DIR = Path(__file__).resolve().parents[1]
trainee= BASE_DIR / "Paso3" / "reseñas_entrenamiento.csv"
tests= BASE_DIR / "Paso3" / "reseñas_prueba.csv"
train_data = pd.read_csv(trainee)
test_data = pd.read_csv(tests)

# Verificar valores nulos y reemplazarlos con cadenas vacías
train_data["Texto"].fillna("", inplace=True)
test_data["Texto"].fillna("", inplace=True)

# Inicializar el vectorizador TF-IDF con palabras vacías personalizadas
tfidf_vectorizer = TfidfVectorizer(
    max_features=5000,
    stop_words=spanish_stopwords,  # Lista de palabras vacías en español
    ngram_range=(1, 2)
)

# Ajustar el vectorizador en el conjunto de entrenamiento y transformar
X_train_tfidf = tfidf_vectorizer.fit_transform(train_data["Texto"])

# Transformar el conjunto de prueba usando el vectorizador ajustado
X_test_tfidf = tfidf_vectorizer.transform(test_data["Texto"])

# Guardar el vectorizador para futuros usos
with open("tfidf_vectorizer.pkl", "wb") as f:
    pickle.dump(tfidf_vectorizer, f)

# Guardar los datos vectorizados como archivos .npz
save_npz("X_train_tfidf.npz", X_train_tfidf)
save_npz("X_test_tfidf.npz", X_test_tfidf)

# Guardar las etiquetas en archivos separados
train_data["Etiqueta"].to_csv("y_train.csv", index=False, header=False)
test_data["Etiqueta"].to_csv("y_test.csv", index=False, header=False)

# Mostrar estadísticas básicas
print(f"Tamaño de X_train_tfidf: {X_train_tfidf.shape}")
print(f"Tamaño de X_test_tfidf: {X_test_tfidf.shape}")
