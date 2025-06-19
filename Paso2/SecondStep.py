import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from pathlib import Path
import re

# Descargar recursos necesarios para NLTK
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

# Cargar el CSV generado en el paso anterior
BASE_DIR = Path(__file__).resolve().parents[1]
unify = BASE_DIR / "Paso1" / "reseñas_unificadas.csv"
df = pd.read_csv(unify)

# Inicializar herramientas de preprocesamiento
stop_words = set(stopwords.words('spanish'))  # Cambiar a "english" si las reseñas están en inglés
stop_words.update(['cleaned', 'body'])
lemmatizer = WordNetLemmatizer()

# Función de preprocesamiento
def preprocesar_texto(texto):
    try:
        # Manejar casos donde el texto es nulo
        if pd.isna(texto):
            return ""
            
        # Convertir a minúsculas
        texto = texto.lower()
        # Eliminar caracteres especiales y números
        texto = re.sub(r'[^a-záéíóúüñ\s]', '', texto)
        # Tokenizar texto
        tokens = word_tokenize(texto)
        # Eliminar stop words
        tokens = [word for word in tokens if word not in stop_words]
        # Lemmatizar tokens
        tokens = [lemmatizer.lemmatize(word) for word in tokens]
        # Reunir en un texto limpio
        return " ".join(tokens)
    except Exception as e:
        print(f"Error en el preprocesamiento del texto: {e}")
        return ""

try:
    # Revisar el DataFrame antes de procesar
    print("Verificando datos nulos o problemas en el DataFrame...")
    print(df.info())  # Información básica del DataFrame
    
    # Confirmar si hay filas con valores nulos
    print("Datos nulos por columna:")
    print(df.isnull().sum())
    
    # Aplicar preprocesamiento al cuerpo de cada reseña
    print("Aplicando preprocesamiento...")
    df["Cuerpo Preprocesado"] = df["Cuerpo"].apply(preprocesar_texto)
    
    # Definir el nombre del archivo de salida explícitamente
    output_preprocessed_csv = "reseñas_preprocesadas.csv"
    
    # Guardar el DataFrame
    df.to_csv(output_preprocessed_csv, index=False, encoding='utf-8')
    print(f"Archivo guardado exitosamente como {output_preprocessed_csv}")
    
    # Mostrar algunas filas para verificar el resultado
    print("\nPrimeras filas del DataFrame procesado:")
    print(df[["Cuerpo", "Cuerpo Preprocesado"]].head())
    
except Exception as e:
    print(f"Error durante el proceso: {e}")
