import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

# Cargar el archivo preprocesado
BASE_DIR = Path(__file__).resolve().parents[1]
prepu= BASE_DIR / "Paso2" / "reseñas_preprocesadas.csv"
df = pd.read_csv(prepu)

# Verificar que las columnas necesarias están presentes
print("Columnas disponibles en el DataFrame:", df.columns)

# Seleccionar características (X) y etiquetas (y)
X = df["Cuerpo Preprocesado"]  # Características: texto preprocesado
y = df["Etiqueta"]  # Etiquetas: IA o Humano

# División en conjuntos de entrenamiento y prueba (80%-20%)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Crear DataFrames para entrenamiento y prueba
train_data = pd.DataFrame({"Texto": X_train, "Etiqueta": y_train})
test_data = pd.DataFrame({"Texto": X_test, "Etiqueta": y_test})

# Guardar los conjuntos en archivos CSV
train_data.to_csv("reseñas_entrenamiento.csv", index=False, encoding="utf-8")
test_data.to_csv("reseñas_prueba.csv", index=False, encoding="utf-8")

# Mostrar estadísticas de los conjuntos
print("Tamaño del conjunto de entrenamiento:", train_data.shape[0])
print("Tamaño del conjunto de prueba:", test_data.shape[0])
print("Distribución en el conjunto de entrenamiento:")
print(train_data["Etiqueta"].value_counts(normalize=True))
print("Distribución en el conjunto de prueba:")
print(test_data["Etiqueta"].value_counts(normalize=True))
