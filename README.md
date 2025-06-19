Clasificador de Reseñas: IA vs Humano
Descripción del Proyecto
Este proyecto es un clasificador de texto que determina si una reseña fue escrita por una Inteligencia Artificial o por un humano. Combina técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP) con un modelo de aprendizaje automático, todo integrado en una interfaz gráfica intuitiva y el api usado en azure functions en dado caso de querer hacer el proyecto en la nube.

Características principales:

Pipeline completo de procesamiento de texto

Generación de reseñas con modelos de lenguaje (Ollama)

Preprocesamiento y limpieza de texto

Vectorización TF-IDF

Modelo de clasificación con Regresión Logística

Interfaz gráfica moderna con CustomTkinter

Estructura del proyecto:

Clasificador_Reseñas_IA_vs_Humano/
│
├── Generación_Dataset/
│   ├── PyOllama(WIN).py
│   ├── reviews_humanas/          # Directorio con reseñas humanas originales
│   └── reviews_generadas/        # Reseñas generadas por IA (output)
│
├── Paso1_Unificacion_Datos/
│   └── FirstStep.py
│   └── Preprocesamiento.py       # Procesa reseñas generadas
│
├── Paso2_Preprocesamiento_Texto/
│   └── SecondStep.py
│
├── Paso3_Division_Datos/
│   └── ThirdStep.py
│
├── Paso4_Vectorizacion/
│   └── FourthStep.py
│
├── Paso5_Entrenamiento/
│   └── FifthStep.py
│
├── Paso6_Interfaz/
│   └── Interfaz.py
│
├── Dataset/
│   ├── reviews_generadas/         # Reseñas IA generadas (de Generación_Dataset)
│   ├── reviews_generadas_procesadas/  # Reseñas IA procesadas (de Paso0)
│   └── reviews_humanas/           # Reseñas humanas originales
│
├── requirements.txt
└── README.md

Requisitos del Sistema
Python 3.8+

Ollama instalado (para generación de reseñas si no quieres generar reseñas omitir este paso)

Las siguientes dependencias de Python:

pandas
nltk
scikit-learn
scipy
joblib
customtkinter
tqdm

Instalación
1: Clona el repositorio:

git clone https://github.com/LaPudy/iadetector.git
cd clasificador-resenas-ia-humano

2: Instala las dependencias:

pip install -r requirements.txt

3: Descarga los recursos de NLTK:

import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')

en dado caso de no querer descargarlos en el momento, los programas lo ejecutan para su descarga

Uso
Pipeline completo
Ejecuta los pasos en orden:

# Generar reseñas con IA (requiere Ollama)
python Generación_Dataset/PyOllama(WIN).py

# Procesar reseñas generadas
python Paso1/Preprocesamiento.py

# Ejecutar pipeline de clasificación
python Paso1/FirstStep.py
python Paso2/SecondStep.py
python Paso3/ThirdStep.py
python Paso4/FourthStep.py
python Paso5/FifthStep.py

# Lanzar interfaz gráfica
python Paso6/Interfaz.py

Interfaz Gráfica
La interfaz permite analizar cualquier texto para determinar si fue escrito por IA o humano:

1:Ingresa o pega el texto a analizar
2:Haz clic en "Analizar Texto"
3:Observa los resultados:
    Predicción (IA o Humano)
    Probabilidades para cada categoría
    Barras de progreso visuales

Metodología
1:Generación de datos: Reseñas humanas y generadas por IA
Preprocesamiento:
    Conversión a minúsculas
    Eliminación de caracteres especiales
    Tokenización
    Eliminación de stopwords
    Lematización
3:Vectorización: TF-IDF con n-gramas (1,2)
4:Modelado: Regresión Logística
5:Interfaz: Aplicación gráfica para predicciones

Licencia
Este proyecto está bajo la licencia MIT.


