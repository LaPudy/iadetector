# Clasificador de Reseñas: IA vs Humano

## Descripción del Proyecto

Este proyecto es un clasificador de texto que determina si una reseña fue escrita por una Inteligencia Artificial o por un humano. Combina técnicas avanzadas de Procesamiento de Lenguaje Natural (NLP) con un modelo de aprendizaje automático, todo integrado en una interfaz gráfica moderna. También se contempla el uso de Azure Functions como API para desplegar el proyecto en la nube.

## Características Principales

- Pipeline completo de procesamiento de texto
- Generación de reseñas con modelos de lenguaje (Ollama)
- Preprocesamiento y limpieza de texto
- Vectorización mediante TF-IDF
- Modelo de clasificación con Regresión Logística
- Interfaz gráfica con CustomTkinter
- Soporte opcional para ejecución en la nube con Azure

## Estructura del Proyecto

```
Clasificador_Reseñas_IA_vs_Humano/
│
├── Generación_Dataset/
│   ├── PyOllama(WIN).py
│   ├── reviews_humanas/
│   └── reviews_generadas/
│
├── Paso1_Unificacion_Datos/
│   ├── FirstStep.py
│   └── Preprocesamiento.py
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
│   ├── reviews_generadas/
│   ├── reviews_generadas_procesadas/
│   └── reviews_humanas/
│
├── requirements.txt
└── README.md
```

## Requisitos del Sistema

- Python 3.8 o superior
- Ollama (opcional, solo si deseas generar nuevas reseñas automáticamente)

### Dependencias de Python

- pandas  
- nltk  
- scikit-learn  
- scipy  
- joblib  
- customtkinter  
- tqdm  

Instálalas con:

```
pip install -r requirements.txt
```

### Recursos de NLTK

Es posible que el programa descargue automáticamente los recursos necesarios, pero puedes hacerlo manualmente:

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
```

## Instalación

1. Clona el repositorio:

```
git clone https://github.com/LaPudy/iadetector.git
cd clasificador-resenas-ia-humano
```

2. Instala las dependencias:

```
pip install -r requirements.txt
```

## Uso

### Pipeline completo

Ejecuta los pasos en orden:

```
# 1. Generar reseñas con IA (requiere Ollama)
python Generación_Dataset/PyOllama(WIN).py

# 2. Procesar reseñas generadas
python Paso1_Unificacion_Datos/Preprocesamiento.py

# 3. Ejecutar pipeline de clasificación
python Paso1_Unificacion_Datos/FirstStep.py
python Paso2_Preprocesamiento_Texto/SecondStep.py
python Paso3_Division_Datos/ThirdStep.py
python Paso4_Vectorizacion/FourthStep.py
python Paso5_Entrenamiento/FifthStep.py

# 4. Lanzar interfaz gráfica
python Paso6_Interfaz/Interfaz.py
```

### Interfaz Gráfica

Permite analizar cualquier texto para determinar si fue escrito por IA o por un humano:

1. Ingresa o pega el texto a analizar  
2. Haz clic en "Analizar Texto"  
3. Observa los resultados:
   - Predicción (IA o Humano)
   - Probabilidades para cada categoría
   - Barras de progreso visuales

## Metodología

1. **Generación de datos**  
   Reseñas humanas y generadas por IA

2. **Preprocesamiento**  
   - Conversión a minúsculas  
   - Eliminación de caracteres especiales  
   - Tokenización  
   - Eliminación de stopwords  
   - Lematización  

3. **Vectorización**  
   TF-IDF con n-gramas (1,2)

4. **Modelado**  
   Regresión Logística

5. **Interfaz**  
   Aplicación gráfica para predicciones en tiempo real

## Licencia

Este proyecto está bajo la licencia MIT.
