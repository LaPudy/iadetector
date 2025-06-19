import logging
import azure.functions as func
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from joblib import load
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
import requests
from io import BytesIO
import nltk
from typing import Dict, Union
from functools import lru_cache

# Initialize NLTK data at startup
nltk.download("stopwords", quiet=True)
nltk.download("wordnet", quiet=True)

# Initialize FastAPI with root_path
app = FastAPI(root_path="/api")

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Configuration constants
MODEL_URL = "" #Poner aca la dirección donde tengas el modelo
TFIDF_URL = "" #Poner la dirección donde tengas el tfidf

class TextRequest(BaseModel):
    text: str

# Load models at startup
try:
    modelo = None
    tfidf_vectorizer = None
    logging.info("Loading models...")
except Exception as e:
    logging.error(f"Error in initial model loading: {str(e)}")

@lru_cache(maxsize=1)
def load_models():
    """Load models with caching to avoid loading on every request"""
    global modelo, tfidf_vectorizer
    try:
        if modelo is None or tfidf_vectorizer is None:
            modelo = load(cargar_archivo(MODEL_URL))
            tfidf_vectorizer = load(cargar_archivo(TFIDF_URL))
        return modelo, tfidf_vectorizer
    except Exception as e:
        logging.error(f"Error loading models: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading models")

def cargar_archivo(url: str) -> BytesIO:
    """Load file from Blob Storage with error handling"""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BytesIO(response.content)
    except requests.exceptions.RequestException as e:
        logging.error(f"Error downloading file from {url}: {str(e)}")
        raise HTTPException(status_code=500, detail="Error loading required files")

@lru_cache(maxsize=1)
def get_stopwords():
    """Cache stopwords to avoid reloading"""
    return set(stopwords.words("spanish"))

def preprocesar_texto(texto: str) -> str:
    """Preprocess text with improved error handling"""
    try:
        lemmatizer = WordNetLemmatizer()
        stop_words = get_stopwords()
        
        texto = texto.lower()
        texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)
        palabras = texto.split()
        palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
        return " ".join(palabras)
    except Exception as e:
        logging.error(f"Error preprocessing text: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing text")

@app.post("/analizar")
async def analizar_texto(request: TextRequest) -> Dict[str, Union[str, float]]:
    logging.info(f"Received request with text: {request.text[:50]}...")
    
    if not request.text.strip():
        raise HTTPException(status_code=400, detail="No se proporcionó texto válido")
    
    try:
        modelo, tfidf_vectorizer = load_models()
        
        texto_preprocesado = preprocesar_texto(request.text)
        texto_tfidf = tfidf_vectorizer.transform([texto_preprocesado])
        prediccion = modelo.predict(texto_tfidf)
        probabilidades = modelo.predict_proba(texto_tfidf)[0]
        
        prob_ia = float(probabilidades[list(modelo.classes_).index('IA')])
        prob_humano = float(probabilidades[list(modelo.classes_).index('Humano')])
        
        response = {
            'prediction': str(prediccion[0]),
            'iaConfidence': prob_ia,
            'humanConfidence': prob_humano
        }
        logging.info(f"Successful prediction: {response}")
        return response
        
    except Exception as e:
        logging.error(f"Error during prediction: {str(e)}")
        raise HTTPException(status_code=500, detail="Error processing prediction")

async def main(req: func.HttpRequest, context: func.Context) -> func.HttpResponse:
    """Main entry point for the Azure Function"""
    logging.info("Function triggered")
    try:
        return await func.AsgiMiddleware(app).handle_async(req, context)
    except Exception as e:
        logging.error(f"Error in main function: {str(e)}")
        return func.HttpResponse(
            body=str(e),
            status_code=500
        )
