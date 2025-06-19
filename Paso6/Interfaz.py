import tkinter as tk
from tkinter import ttk
from tkinter import scrolledtext
from joblib import load
import pandas as pd
from pathlib import Path
from sklearn.feature_extraction.text import TfidfVectorizer
import customtkinter as ctk 


ctk.set_appearance_mode("dark")
ctk.set_default_color_theme("blue")

BASE_DIR = Path(__file__).resolve().parents[1]
modelp= BASE_DIR / "Paso5" / "modelo_clasificacion.pkl"
tfidfp= BASE_DIR / "Paso4" / "tfidf_vectorizer.pkl"
modelo_path = modelp
tfidf_path = tfidfp
modelo = load(modelo_path)
tfidf_vectorizer = load(tfidf_path)

class DetectorApp(ctk.CTk):
    def __init__(self):
        super().__init__()

        self.title("Detector IA vs Humano")
        self.geometry("3600x1800")  

        # Crear el contenedor principal
        self.grid_columnconfigure(0, weight=1)
        self.grid_rowconfigure(0, weight=1)

        # Frame principal
        self.main_frame = ctk.CTkFrame(self)
        self.main_frame.grid(row=0, column=0, padx=40, pady=40, sticky="nsew")  
        self.main_frame.grid_columnconfigure(0, weight=1)

        # Título
        self.title_label = ctk.CTkLabel(
            self.main_frame, 
            text="Detector de Texto IA vs Humano",
            font=ctk.CTkFont(size=48, weight="bold")  
        )
        self.title_label.grid(row=0, column=0, pady=(40, 20))  

        # Subtítulo
        self.subtitle_label = ctk.CTkLabel(
            self.main_frame,
            text="Analiza si un texto fue escrito por IA o por un humano",
            font=ctk.CTkFont(size=65)  
        )
        self.subtitle_label.grid(row=1, column=0, pady=(0, 40)) 

        self.text_input = ctk.CTkTextbox(
            self.main_frame,
            width=1400,  # Duplicar ancho
            height=600,  # Duplicar alto
            font=ctk.CTkFont(size=28)  # Duplicar tamaño de fuente
        )
        self.text_input.grid(row=2, column=0, padx=40, pady=(0, 40))  # Duplicar padding
        self.text_input.insert("1.0", "Ingresa o pega aquí el texto que deseas analizar...")

        # Botón de análisis
        self.analyze_button = ctk.CTkButton(
            self.main_frame,
            text="Analizar Texto",
            command=self.analyze_text,
            width=400,  # Duplicar ancho
            height=80,  # Duplicar alto
            font=ctk.CTkFont(size=30, weight="bold")  # Duplicar tamaño de fuente
        )
        self.analyze_button.grid(row=3, column=0, pady=40)  # Duplicar padding

        # Frame para resultados
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.grid(row=4, column=0, padx=40, pady=40, sticky="ew")  # Duplicar padding
        self.results_frame.grid_columnconfigure(0, weight=1)

        # Etiqueta de predicción
        self.prediction_label = ctk.CTkLabel(
            self.results_frame,
            text="",
            font=ctk.CTkFont(size=36, weight="bold")  # Duplicar tamaño de fuente
        )
        self.prediction_label.grid(row=0, column=0, pady=(40, 20))  # Duplicar padding

        # Progreso IA
        self.ia_label = ctk.CTkLabel(self.results_frame, text="Probabilidad IA:", font=ctk.CTkFont(size=28))  # Duplicar tamaño de fuente
        self.ia_label.grid(row=1, column=0, pady=(20, 0))  # Duplicar padding
        
        self.ia_progress = ctk.CTkProgressBar(self.results_frame, width=800)  # Duplicar ancho
        self.ia_progress.grid(row=2, column=0, pady=(0, 20))  # Duplicar padding
        self.ia_progress.set(0)

        # Progreso Humano
        self.human_label = ctk.CTkLabel(self.results_frame, text="Probabilidad Humano:", font=ctk.CTkFont(size=28))  # Duplicar tamaño de fuente
        self.human_label.grid(row=3, column=0, pady=(20, 0))  # Duplicar padding
        
        self.human_progress = ctk.CTkProgressBar(self.results_frame, width=800)  # Duplicar ancho
        self.human_progress.grid(row=4, column=0, pady=(0, 40))  # Duplicar padding
        self.human_progress.set(0)

    def preprocesar_texto(self, texto):
        from nltk.corpus import stopwords
        from nltk.stem import WordNetLemmatizer
        import re
        
        lemmatizer = WordNetLemmatizer()
        stop_words = set(stopwords.words("spanish"))
        
        texto = texto.lower()
        texto = re.sub(r"[^a-záéíóúñü\s]", "", texto)
        palabras = texto.split()
        palabras = [lemmatizer.lemmatize(palabra) for palabra in palabras if palabra not in stop_words]
        return " ".join(palabras)

    def analyze_text(self):
        texto = self.text_input.get("1.0", tk.END).strip()
        if not texto:
            self.prediction_label.configure(text="Por favor, ingresa un texto para analizar")
            return

        # Preprocesar y predecir
        texto_preprocesado = self.preprocesar_texto(texto)
        texto_tfidf = tfidf_vectorizer.transform([texto_preprocesado])
        prediccion = modelo.predict(texto_tfidf)
        probabilidades = modelo.predict_proba(texto_tfidf)[0]

        # Obtener probabilidades
        prob_ia = probabilidades[list(modelo.classes_).index('IA')]
        prob_humano = probabilidades[list(modelo.classes_).index('Humano')]

        # Actualizar interfaz
        self.prediction_label.configure(
            text=f"Predicción: {prediccion[0]}"
        )
        
        # Actualizar barras de progreso
        self.ia_progress.set(prob_ia)
        self.ia_label.configure(text=f"Probabilidad IA: {prob_ia*100:.1f}%")
        
        self.human_progress.set(prob_humano)
        self.human_label.configure(text=f"Probabilidad Humano: {prob_humano*100:.1f}%")

if __name__ == "__main__":
    app = DetectorApp()
    app.mainloop()
