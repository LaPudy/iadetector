import os
import pandas as pd
import re
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

class ProcesadorResenas:
    def __init__(self):
        self.data = []
        
    def procesar_archivo(self, args):
        """
        Procesa un único archivo de reseña
        """
        ruta, es_ia = args
        try:
            with open(ruta, 'r', encoding='utf-8') as f:
                contenido = f.read()
                
                # Extraer campos usando expresiones regulares más precisas
                titulo_match = re.search(r'Title:\s*(.*?)(?=\n(?:Rank|Cleaned Body):)', contenido, re.DOTALL)
                rank_match = re.search(r'Rank:\s*(\d+(?:\.\d+)?)', contenido)
                cuerpo_match = re.search(r'Cleaned Body:\s*(.*)', contenido, re.DOTALL)
                
                titulo = titulo_match.group(1).strip() if titulo_match else "N/A"
                rank = rank_match.group(1) if rank_match else "N/A"
                cuerpo = cuerpo_match.group(1).strip() if cuerpo_match else "N/A"
                
                # Validaciones adicionales
                if len(titulo) < 2:
                    titulo = "N/A"
                if not rank.replace('.', '').isdigit():
                    rank = "N/A"
                if len(cuerpo) < 10:  # Asumimos que una reseña válida tiene al menos 10 caracteres
                    cuerpo = "N/A"
                
                return {
                    "Archivo": os.path.basename(ruta),
                    "Título": titulo,
                    "Rank": rank,
                    "Cuerpo": cuerpo,
                    "Etiqueta": "IA" if es_ia else "Humano"
                }
                
        except Exception as e:
            print(f"\nError procesando {ruta}: {str(e)}")
            return None

    def procesar_directorio(self, directorio, es_ia=False):
        """
        Procesa todos los archivos en un directorio usando múltiples hilos
        """
        archivos = [f for f in os.listdir(directorio) if f.endswith('.txt')]
        rutas = [(os.path.join(directorio, archivo), es_ia) for archivo in archivos]
        
        with ThreadPoolExecutor(max_workers=os.cpu_count()) as executor:
            futures = [executor.submit(self.procesar_archivo, ruta) for ruta in rutas]
            
            for future in tqdm(as_completed(futures), 
                             total=len(futures), 
                             desc=f"Procesando {'IA' if es_ia else 'Humano'}"):
                resultado = future.result()
                if resultado:
                    self.data.append(resultado)

    def analizar_resultados(self, df):
        """
        Realiza un análisis detallado de los resultados
        """
        print("\n=== Análisis de Resultados ===")
        print(f"\nEstadísticas Generales:")
        print(f"- Total de reseñas: {len(df):,}")
        print(f"- Distribución por etiqueta:\n{df['Etiqueta'].value_counts()}")
        
        print("\nCalidad de Datos:")
        for columna in ['Rank', 'Título', 'Cuerpo']:
            na_count = df[columna].eq('N/A').sum()
            print(f"- {columna}: {na_count:,} valores N/A ({(na_count/len(df))*100:.1f}%)")
        
        print("\nDistribución de Rankings:")
        rank_stats = df[df['Rank'] != 'N/A']['Rank'].astype(float).describe()
        print(f"- Promedio: {rank_stats['mean']:.2f}")
        print(f"- Mediana: {rank_stats['50%']:.2f}")
        print(f"- Mín/Máx: {rank_stats['min']:.0f}/{rank_stats['max']:.0f}")
        print("\nFrecuencia de Rankings:")
        print(df['Rank'].value_counts().sort_index().head())

        print("\nEstadísticas de Longitud:")
        df['longitud_titulo'] = df['Título'].apply(lambda x: len(x) if x != 'N/A' else 0)
        df['longitud_cuerpo'] = df['Cuerpo'].apply(lambda x: len(x) if x != 'N/A' else 0)
        print(f"- Longitud promedio título: {df['longitud_titulo'].mean():.1f} caracteres")
        print(f"- Longitud promedio cuerpo: {df['longitud_cuerpo'].mean():.1f} caracteres")

def main():
    # Configuración de directorios
    BASE_DIR = Path(__file__).resolve().parents[1]
    human = BASE_DIR / "Dataset" / "reviews_generadas_procesadas"
    AI = BASE_DIR / "Dataset" / "reviews_generadas"
    cleaned_reviews_dir = human
    reviews_generadas_dir = AI
    
    # Inicializar procesador
    procesador = ProcesadorResenas()
    
    # Procesar ambos directorios
    print("Iniciando procesamiento...")
    procesador.procesar_directorio(cleaned_reviews_dir, es_ia=False)
    procesador.procesar_directorio(reviews_generadas_dir, es_ia=True)
    
    # Crear DataFrame
    df = pd.DataFrame(procesador.data)
    
    # Analizar resultados
    procesador.analizar_resultados(df)
    
    # Guardar resultados
    output_csv = "reseñas_unificadas.csv"
    df.to_csv(output_csv, index=False, encoding='utf-8')
    print(f"\nDatos guardados en {output_csv}")

if __name__ == "__main__":
    main()
