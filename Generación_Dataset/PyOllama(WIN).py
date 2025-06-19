mport subprocess
import shutil
import os
import re
from datetime import datetime
from pathlib import Path

class OllamaError(Exception):
    """Excepción personalizada para errores de Ollama."""
    pass

def count_words(text):
    """Cuenta las palabras en un texto usando text.split()."""
    return len(text.split())

def process_review(review_path, model_name, output_dir):
    """Procesa una reseña humana y genera una nueva reseña con Ollama."""
    try:
        review_path = Path(review_path)
        if not review_path.exists():
            print(f"El archivo no existe: {review_path}")
            return None

        try:
            review_content = review_path.read_text(encoding='utf-8')
        except UnicodeDecodeError:
            print(f"Error de codificación en el archivo: {review_path}")
            return None

        # Extraer el título usando una expresión regular
        title_match = re.search(r"Title:\s*(.+)", review_content)
        if not title_match:
            print(f"No se encontró el título en: {review_path}")
            return None

        title = title_match.group(1).strip()
        
        rank_match = re.search(r"Rank:\s*(\d+)", review_content)
        if rank_match:
            rank = rank_match.group(1).strip()
        else:
            rank = "No especificado"

        # Contar palabras de la reseña original
        body_match = re.search(r"Cleaned Body:\s*(.+?)(?=\n\n|\Z)", review_content, re.DOTALL)
        if not body_match or not body_match.group(1).strip():
            print(f"No se encontró el cuerpo en: {review_path}")
            return None

        body_text = body_match.group(1).strip()
        word_count = count_words(body_text)

        if word_count == 0:
            print(f"Advertencia: El cuerpo del texto está vacío en: {review_path}")
            return None

        prompt = f"""
        Genera una reseña únicamente en español, sin incluir ningún otro idioma ni símbolos desconocidos, utiliza el rank de "{rank}" y presta atención a las instrucciones. Usa la siguiente estructura:

        Title: [Título de la película]
        Rank: [{rank}]
        Cleaned Body: [Un texto limpio y fluido, escrito en un estilo que parezca hecho por un humano y que intente representar las emociones de si es buena o pesima pelicula. Con respecto al Rank que es {rank} sigue estas reglas para hacer la reseña: positivo si el rank es (4 o 5), neutral si el rank es (3), y crítico de forma negativa si el rank es (1 o 2), que el tipo de lenguaje que usas sea coloquial o bastante informal, no quiero que se note nada de formalidad]

        Basándote en la película: {title}.
        La reseña debe tener aproximadamente {word_count} palabras y solo que lleve estas 3 secciones ni una mas, no menciones al final que tipo de lenguaje usaste, no uses letra en negrita ni ninguna nomenclatura de títulos fuera de la estructura solicitada.
        """

        # Verificar si ollama está instalado
        if not shutil.which("ollama"):
            raise OllamaError("Ollama no está instalado. Asegúrate de que esté en tu PATH.")

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        print(f"Generando reseña para: {title} Rank: {rank} (aprox. {word_count} palabras)")

        # Ejecutar el comando Ollama
        result = subprocess.run(
            ['ollama', 'run', model_name],
            input=prompt,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            universal_newlines=True,
            encoding='utf-8',  # Asegura que el texto se maneje como UTF-8
            check=True,
            timeout=3000
        )

        output = result.stdout.strip()
        if not output:
            print(f"La respuesta de Ollama está vacía para: {title}")
            return None

        output_filename = f"{review_path.stem}_AI.txt"
        output_path = output_dir / output_filename

        # Escribir el archivo con codificación UTF-8
        output_path.write_text(output, encoding='utf-8')
        print(f"Reseña generada y guardada en: {output_path}")

        return output_path

    except subprocess.TimeoutExpired:
        print(f"Timeout al generar la reseña para: {title}")
        return None
    except subprocess.CalledProcessError as e:
        print(f"Error al ejecutar Ollama: {e.stderr}")
        return None
    except OllamaError as e:
        print(f"Error: {e}")
        return None
    except Exception as e:
        print(f"Error al procesar {review_path}: {str(e)}")
        return None

def process_directory(input_dir, model_name, output_dir):
    """Procesa recursivamente todos los archivos .txt en un directorio."""
    input_dir = Path(input_dir)

    if not input_dir.exists():
        print(f"El directorio de entrada no existe: {input_dir}")
        return

    for filepath in input_dir.rglob('*.txt'):
        if filepath.is_file():
            process_review(filepath, model_name, output_dir)

# Configuración inicial
input_directory = Path.cwd() / "reviews_humanas"
output_directory = Path.cwd() / "reviews_generadas"
#model_used = "granite3.1-moe"
#model_used = "qwen:7b"
model_used = "llama3.2:3b-instruct-q8_0"

# Verificar que el directorio de entrada existe
if not input_directory.exists():
    print(f"El directorio de entrada no existe: {input_directory}")
    exit(1)

process_directory(input_directory, model_used, output_directory)
