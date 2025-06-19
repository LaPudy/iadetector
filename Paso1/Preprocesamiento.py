import os
import re
from pathlib import Path

def extract_title_rank_body(content):
    """Extract title, rank, and body from review content."""
    lines = [line.strip() for line in content.split('\n') if line.strip()]
    
    if not lines:
        return None, None, None, None
    
    first_line = lines[0]
    title = None
    rank = None
    body = []
    
    # Check for different title and rank patterns
    title_rank_pattern = r'^Title:\s*"([^"]+)"\s*Rank:\s*(\d+(?:\.\d+)?)'
    bold_title_pattern = r'\*\*Título:\*\*\s*(.*)'
    bold_rank_pattern = r'\*\*Rank:\*\*\s*(\d+(?:\.\d+)?)'
    
    # Check for "Title: ... Rank: X" pattern
    match_title_rank = re.match(title_rank_pattern, first_line)
    
    if match_title_rank:
        title = match_title_rank.group(1)
        rank = match_title_rank.group(2)
        body = lines[1:]
    else:
        # Check for bold format
        match_bold_title = re.match(bold_title_pattern, first_line)
        if match_bold_title:
            title = match_bold_title.group(1)
            # Look for bold rank in next lines
            for line in lines[1:]:
                match_bold_rank = re.match(bold_rank_pattern, line)
                if match_bold_rank:
                    rank = match_bold_rank.group(1)
                    body = lines[lines.index(line) + 1:]
                    break
        else:
            # Original format processing
            title = re.sub(r'^Title:\s*', '', first_line)
            
            for line in lines[1:]:
                if rank is None:
                    rank_patterns = [
                        r'^(\d+(?:\.\d+)?(?:/5)?)\s*(.*)$',
                        r'^(\d+(?:\.\d+)?)\s*([A-Z].*)',
                        r'Rank:\s*(\d+(?:\.\d+)?)',
                        r'.*Rank:\s*(\d+(?:\.\d+)?)\s*',
                        r'\*\*Rank:\*\*\s*(\d+(?:\.\d+)?)'
                    ]
                    
                    for pattern in rank_patterns:
                        match = re.search(pattern, line)
                        if match:
                            number = match.group(1)
                            rank = number.split('/')[0] if '/' in number else number
                            rest = match.groups()[-1]
                            if rest.strip():
                                body.append(rest)
                            break
                    if rank is not None:
                        continue
                
                body.append(line)
    
    # Clean up extracted content
    body = ' '.join(body)
    title = title.strip('"') if title else None
    body = re.sub(r'^Cleaned Body:\s*', '', body) if body else None
    body = re.sub(r'Cleaned Body:\s*', '', body) if body else None
    
    # Ensure Cleaned Body does not start with a number
    if body and re.match(r'^\d', body):
        body = re.sub(r'^\d+\s*', '', body)
    
    cleaned_body = f"Cleaned Body:\n{body}" if body else "Cleaned Body:\n"
    
    return title, rank, body, cleaned_body

def process_reviews(input_dir):
    """Main function to process review files."""
    # Create necessary directories
    dest_dir = os.path.join(os.path.dirname(input_dir), 'reviews_generadas_procesadas')
    error_dir = os.path.join(os.path.dirname(input_dir), 'reviews_con_errores')
    
    for directory in [dest_dir, error_dir]:
        if not os.path.exists(directory):
            os.makedirs(directory)
    
    files = [f for f in os.listdir(input_dir) 
             if f.endswith('.txt') and not f.startswith('.')]
    total_files = len(files)
    processed = 0
    errors = 0
    error_log = []
    
    print(f"\nTotal de archivos encontrados: {total_files}")
    
    for i, filename in enumerate(files, 1):
        file_path = os.path.join(input_dir, filename)
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            title, rank, body, cleaned_body = extract_title_rank_body(content)
            
            if not all([title, rank]):
                errors += 1
                missing = []
                if not title: missing.append('título')
                if not rank: missing.append('rank')
                error_log.append(f"Error en {filename}: Falta {', '.join(missing)}")
                # Move error file
                error_path = os.path.join(error_dir, filename)
                with open(file_path, 'r', encoding='utf-8') as f_src:
                    with open(error_path, 'w', encoding='utf-8') as f_dest:
                        f_dest.write(f_src.read())
                os.remove(file_path)
                continue
            
            formatted_content = f"Title: {title}\nRank: {rank}\n{cleaned_body}"
            dest_path = os.path.join(dest_dir, filename)
            
            with open(dest_path, 'w', encoding='utf-8') as f_dest:
                f_dest.write(formatted_content)
                
            processed += 1
            
        except Exception as e:
            errors += 1
            error_log.append(f"Error procesando {filename}: {str(e)}")
            # Move error file
            error_path = os.path.join(error_dir, filename)
            with open(file_path, 'r', encoding='utf-8') as f_src:
                with open(error_path, 'w', encoding='utf-8') as f_dest:
                    f_dest.write(f_src.read())
            os.remove(file_path)
        
        if i % 100 == 0:
            print(f"\rProcesando: {int((i/total_files)*100)}% ({i}/{total_files})", end='')
    
    # Print summary
    print(f"\n\nResumen:")
    print(f"Total archivos encontrados: {total_files}")
    print(f"Archivos procesados exitosamente: {processed}")
    print(f"Errores encontrados: {errors}")
    print(f"Los archivos con errores se han movido a: {error_dir}")
    
    if error_log:
        with open('errores_procesamiento.log', 'w', encoding='utf-8') as f:
            f.write('\n'.join(error_log))
        print("\nSe ha guardado el detalle de los errores en 'errores_procesamiento.log'")
        print("\nPrimeros 5 errores encontrados:")
        for error in error_log[:5]:
            print(error)

if __name__ == "__main__":
    BASE_DIR = Path(__file__).resolve().parents[1]
    input_dir = BASE_DIR / "Dataset" / "reviews_generadas"
    process_reviews(input_dir)
