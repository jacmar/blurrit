#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFilter
import io
import base64
import random
import os
from datetime import datetime
import zipfile

# Configurazione base
st.set_page_config(
    page_title="Computational Photography Effects",
    page_icon="📷",
    layout="wide"
)

def simulate_depth_map(image, blur_range=(3, 25), randomness=0.5, ghosting=0.3, seed=None):
    """
    Simula una depth map con tecniche di edge detection e gradiente
    Usa la depth map per creare un effetto di messa a fuoco computazionale
    """
    # Imposta seed per risultati riproducibili
    if seed is None:
        seed = random.randint(1, 9999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Converti l'immagine in formato numpy array
    img_array = np.array(image)
    
    # Converti in scala di grigi per l'analisi
    if len(img_array.shape) == 3:
        gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
    else:
        gray = img_array.copy()
    
    # 1. Genera una depth map approssimativa utilizzando vari segnali visivi
    
    # Rileva bordi usando Canny (i bordi sono spesso a diverse profondità)
    edges = cv2.Canny(gray, 50, 150)
    
    # Calcola il gradiente dell'immagine (le aree con gradiente alto sono spesso in primo piano)
    sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(sobelx**2 + sobely**2)
    gradient_magnitude = cv2.normalize(gradient_magnitude, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Calcola una mappa di texture (aree ricche di texture sono spesso in primo piano)
    texture = cv2.Laplacian(gray, cv2.CV_64F)
    texture = np.abs(texture)
    texture = cv2.normalize(texture, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    # Combina questi segnali in una mappa di profondità primitiva
    depth_map = cv2.addWeighted(
        cv2.addWeighted(edges, 0.3, gradient_magnitude, 0.3, 0),
        0.6, texture, 0.4, 0
    )
    
    # 2. Migliora la depth map
    
    # Applica smoothing per rendere più naturale la transizione di profondità
    depth_map = cv2.GaussianBlur(depth_map, (21, 21), 0)
    
    # Aggiungi variazioni casuali per simulare l'incertezza della depth estimation
    if randomness > 0:
        noise = np.random.normal(0, randomness * 30, depth_map.shape).astype(np.uint8)
        depth_map = cv2.add(depth_map, noise)
    
    # Normalizza i valori
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. Simula il "neural engine" che seleziona punti focali in modo non uniforme
    
    # Seleziona casualmente alcune aree da mettere a fuoco in base alla depth map
    height, width = depth_map.shape
    focus_points = []
    
    # Numero di punti focali basato sulla casualità ma più controllato
    num_points = int(3 + randomness * 4)
    
    # Cerca aree con alti valori nella depth map (potenziali punti di interesse)
    potential_points = []
    step_y = max(1, height // 30)
    step_x = max(1, width // 30)
    
    for y in range(0, height, step_y):
        for x in range(0, width, step_x):
            region = depth_map[max(0, y-10):min(height, y+10), max(0, x-10):min(width, x+10)]
            if region.size > 0:
                avg_depth = np.mean(region)
                # Aggiungi casualità alla selezione dei punti
                if avg_depth > 100 * random.uniform(0.7, 1.3):
                    potential_points.append((x, y, avg_depth))
    
    # Se non ci sono abbastanza punti potenziali, aggiungi punti casuali
    while len(potential_points) < num_points:
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        depth = depth_map[y, x] * random.uniform(0.8, 1.2)  # Aggiungi variazione
        potential_points.append((x, y, depth))
    
    # Ordina per profondità e seleziona i più prominenti con casualità
    random.shuffle(potential_points)  # Mescola per aggiungere variazione
    potential_points.sort(key=lambda p: p[2] * random.uniform(0.9, 1.1), reverse=True)
    
    # Seleziona punti non troppo vicini tra loro
    min_distance = min(width, height) / (6 + randomness * 4)  # Distanza minima basata sulla casualità
    
    for x, y, _ in potential_points:
        # Controlla se il punto è abbastanza distante dagli altri punti selezionati
        if all(np.sqrt((x-fx)**2 + (y-fy)**2) > min_distance for fx, fy in focus_points):
            focus_points.append((x, y))
            if len(focus_points) >= num_points:
                break
    
    # 4. Crea una maschera di messa a fuoco basata sui punti selezionati
    focus_mask = np.zeros_like(depth_map, dtype=np.float32)
    
    for x, y in focus_points:
        # Dimensione del punto di messa a fuoco variabile e naturale
        depth_value = depth_map[y, x]
        
        # Calcola il raggio in base alla profondità e alla casualità
        # Più piccolo per randomness alto (aree di focus più piccole)
        base_radius = min(width, height) / (6 + randomness * 4)
        radius = int(base_radius * random.uniform(0.8, 1.2))
        
        # Crea forme più organiche e meno circolari
        if random.random() < 0.7:  # 70% delle volte utilizziamo forme più organiche
            # Crea una forma irregolare
            num_vertices = random.randint(5, 8)
            vertices = []
            
            for i in range(num_vertices):
                angle = 2 * np.pi * i / num_vertices
                # Raggio variabile per ogni vertice
                r = radius * random.uniform(0.7, 1.3)
                vx = x + int(r * np.cos(angle))
                vy = y + int(r * np.sin(angle))
                
                # Assicurati che i vertici siano all'interno dell'immagine
                vx = max(0, min(width-1, vx))
                vy = max(0, min(height-1, vy))
                vertices.append([vx, vy])
            
            # Disegna il poligono
            temp_mask = np.zeros_like(depth_map, dtype=np.float32)
            cv2.fillPoly(temp_mask, [np.array(vertices, dtype=np.int32)], 1.0)
            
            # Sfuma i bordi
            temp_mask = cv2.GaussianBlur(temp_mask, (21, 21), 0)
            
            # Aggiungi alla maschera principale
            focus_mask = np.maximum(focus_mask, temp_mask)
        else:
            # Usa un'ellisse con proporzioni casuali
            a = radius * random.uniform(0.8, 1.2)
            b = radius * random.uniform(0.8, 1.2)
            angle = random.uniform(0, 360)
            
            # Crea una maschera temporanea
            temp_mask = np.zeros_like(depth_map, dtype=np.float32)
            cv2.ellipse(
                temp_mask, 
                (int(x), int(y)), 
                (int(a), int(b)), 
                angle, 0, 360, 1.0, -1
            )
            
            # Sfuma i bordi molto di più per evitare bordi visibili
            temp_mask = cv2.GaussianBlur(temp_mask, (41, 41), 0)
            
            # Aggiungi alla maschera principale
            focus_mask = np.maximum(focus_mask, temp_mask)
    
    # 5. Applica l'effetto di messa a fuoco utilizzando la maschera
    
    # Assicurati che i valori di blur siano nel range corretto
    min_blur, max_blur = blur_range
    min_blur = max(1, min_blur)
    max_blur = max(min_blur + 2, max_blur)
    
    # Crea versioni dell'immagine con diversi livelli di sfocatura
    blurred_images = []
    blur_steps = 5  # Numero di livelli di sfocatura
    
    blur_values = np.linspace(min_blur, max_blur, blur_steps)
    
    for blur_radius in blur_values:
        blur_radius = int(blur_radius)
        if blur_radius % 2 == 0:  # Assicurati che il kernel size sia dispari
            blur_radius += 1
        blurred = cv2.GaussianBlur(img_array, (blur_radius, blur_radius), 0)
        blurred_images.append(blurred)
    
    # Inizializza l'immagine risultato
    result = np.zeros_like(img_array, dtype=np.float32)
    
    # Normalizza la depth map e la maschera di focus per l'interpolazione
    normalized_depth = depth_map.astype(np.float32) / 255.0
    normalized_focus = focus_mask
    
    # Applica effetto ghosting se richiesto (simula la sovrapposizione di più scatti leggermente diversi)
    if ghosting > 0:
        # Crea copie spostate dell'immagine originale
        ghost_images = []
        
        # Numero di "ghost" basato sull'intensità del ghosting
        num_ghosts = int(2 + ghosting * 3)
        
        for i in range(num_ghosts):
            # Calcola spostamento
            shift_x = int(ghosting * width * 0.01 * random.uniform(-1, 1))
            shift_y = int(ghosting * height * 0.01 * random.uniform(-1, 1))
            
            # Matrice di traslazione
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            
            # Applica la traslazione
            shifted = cv2.warpAffine(img_array, M, (width, height))
            
            # Peso decrescente per ogni ghost successivo
            weight = 1.0 / (i + 1) * 0.5
            
            ghost_images.append((shifted, weight))
    
    # Combina l'immagine originale e le versioni sfocate in base alla profondità e ai punti di focus
    for y in range(height):
        for x in range(width):
            # Calcola il peso di messa a fuoco (combinazione di profondità e maschera di focus)
            focus_weight = normalized_focus[y, x]
            depth_weight = 1.0 - normalized_depth[y, x]
            
            # Combinazione ponderata
            weight = focus_weight * 0.7 + depth_weight * 0.3
            
            # Scegli l'indice di sfocatura in base al peso
            blur_idx = min(blur_steps - 1, int((1.0 - weight) * blur_steps))
            
            # Interpola tra l'originale e la versione sfocata appropriata
            if len(img_array.shape) == 3:
                # Immagine a colori
                result[y, x] = weight * img_array[y, x] + (1.0 - weight) * blurred_images[blur_idx][y, x]
            else:
                # Immagine in scala di grigi
                result[y, x] = weight * img_array[y, x] + (1.0 - weight) * blurred_images[blur_idx][y, x]
    
    # Applica l'effetto ghosting se richiesto
    if ghosting > 0 and 'ghost_images' in locals():
        for ghost_img, weight in ghost_images:
            # Mescola il ghost con il risultato
            result = result * (1.0 - weight) + ghost_img * weight
    
    # Converti il risultato in un'immagine PIL
    result_img = Image.fromarray(result.astype(np.uint8))
    
    # Opzionalmente, restituisci anche la depth map per visualizzazione/debug
    depth_map_img = Image.fromarray(depth_map)
    focus_mask_img = Image.fromarray((focus_mask * 255).astype(np.uint8))
    
    return result_img, depth_map_img, focus_mask_img, seed

def get_image_download_link(img, filename):
    """Genera un link per scaricare un'immagine"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" style="display:inline-block; padding:0.5rem 1rem; background-color:#4285f4; color:white; text-decoration:none; border-radius:4px;">📥 Scarica</a>'
    return href

def get_zip_download_link(images, filenames, zip_filename="varianti.zip"):
    """Genera un link per scaricare un file zip contenente più immagini"""
    buffer = io.BytesIO()
    
    with zipfile.ZipFile(buffer, 'w') as zip_file:
        for img, filename in zip(images, filenames):
            img_buffer = io.BytesIO()
            img.save(img_buffer, format="JPEG", quality=95)
            zip_file.writestr(filename, img_buffer.getvalue())
    
    buffer.seek(0)
    zip_str = base64.b64encode(buffer.getvalue()).decode()
    
    href = f'<a href="data:application/zip;base64,{zip_str}" download="{zip_filename}" style="display:inline-block; padding:0.5rem 1rem; background-color:#4CAF50; color:white; text-decoration:none; border-radius:4px;">📥 Scarica tutte le varianti (ZIP)</a>'
    return href

def create_filename(base_name, blur_min, blur_max, random, ghosting, seed):
    """Crea nome file con i parametri inclusi"""
    b_min = f"{blur_min:.1f}".replace('.', '_')
    b_max = f"{blur_max:.1f}".replace('.', '_')
    r_str = f"{random:.2f}".replace('.', '_')
    g_str = f"{ghosting:.2f}".replace('.', '_')
    seed_str = f"{seed % 10000:04d}"
    return f"{base_name}_b{b_min}-{b_max}_r{r_str}_g{g_str}_s{seed_str}.jpg"

def stack_images(images, blend_mode="average", randomness=0.5, seed=None):
    """
    Sovrappone più immagini con leggeri spostamenti casuali
    per simulare leggere variazioni nella prospettiva
    """
    if not images or len(images) == 0:
        return None
    
    # Se c'è una sola immagine, ritorna direttamente
    if len(images) == 1:
        return np.array(images[0])
    
    # Imposta seed per risultati riproducibili
    if seed is None:
        seed = random.randint(1, 9999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Converti tutte le immagini in array numpy
    np_images = [np.array(img) for img in images]
    
    # Usa la prima immagine come riferimento per le dimensioni
    height, width = np_images[0].shape[:2]
    channels = 3 if len(np_images[0].shape) == 3 else 1
    
    # Inizializza l'immagine risultato
    if blend_mode == "average":
        # Media ponderata
        result = np.zeros((height, width, channels) if channels == 3 else (height, width), dtype=np.float32)
        total_weight = 0
        
        for i, img in enumerate(np_images):
            # Ridimensiona se necessario
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            # Applica una leggera traslazione casuale
            shift_x = int(randomness * width * 0.05 * random.uniform(-1, 1))
            shift_y = int(randomness * height * 0.05 * random.uniform(-1, 1))
            
            # Matrice di traslazione
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            
            # Applica la traslazione
            shifted = cv2.warpAffine(img, M, (width, height))
            
            # Peso maggiore per le prime immagini
            weight = 1.0 - (i / len(np_images)) * 0.3
            total_weight += weight
            
            # Aggiungi al risultato
            result += shifted.astype(np.float32) * weight
        
        # Normalizza
        result /= total_weight
        
    elif blend_mode == "lighten":
        # Prendi il valore più luminoso per ogni pixel
        result = np.zeros_like(np_images[0], dtype=np.float32)
        
        for i, img in enumerate(np_images):
            # Ridimensiona se necessario
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            # Applica una leggera traslazione casuale
            shift_x = int(randomness * width * 0.05 * random.uniform(-1, 1))
            shift_y = int(randomness * height * 0.05 * random.uniform(-1, 1))
            
            # Matrice di traslazione
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            
            # Applica la traslazione
            shifted = cv2.warpAffine(img, M, (width, height))
            
            # Aggiorna il risultato con il valore massimo
            result = np.maximum(result, shifted.astype(np.float32))
    
    else:  # "darken"
        # Prendi il valore più scuro per ogni pixel
        result = np.ones_like(np_images[0], dtype=np.float32) * 255
        
        for i, img in enumerate(np_images):
            # Ridimensiona se necessario
            if img.shape[:2] != (height, width):
                img = cv2.resize(img, (width, height))
            
            # Applica una leggera traslazione casuale
            shift_x = int(randomness * width * 0.05 * random.uniform(-1, 1))
            shift_y = int(randomness * height * 0.05 * random.uniform(-1, 1))
            
            # Matrice di traslazione
            M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
            
            # Applica la traslazione
            shifted = cv2.warpAffine(img, M, (width, height))
            
            # Aggiorna il risultato con il valore minimo
            result = np.minimum(result, shifted.astype(np.float32))
    
    return result.astype(np.uint8)

# Layout principale
st.title("📷 Computational Photography Effects")
st.markdown("Simulazione di effetti di fotografia computazionale")

# Layout a colonne
col1, col2 = st.columns([1, 2])

with col1:
    st.subheader("Controlli")
    
    # Upload immagini
    uploaded_files = st.file_uploader(
        "Carica immagini",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Carica una o più immagini. Se carichi più immagini, verranno combinate."
    )
    
    # Parametri
    st.subheader("Parametri")
    
    min_blur = st.slider(
        "Sfocatura minima:",
        min_value=1,
        max_value=15,
        value=3,
        step=2,
        help="Valore minimo di sfocatura"
    )
    
    max_blur = st.slider(
        "Sfocatura massima:",
        min_value=min_blur + 2,
        max_value=35,
        value=25,
        step=2,
        help="Valore massimo di sfocatura"
    )
    
    randomness = st.slider(
        "Casualità:",
        min_value=0.0,
        max_value=1.0,
        value=0.5,
        step=0.05,
        help="Aggiunge variazione nella selezione dei punti a fuoco"
    )
    
    ghosting = st.slider(
        "Effetto Ghosting:",
        min_value=0.0,
        max_value=1.0,
        value=0.3,
        step=0.05,
        help="Simula la sovrapposizione di immagini leggermente diverse"
    )
    
    # Opzioni avanzate
    with st.expander("Opzioni avanzate"):
        show_debug = st.checkbox(
            "Mostra depth map",
            value=False,
            help="Visualizza la depth map e la maschera di focus"
        )
        
        if len(uploaded_files) > 1:
            blend_mode = st.selectbox(
                "Modalità di stacking:",
                ["average", "lighten", "darken"],
                index=0,
                help="Come combinare più immagini: media ponderata, solo pixel più chiari, solo pixel più scuri"
            )
    
    # Seed
    seed_option = st.radio(
        "Seed:",
        ["Casuale", "Specifico"]
    )
    
    seed = None
    if seed_option == "Specifico":
        seed = st.number_input(
            "Valore seed (1-9999):",
            min_value=1,
            max_value=9999,
            value=42
        )
    
    # Pulsante elabora
    process_button = st.button(
        "Elabora immagine",
        type="primary",
        use_container_width=True
    )
    
    # Pulsante genera varianti
    if uploaded_files:
        generate_variants = st.button(
            "Genera 4 varianti",
            use_container_width=True
        )

with col2:
    if uploaded_files:
        # Mostra immagini caricate
        st.subheader("Immagini caricate")
        
        # Mostra miniature in galleria
        cols = st.columns(min(3, len(uploaded_files)))
        images = []
        
        for i, file in enumerate(uploaded_files):
            with cols[i % len(cols)]:
                img = Image.open(file)
                
                # Ridimensiona se troppo grande
                if max(img.size) > 1000:
                    ratio = 1000 / max(img.size)
                    new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                    img = img.resize(new_size, Image.LANCZOS)
                
                images.append(img)
                st.image(img, width=150)
        
        # Elabora immagini quando viene premuto il pulsante
        if process_button or generate_variants:
            st.markdown("---")
            
            if process_button:
                st.subheader("Risultato")
                
                try:
                    # Se ci sono più immagini, combinale
                    if len(images) > 1:
                        stacked = stack_images(images, blend_mode if 'blend_mode' in locals() else "average", randomness, seed)
                        combined_img = Image.fromarray(stacked)
                    else:
                        combined_img = images[0]
                    
                    # Applica l'effetto di depth map
                    result, depth_map, focus_mask, used_seed = simulate_depth_map(
                        combined_img,
                        (min_blur, max_blur),
                        randomness,
                        ghosting,
                        seed
                    )
                    
                    # Mostra risultato
                    st.image(result, use_container_width=True)
                    
                    # Mostra i parametri utilizzati
                    st.info(f"""
                    **Parametri utilizzati:**
                    - Sfocatura: min={min_blur}, max={max_blur}
                    - Casualità: {randomness:.2f}
                    - Ghosting: {ghosting:.2f}
                    - Seed: {used_seed}
                    """)
                    
                    # Mostra debug info se richiesto
                    if show_debug:
                        debug_cols = st.columns(2)
                        with debug_cols[0]:
                            st.subheader("Depth Map")
                            st.image(depth_map, use_container_width=True)
                        with debug_cols[1]:
                            st.subheader("Focus Mask")
                            st.image(focus_mask, use_container_width=True)
                    
                    # Crea nome file
                    if len(uploaded_files) == 1:
                        base_name = os.path.splitext(uploaded_files[0].name)[0]
                    else:
                        base_name = f"combined_{len(uploaded_files)}_images"
                    
                    filename = create_filename(
                        base_name,
                        min_blur,
                        max_blur,
                        randomness,
                        ghosting,
                        used_seed
                    )
                    
                    # Mostra informazioni e link download
                    st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Si è verificato un errore: {str(e)}")
                    st.info("Prova a modificare i parametri o a caricare un'immagine diversa.")
            
            elif generate_variants:
                st.subheader("Varianti")
                
                try:
                    # Se ci sono più immagini, combinale
                    if len(images) > 1:
                        stacked = stack_images(images, blend_mode if 'blend_mode' in locals() else "average", randomness, seed)
                        combined_img = Image.fromarray(stacked)
                    else:
                        combined_img = images[0]
                    
                    # Crea 4 varianti UNICHE con seed diversi
                    variant_images = []
                    variant_filenames = []
                    
                    # Base name per i file
                    if len(uploaded_files) == 1:
                        base_name = os.path.splitext(uploaded_files[0].name)[0]
                    else:
                        base_name = f"combined_{len(uploaded_files)}_images"
                    
                    # Crea 2x2 grid for visualization
                    variant_cols1 = st.columns(2)
                    variant_cols2 = st.columns(2)
                    
                    for i, col in enumerate([*variant_cols1, *variant_cols2]):
                        with col:
                            # Ogni variante ha un seed diverso E parametri diversi
                            variant_seed = (seed + i * 100) if seed else random.randint(1, 9999)
                            
                            # Varia anche i parametri
                            variant_randomness = randomness * random.uniform(0.8, 1.2)
                            variant_ghosting = ghosting * random.uniform(0.8, 1.2)
                            variant_min_blur = min(min_blur + random.randint(-2, 2), min_blur)
                            variant_max_blur = max(max_blur + random.randint(-5, 5), variant_min_blur + 5)
                            
                            # Applica l'effetto
                            result, _, _, used_seed = simulate_depth_map(
                                combined_img,
                                (variant_min_blur, variant_max_blur),
                                variant_randomness,
                                variant_ghosting,
                                variant_seed
                            )
                            
                            # Salva per download multiplo
                            variant_images.append(result)
                            variant_filename = create_filename(
                                f"{base_name}_variant_{i+1}",
                                variant_min_blur,
                                variant_max_blur,
                                variant_randomness,
                                variant_ghosting,
                                used_seed
                            )
                            variant_filenames.append(variant_filename)
                            
                            # Mostra risultato
                            st.image(result, use_container_width=True)
                            
                            # Mostra i parametri utilizzati
                            st.info(f"""
                            **Variante {i+1}:**
                            - Sfocatura: min={variant_min_blur}, max={variant_max_blur}
                            - Casualità: {variant_randomness:.2f}
                            - Ghosting: {variant_ghosting:.2f}
                            - Seed: {used_seed}
                            """)
                            
                            # Mostra il link per il download
                            st.markdown(get_image_download_link(result, variant_filename), unsafe_allow_html=True)
                    
                    # Aggiungi un link per scaricare tutte le varianti in un file ZIP
                    st.markdown("---")
                    st.markdown(get_zip_download_link(variant_images, variant_filenames, f"{base_name}_varianti.zip"), unsafe_allow_html=True)
                    
                except Exception as e:
                    st.error(f"Si è verificato un errore: {str(e)}")
                    st.info("Prova a modificare i parametri o a caricare un'immagine diversa.")
    else:
        # Messaggio iniziale
        st.info("👈 Carica una o più immagini per iniziare.")
        
        # Info sull'app
        with st.expander("Informazioni sull'app"):
            st.markdown("""
            ## Computational Photography Effects
            
            Questa app simula effetti di fotografia computazionale simili a quelli che troveresti in smartphone moderni.
            
            ### Come funziona:
            
            1. **Stima della profondità**: L'app analizza l'immagine e genera una mappa di profondità approssimativa
            
            2. **Selezione punti focali**: Punti di interesse vengono selezionati in base a vari segnali visivi
            
            3. **Sfocatura variabile**: Diverse parti dell'immagine vengono sfocate in base alla loro profondità stimata
            
            4. **Effetto ghosting**: Simula la sovrapposizione di immagini leggermente diverse per un effetto più realistico
            
            5. **Stacking di immagini**: Se carichi più immagini, vengono combinate con leggere variazioni
            
            ### Suggerimenti:
            
            - Carica immagini con elementi interessanti distribuiti in diverse aree
            - Prova diverse impostazioni di casualità per ottenere effetti variabili
            - L'effetto ghosting simula la sovrapposizione di scatti leggermente diversi
            - Se carichi più immagini simili, l'algoritmo creerà un effetto di stacking
            
            ### Parametri:
            
            - **Sfocatura min/max**: Controlla l'intensità dell'effetto sfocatura
            - **Casualità**: Aggiunge variazione nella selezione dei punti a fuoco
            - **Ghosting**: Simula la sovrapposizione di immagini leggermente diverse
            """)
