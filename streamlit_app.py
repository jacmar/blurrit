import streamlit as st
import numpy as np
import cv2
from PIL import Image, ImageFilter
import io
import base64
import random
import os
from datetime import datetime

# Configurazione base
st.set_page_config(
    page_title="Computational Photography Effects",
    page_icon="📷",
    layout="wide"
)

def simulate_depth_map(image, blur_range=(3, 25), randomness=0.5, seed=None):
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
        noise = np.random.normal(0, randomness * 50, depth_map.shape).astype(np.uint8)
        depth_map = cv2.add(depth_map, noise)
    
    # Normalizza i valori
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    
    # 3. Simula il "neural engine" che seleziona punti focali in modo non uniforme
    
    # Seleziona casualmente alcune aree da mettere a fuoco in base alla depth map
    height, width = depth_map.shape
    focus_points = []
    
    # Numero di punti focali basato sulla casualità
    num_points = int(3 + randomness * 5)
    
    # Cerca aree con alti valori nella depth map (potenziali punti di interesse)
    potential_points = []
    for y in range(0, height, 20):
        for x in range(0, width, 20):
            region = depth_map[max(0, y-10):min(height, y+10), max(0, x-10):min(width, x+10)]
            if region.size > 0:
                avg_depth = np.mean(region)
                if avg_depth > 100:  # Solo aree con profondità significativa
                    potential_points.append((x, y, avg_depth))
    
    # Se non ci sono abbastanza punti potenziali, aggiungi punti casuali
    while len(potential_points) < num_points:
        x = random.randint(0, width-1)
        y = random.randint(0, height-1)
        depth = depth_map[y, x]
        potential_points.append((x, y, depth))
    
    # Ordina per profondità e seleziona i più prominenti
    potential_points.sort(key=lambda p: p[2], reverse=True)
    
    # Seleziona punti non troppo vicini tra loro
    for x, y, _ in potential_points:
        # Controlla se il punto è abbastanza distante dagli altri punti selezionati
        if all(np.sqrt((x-fx)**2 + (y-fy)**2) > width/5 for fx, fy in focus_points):
            focus_points.append((x, y))
            if len(focus_points) >= num_points:
                break
    
    # 4. Crea una maschera di messa a fuoco basata sui punti selezionati
    focus_mask = np.zeros_like(depth_map, dtype=np.float32)
    
    for x, y in focus_points:
        # Dimensione del punto di messa a fuoco basata sulla profondità
        depth_value = depth_map[y, x]
        radius = int(50 + (255 - depth_value) / 255 * 100 * random.uniform(0.8, 1.2))
        
        # Crea una maschera circolare per questo punto
        temp_mask = np.zeros_like(depth_map, dtype=np.float32)
        cv2.circle(temp_mask, (x, y), radius, 1.0, -1)
        
        # Sfuma i bordi
        temp_mask = cv2.GaussianBlur(temp_mask, (51, 51), 0)
        
        # Aggiungi alla maschera principale
        focus_mask = np.maximum(focus_mask, temp_mask)
    
    # 5. Applica l'effetto di messa a fuoco utilizzando la maschera
    
    # Crea versioni dell'immagine con diversi livelli di sfocatura
    blurred_images = []
    blur_steps = 5  # Numero di livelli di sfocatura
    
    min_blur, max_blur = blur_range
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
    
    # Combina l'immagine originale e le versioni sfocate in base alla profondità e ai punti di focus
    for y in range(height):
        for x in range(width):
            # Calcola il peso di messa a fuoco (combinazione di profondità e maschera di focus)
            # Aree in primo piano e punti di interesse hanno peso maggiore
            focus_weight = normalized_focus[y, x]
            depth_weight = 1.0 - normalized_depth[y, x]
            
            # Combinazione ponderata
            weight = focus_weight * 0.7 + depth_weight * 0.3
            
            # Scegli l'indice di sfocatura in base al peso
            blur_idx = min(blur_steps - 1, int(weight * blur_steps))
            
            # Interpola tra l'originale e la versione sfocata appropriata
            if len(img_array.shape) == 3:
                # Immagine a colori
                result[y, x] = (1.0 - weight) * blurred_images[blur_idx][y, x] + weight * img_array[y, x]
            else:
                # Immagine in scala di grigi
                result[y, x] = (1.0 - weight) * blurred_images[blur_idx][y, x] + weight * img_array[y, x]
    
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

def create_filename(base_name, blur_min, blur_max, random, seed):
    """Crea nome file con i parametri inclusi"""
    b_min = f"{blur_min:.1f}".replace('.', '_')
    b_max = f"{blur_max:.1f}".replace('.', '_')
    r_str = f"{random:.2f}".replace('.', '_')
    seed_str = f"{seed % 10000:04d}"
    return f"{base_name}_b{b_min}-{b_max}_r{r_str}_s{seed_str}.jpg"

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
        min_value=5,
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
                        seed
                    )
                    
                    # Mostra risultato
                    st.image(result, use_container_width=True)
                    
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
                        used_seed
                    )
                    
                    # Mostra informazioni e link download
                    st.caption(f"Seed: {used_seed}")
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
                    
                    # Crea 4 varianti
                    variant_cols1 = st.columns(2)
                    variant_cols2 = st.columns(2)
                    
                    for i, col in enumerate([*variant_cols1, *variant_cols2]):
                        with col:
                            # Usa un seed diverso per ogni variante
                            variant_seed = seed + i if seed else random.randint(1, 9999)
                            variant_randomness = randomness * random.uniform(0.8, 1.2)
                            
                            # Applica l'effetto
                            result, _, _, used_seed = simulate_depth_map(
                                combined_img,
                                (min_blur, max_blur),
                                variant_randomness,
                                variant_seed
                            )
                            
                            # Mostra risultato
                            st.image(result, use_container_width=True)
                            
                            # Crea nome file
                            if len(uploaded_files) == 1:
                                base_name = os.path.splitext(uploaded_files[0].name)[0]
                            else:
                                base_name = f"combined_{len(uploaded_files)}_images"
                            
                            filename = create_filename(
                                f"{base_name}_variant_{i+1}",
                                min_blur,
                                max_blur,
                                variant_randomness,
                                used_seed
                            )
                            
                            # Mostra info e link download
                            st.caption(f"Seed: {used_seed}")
                            st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                    
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
            
            4. **Stacking di immagini**: Se carichi più immagini, vengono combinate con leggere variazioni
            
            ### Suggerimenti:
            
            - Carica immagini con un chiaro soggetto e sfondo distinguibile
            - Prova diverse impostazioni di casualità per ottenere effetti variabili
            - Usa il pulsante "Genera varianti" per creare rapidamente diverse versioni
            - Se carichi più immagini simili, l'algoritmo creerà un effetto di stacking
            """)
