import streamlit as st
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance
import io
import base64
from datetime import datetime
from skimage.feature import peak_local_max
from skimage.segmentation import watershed
from skimage.measure import regionprops
from scipy import ndimage

# Set page configuration
st.set_page_config(
    page_title="AI Selective Focus",
    page_icon="🔍",
    layout="wide"
)

def detect_interesting_regions(img, num_regions=5, randomness=0.5, min_size=0.01, max_size=0.1, seed=None):
    """
    Utilizza tecniche di computer vision per identificare regioni interessanti nell'immagine
    """
    # Imposta seed per riproducibilità
    if seed is None:
        seed = random.randint(1, 9999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Converti l'immagine PIL in formato OpenCV
    img_cv = np.array(img)
    if len(img_cv.shape) == 3:
        img_gray = cv2.cvtColor(img_cv, cv2.COLOR_RGB2GRAY)
    else:
        img_gray = img_cv.copy()
    
    # Dimensioni dell'immagine
    height, width = img_gray.shape
    total_pixels = height * width
    min_region_size = int(min_size * total_pixels)
    max_region_size = int(max_size * total_pixels)
    
    # 1. Rileva bordi utilizzando Canny
    edges = cv2.Canny(img_gray, 50, 150)
    
    # 2. Calcola la mappa di saliency (regioni visivamente importanti)
    saliency = cv2.saliency.StaticSaliencyFineGrained_create()
    success, saliency_map = saliency.computeSaliency(img_cv)
    if success:
        saliency_map = (saliency_map * 255).astype("uint8")
    else:
        # Fallback in caso di errore
        saliency_map = img_gray.copy()
    
    # 3. Calcola la mappa di dettagli (variazioni locali)
    blurred = cv2.GaussianBlur(img_gray, (21, 21), 0)
    detail_map = cv2.absdiff(img_gray, blurred)
    
    # 4. Combina le mappe (bordi, saliency, dettagli)
    combined_map = cv2.addWeighted(
        cv2.addWeighted(edges, 0.3, saliency_map, 0.3, 0),
        0.5, detail_map, 0.5, 0
    )
    
    # 5. Trova i massimi locali come punti di interesse
    # Aggiungi un po' di rumore per aumentare la casualità se richiesto
    if randomness > 0:
        noise = np.random.normal(0, randomness * 30, combined_map.shape).astype(np.uint8)
        combined_map = cv2.add(combined_map, noise)
    
    # Trova i massimi locali
    distance = ndimage.distance_transform_edt(combined_map)
    coords = peak_local_max(distance, min_distance=30, num_peaks=min(20, num_regions*3))
    
    # 6. Seleziona regioni casuali tra quelle trovate
    if len(coords) > 0:
        # Seleziona un sottoinsieme casuale di regioni
        indices = random.sample(range(len(coords)), min(num_regions, len(coords)))
        selected_coords = [coords[i] for i in indices]
        
        # Crea la maschera con le regioni interessanti
        mask = np.zeros_like(img_gray)
        
        # Per ogni punto di interesse, crea una regione di focus
        regions = []
        for y, x in selected_coords:
            # Dimensione variabile per ogni regione basata sul dettaglio locale
            local_detail = detail_map[y, x]
            size_factor = 0.5 + local_detail / 255
            
            # Dimensione base variabile con un po' di casualità
            base_size = random.uniform(
                min(width, height) * 0.05, 
                min(width, height) * 0.15
            ) * size_factor
            
            # Aggiungi casualità alla dimensione
            if randomness > 0:
                base_size *= random.uniform(1 - randomness * 0.5, 1 + randomness * 0.5)
            
            # Controlla che la dimensione sia nel range accettabile
            region_size = min(max_region_size, max(min_region_size, int(base_size)))
            
            # Disegna una regione ellittica o irregolare
            if random.random() < 0.7:  # 70% ellissi
                # Ellisse con assi variabili
                a = region_size * random.uniform(0.8, 1.2)
                b = region_size * random.uniform(0.8, 1.2)
                angle = random.uniform(0, 360)  # Rotazione casuale
                
                # Crea una ROI temporanea per l'ellisse
                temp_mask = np.zeros_like(img_gray)
                cv2.ellipse(
                    temp_mask, 
                    (x, y), 
                    (int(a), int(b)), 
                    angle, 0, 360, 255, -1
                )
                
                # Aggiungi alla maschera principale
                mask = cv2.bitwise_or(mask, temp_mask)
                
                regions.append({
                    'center': (x, y),
                    'type': 'ellipse',
                    'size': (a, b),
                    'angle': angle
                })
            else:  # 30% forme più irregolari
                # Crea una forma irregolare
                points = []
                num_points = random.randint(5, 10)
                for i in range(num_points):
                    angle = 2 * np.pi * i / num_points
                    r = region_size * random.uniform(0.7, 1.3)
                    px = x + int(r * np.cos(angle))
                    py = y + int(r * np.sin(angle))
                    # Assicurati che i punti siano all'interno dell'immagine
                    px = max(0, min(width-1, px))
                    py = max(0, min(height-1, py))
                    points.append((px, py))
                
                # Disegna il poligono
                temp_mask = np.zeros_like(img_gray)
                cv2.fillPoly(temp_mask, [np.array(points)], 255)
                
                # Aggiungi alla maschera principale
                mask = cv2.bitwise_or(mask, temp_mask)
                
                regions.append({
                    'center': (x, y),
                    'type': 'polygon',
                    'points': points
                })
        
        # Applica sfocatura gaussiana alla maschera per ammorbidire i bordi
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return Image.fromarray(mask), regions, seed
    else:
        # Fallback: se non vengono trovate regioni, crea punti casuali
        mask = np.zeros_like(img_gray)
        regions = []
        
        for _ in range(num_regions):
            x = random.randint(0, width-1)
            y = random.randint(0, height-1)
            radius = random.randint(
                int(min(width, height) * 0.05),
                int(min(width, height) * 0.15)
            )
            cv2.circle(mask, (x, y), radius, 255, -1)
            regions.append({
                'center': (x, y),
                'type': 'circle',
                'radius': radius
            })
        
        # Sfoca i bordi
        mask = cv2.GaussianBlur(mask, (21, 21), 0)
        
        return Image.fromarray(mask), regions, seed

def apply_selective_focus(img, focus_ratio=0.3, blur_strength=0.7, randomness=0.5, 
                         ghost_threshold=0.5, num_regions=3, seed=None):
    """
    Applica selective focus basato su regioni interessanti rilevate con AI
    """
    # Rileva regioni interessanti con computer vision
    focus_mask, regions, used_seed = detect_interesting_regions(
        img, num_regions, randomness, 
        min_size=0.02*(1-focus_ratio), 
        max_size=0.1*(1-focus_ratio),
        seed=seed
    )
    
    # Crea versione sfocata dell'immagine
    blur_radius = int(max(1, 15 * blur_strength))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Converti la maschera in formato PIL compatibile
    focus_mask = focus_mask.convert("L")
    
    # Crea l'immagine risultante utilizzando la maschera
    result = Image.composite(img, blurred, focus_mask)
    
    # Applica effetto ghost (BW) alle aree fuori fuoco
    if ghost_threshold > 0:
        result = apply_ghost_effect(img, blurred, focus_mask, ghost_threshold)
    
    return result, used_seed, regions

def apply_ghost_effect(orig_img, blurred_img, mask, threshold=0.5):
    """
    Applica effetto ghost alle aree fuori focus:
    Aree a fuoco: colori originali
    Aree sfocate: bianco e nero con contrasto aumentato
    """
    # Converti in array numpy
    img_array = np.array(orig_img)
    blurred_array = np.array(blurred_img)
    mask_array = np.array(mask)
    
    # Crea versione B&W dell'immagine sfocata con contrasto aumentato
    if len(blurred_array.shape) == 3:
        # Immagine a colori
        gray = cv2.cvtColor(blurred_array, cv2.COLOR_RGB2GRAY)
        # Aumenta contrasto
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        enhanced_gray = clahe.apply(gray)
        # Espandi a 3 canali mantenendo grigio
        bw_enhanced = cv2.cvtColor(enhanced_gray, cv2.COLOR_GRAY2RGB)
    else:
        # Immagine già in scala di grigi
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        bw_enhanced = clahe.apply(blurred_array)
    
    # Crea maschera binaria basata sulla threshold
    focus_binary = mask_array > (255 * threshold)
    
    # Crea array risultante
    result_array = np.copy(img_array)
    
    # Sostituisci le parti fuori fuoco con versione B&W
    if len(img_array.shape) == 3:
        # Immagine a colori
        for c in range(3):  # Per ogni canale RGB
            result_array[:,:,c] = np.where(focus_binary, img_array[:,:,c], bw_enhanced[:,:,c])
    else:
        # Immagine in scala di grigi
        result_array = np.where(focus_binary, img_array, bw_enhanced)
    
    return Image.fromarray(result_array)

def get_image_download_link(img, filename):
    """Genera un link per scaricare un'immagine"""
    buffered = io.BytesIO()
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" class="download-button">📥 Scarica</a>'
    return href

def create_filename(base_name, focus, blur, random, ghost, seed):
    """Crea nome file con i parametri inclusi"""
    # Formatta i parametri con 2 decimali
    f_str = f"{focus:.2f}".replace('.', '_')
    b_str = f"{blur:.2f}".replace('.', '_')
    r_str = f"{random:.2f}".replace('.', '_')
    g_str = f"{ghost:.2f}".replace('.', '_')
    
    # Formatta il seed come 4 cifre
    seed_str = f"{seed % 10000:04d}"
    
    return f"{base_name}_f{f_str}_b{b_str}_r{r_str}_g{g_str}_s{seed_str}.jpg"

# Main Streamlit app
def main():
    st.title("🔍 AI Selective Focus")
    st.markdown("Applica effetti di messa a fuoco selettiva usando AI per rilevare dettagli interessanti")
    st.markdown("---")
    
    # Carica CSS personalizzato se esiste
    css_file = os.path.join(os.path.dirname(__file__), "style.css")
    if os.path.exists(css_file):
        with open(css_file) as f:
            st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
    
    # Layout a colonne
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.header("Controlli")
        
        # Upload immagini
        uploaded_files = st.file_uploader(
            "Carica immagini",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True,
            help="Carica una o più immagini per elaborarle"
        )
        
        # Parametri
        st.subheader("Parametri")
        
        focus_ratio = st.slider(
            "Area a fuoco:",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Dimensione delle aree a fuoco (valori più alti = aree più piccole)"
        )
        
        blur_strength = st.slider(
            "Intensità sfocatura:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Quanto saranno sfocate le aree fuori fuoco"
        )
        
        randomness = st.slider(
            "Casualità:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Aggiunge variazione nella scelta delle aree a fuoco"
        )
        
        ghost_threshold = st.slider(
            "Effetto Ghost:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Intensità dell'effetto bianco e nero nelle aree sfocate"
        )
        
        num_regions = st.slider(
            "Numero di regioni:",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            help="Quante aree interessanti rilevare e mettere a fuoco"
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
        
        # Modalità
        st.subheader("Modalità")
        generate_variants = st.checkbox(
            "Genera varianti",
            value=False,
            help="Genera 4 diverse varianti con seed diversi"
        )
        
        show_focus_regions = st.checkbox(
            "Mostra regioni di focus",
            value=False,
            help="Visualizza le regioni che l'AI ha rilevato come interessanti"
        )
        
        # Pulsante elabora
        process_button = st.button(
            "Elabora immagini",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if uploaded_files:
            # Mostra anteprima immagini caricate
            st.subheader("Immagini caricate")
            
            # Mostra miniature in galleria
            image_cols = st.columns(min(4, len(uploaded_files)))
            images = []
            filenames = []
            
            for i, file in enumerate(uploaded_files):
                with image_cols[i % len(image_cols)]:
                    img = Image.open(file)
                    images.append(img)
                    filenames.append(os.path.splitext(file.name)[0])
                    st.image(img, width=150, caption=file.name)
            
            # Elabora immagini quando viene premuto il pulsante
            if process_button:
                st.markdown("---")
                st.subheader("Risultati")
                
                # Progress bar
                progress_bar = st.progress(0)
                status = st.empty()
                
                # Numero di varianti
                num_variants = 4 if generate_variants else 1
                
                # Elabora ogni immagine
                for img_index, (img, base_name) in enumerate(zip(images, filenames)):
                    status.text(f"Elaborazione di {base_name}...")
                    
                    # Se genera varianti, mostra in griglia
                    if generate_variants:
                        st.markdown(f"**{base_name} - Varianti**")
                        
                        # Crea griglia 2x2
                        for row in range(2):
                            cols = st.columns(2)
                            for col in range(2):
                                variant_idx = row * 2 + col
                                variant_seed = seed + variant_idx if seed else random.randint(1, 9999)
                                
                                with cols[col]:
                                    # Applica effetto con rilevamento AI
                                    result, used_seed, regions = apply_selective_focus(
                                        img,
                                        focus_ratio,
                                        blur_strength,
                                        randomness,
                                        ghost_threshold,
                                        num_regions,
                                        variant_seed
                                    )
                                    
                                    # Mostra risultato
                                    st.image(result, use_container_width=True)
                                    
                                    # Se richiesto, mostra le regioni rilevate
                                    if show_focus_regions:
                                        # Crea visualizzazione delle regioni
                                        regions_img = img.copy()
                                        draw = ImageDraw.Draw(regions_img)
                                        
                                        for region in regions:
                                            if region['type'] == 'ellipse':
                                                # Disegna ellisse
                                                x, y = region['center']
                                                a, b = region['size']
                                                # Disegna bordo ellisse in rosso
                                                bbox = (
                                                    x - a, y - b,
                                                    x + a, y + b
                                                )
                                                draw.ellipse(bbox, outline="red", width=3)
                                            elif region['type'] == 'polygon':
                                                # Disegna poligono
                                                draw.polygon(region['points'], outline="red", width=3)
                                            elif region['type'] == 'circle':
                                                # Disegna cerchio
                                                x, y = region['center']
                                                r = region['radius']
                                                bbox = (
                                                    x - r, y - r,
                                                    x + r, y + r
                                                )
                                                draw.ellipse(bbox, outline="red", width=3)
                                        
                                        # Mostra immagine con regioni evidenziate
                                        st.image(regions_img, caption="Regioni rilevate", use_container_width=True)
                                    
                                    # Crea nome file con parametri
                                    filename = create_filename(
                                        f"{base_name}_variant_{variant_idx+1}",
                                        focus_ratio,
                                        blur_strength,
                                        randomness,
                                        ghost_threshold,
                                        used_seed
                                    )
                                    
                                    # Mostra info e link download
                                    st.caption(f"Seed: {used_seed}")
                                    st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                    else:
                        # Elabora singola variante
                        result, used_seed, regions = apply_selective_focus(
                            img,
                            focus_ratio,
                            blur_strength,
                            randomness,
                            ghost_threshold,
                            num_regions,
                            seed
                        )
                        
                        # Mostra risultato
                        st.markdown(f"**{base_name}**")
                        st.image(result, use_container_width=True)
                        
                        # Se richiesto, mostra le regioni rilevate
                        if show_focus_regions:
                            # Crea visualizzazione delle regioni
                            regions_img = img.copy()
                            draw = ImageDraw.Draw(regions_img)
                            
                            for region in regions:
                                if region['type'] == 'ellipse':
                                    # Disegna ellisse
                                    x, y = region['center']
                                    a, b = region['size']
                                    # Disegna bordo ellisse in rosso
                                    bbox = (
                                        x - a, y - b,
                                        x + a, y + b
                                    )
                                    draw.ellipse(bbox, outline="red", width=3)
                                elif region['type'] == 'polygon':
                                    # Disegna poligono
                                    draw.polygon(region['points'], outline="red", width=3)
                                elif region['type'] == 'circle':
                                    # Disegna cerchio
                                    x, y = region['center']
                                    r = region['radius']
                                    bbox = (
                                        x - r, y - r,
                                        x + r, y + r
                                    )
                                    draw.ellipse(bbox, outline="red", width=3)
                            
                            # Mostra immagine con regioni evidenziate
                            st.image(regions_img, caption="Regioni rilevate", use_container_width=True)
                        
                        # Crea nome file con parametri
                        filename = create_filename(
                            base_name,
                            focus_ratio,
                            blur_strength,
                            randomness,
                            ghost_threshold,
                            used_seed
                        )
                        
                        # Mostra info e link download
                        st.caption(f"Parametri: Focus={focus_ratio}, Blur={blur_strength}, Random={randomness}, Ghost={ghost_threshold}, Seed={used_seed}")
                        st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                    
                    # Aggiorna progress bar
                    progress_bar.progress((img_index + 1) / len(images))
                    
                    # Linea di separazione tra immagini
                    if img_index < len(images) - 1:
                        st.markdown("---")
                
                # Pulisci status
                status.empty()
                progress_bar.empty()
                
                # Messaggio finale
                st.success("✅ Elaborazione completata!")
        else:
            # Messaggio iniziale
            st.info("👈 Carica una o più immagini per iniziare.")
            
            # Info sull'app
            with st.expander("Informazioni sull'app"):
                st.markdown("""
                ## AI Selective Focus
                
                Questa app utilizza tecniche di computer vision per identificare automaticamente elementi 
                interessanti nelle tue immagini e applicare effetti di messa a fuoco selettiva.
                
                ### Come funziona:
                
                1. **Rilevamento di regioni interessanti**: L'algoritmo analizza l'immagine per individuare 
                    dettagli, bordi, texture e aree visivamente rilevanti
                    
                2. **Selezione casuale**: Tra le regioni rilevate, l'app ne seleziona alcune in base ai parametri
                    specificati e alla casualità impostata
                    
                3. **Applicazione degli effetti**: Le regioni selezionate vengono mantenute a fuoco e a colori,
                    mentre il resto dell'immagine viene sfocato e convertito in bianco e nero
                
                ### Parametri:
                
                - **Area a fuoco**: Dimensione delle aree mantenute a fuoco
                - **Intensità sfocatura**: Quanto sfocate saranno le aree fuori fuoco
                - **Casualità**: Aggiunge variazione nella scelta e forma delle aree a fuoco
                - **Effetto Ghost**: Intensità dell'effetto bianco e nero nelle aree sfocate
                - **Numero di regioni**: Quante aree interessanti mettere a fuoco
                - **Seed**: Valore che garantisce risultati riproducibili
                
                ### Opzioni:
                
                - **Genera varianti**: Crea 4 varianti diverse dell'effetto
                - **Mostra regioni di focus**: Visualizza i contorni delle aree che l'AI ha rilevato
                """)

if __name__ == "__main__":
    main()
