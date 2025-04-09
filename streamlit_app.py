import streamlit as st
import os
import random
import numpy as np
import cv2
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageChops
import io
import base64
from datetime import datetime
from scipy import ndimage

# Set page configuration
st.set_page_config(
    page_title="Selective Focus Stacking",
    page_icon="🔍",
    layout="wide"
)

def align_images(images):
    """
    Allinea più immagini (assumendo che siano già pressoché allineate)
    Ritorna le immagini allineate come array numpy
    """
    # Se c'è solo un'immagine, ritorna direttamente
    if len(images) <= 1:
        return [np.array(img) for img in images]
    
    # Converti tutte le immagini in formato numpy
    np_images = [np.array(img) for img in images]
    
    # Usa la prima immagine come riferimento
    reference = cv2.cvtColor(np_images[0], cv2.COLOR_RGB2GRAY)
    
    aligned_images = [np_images[0]]  # La prima è già allineata con sé stessa
    
    # Per ogni immagine dopo la prima
    for i in range(1, len(np_images)):
        # Converti in scala di grigi per l'allineamento
        gray = cv2.cvtColor(np_images[i], cv2.COLOR_RGB2GRAY)
        
        # Trova le caratteristiche da tracciare
        try:
            # Utilizza l'algoritmo ORB per trovare punti chiave e descrittori
            orb = cv2.ORB_create()
            keypoints1, descriptors1 = orb.detectAndCompute(reference, None)
            keypoints2, descriptors2 = orb.detectAndCompute(gray, None)
            
            # Crea il matcher
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            
            if descriptors1 is not None and descriptors2 is not None:
                # Trova le corrispondenze
                matches = bf.match(descriptors1, descriptors2)
                
                # Ordina le corrispondenze per distanza
                matches = sorted(matches, key=lambda x: x.distance)
                
                # Prendi solo le migliori corrispondenze
                good_matches = matches[:min(50, len(matches))]
                
                # Estrai le posizioni dei punti
                src_pts = np.float32([keypoints1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([keypoints2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                
                # Trova la trasformazione
                if len(good_matches) >= 4:
                    M, _ = cv2.findHomography(dst_pts, src_pts, cv2.RANSAC, 5.0)
                    
                    # Applica la trasformazione
                    h, w = reference.shape
                    aligned = cv2.warpPerspective(np_images[i], M, (w, h))
                    aligned_images.append(aligned)
                else:
                    # Non ci sono abbastanza corrispondenze, usa l'originale
                    aligned_images.append(np_images[i])
            else:
                # Non ci sono descrittori, usa l'originale
                aligned_images.append(np_images[i])
                
        except Exception as e:
            # In caso di errore, usa l'originale
            print(f"Errore nell'allineamento: {str(e)}")
            aligned_images.append(np_images[i])
    
    return aligned_images

def create_focus_mask(image, num_points=3, randomness=0.5, seed=None):
    """
    Crea una maschera con punti di focus casuali
    """
    # Imposta seed per risultati riproducibili
    if seed is None:
        seed = random.randint(1, 9999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Converti l'immagine in scala di grigi per l'analisi
    if len(image.shape) == 3:
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    else:
        gray = image.copy()
    
    height, width = gray.shape
    
    # Crea una maschera vuota
    mask = np.zeros_like(gray)
    
    # Rileva bordi e dettagli per trovare aree interessanti
    edges = cv2.Canny(gray, 50, 150)
    details = cv2.Laplacian(gray, cv2.CV_64F)
    details = np.uint8(np.absolute(details))
    
    # Combina bordi e dettagli
    interest_map = cv2.addWeighted(edges, 0.5, details, 0.5, 0)
    
    # Aggiungi randomness
    if randomness > 0:
        noise = np.random.normal(0, randomness * 30, interest_map.shape).astype(np.uint8)
        interest_map = cv2.add(interest_map, noise)
    
    # Trova massimi locali (punti di interesse)
    distance = ndimage.distance_transform_edt(interest_map)
    coords = peak_local_max(distance, min_distance=width//10, num_peaks=num_points*2)
    
    # Se non ci sono abbastanza punti, crea punti casuali
    if len(coords) < num_points:
        additional_needed = num_points - len(coords)
        for _ in range(additional_needed):
            y = random.randint(0, height-1)
            x = random.randint(0, width-1)
            coords = np.append(coords, [[y, x]], axis=0)
    
    # Seleziona un subset casuale di punti
    if len(coords) > num_points:
        indices = random.sample(range(len(coords)), num_points)
        selected_coords = coords[indices]
    else:
        selected_coords = coords
    
    # Per ogni punto di interesse, crea una regione di focus
    for y, x in selected_coords:
        # Dimensione casuale, influenzata dalla posizione nell'immagine
        size = random.uniform(min(width, height) * 0.1, min(width, height) * 0.2)
        size *= (1 + randomness * random.uniform(-0.5, 0.5))
        
        # Tipo di forma: ellisse o poligono
        if random.random() < 0.7:  # 70% ellissi
            # Crea un'ellisse
            a = size * random.uniform(0.8, 1.2)
            b = size * random.uniform(0.8, 1.2)
            angle = random.uniform(0, 360)
            
            cv2.ellipse(
                mask, 
                (int(x), int(y)), 
                (int(a), int(b)), 
                angle, 0, 360, 255, -1
            )
        else:  # 30% poligoni
            # Crea un poligono irregolare
            num_vertices = random.randint(5, 8)
            vertices = []
            
            for i in range(num_vertices):
                angle = 2 * np.pi * i / num_vertices
                r = size * random.uniform(0.8, 1.2)
                vx = x + r * np.cos(angle)
                vy = y + r * np.sin(angle)
                
                # Assicurati che i vertici siano all'interno dell'immagine
                vx = max(0, min(width-1, vx))
                vy = max(0, min(height-1, vy))
                vertices.append([vx, vy])
            
            cv2.fillPoly(mask, [np.array(vertices, dtype=np.int32)], 255)
    
    # Sfuma i bordi delle regioni
    mask = cv2.GaussianBlur(mask, (31, 31), 0)
    
    return mask

def peak_local_max(image, min_distance=1, num_peaks=None):
    """
    Trova i massimi locali in un'immagine
    Implementazione semplificata di skimage.feature.peak_local_max
    """
    # Dimensioni immagine
    height, width = image.shape
    
    # Crea array per i massimi
    peaks = []
    
    # Dimensione del kernel
    size = 2 * min_distance + 1
    
    # Padding dell'immagine
    padded = np.pad(image, min_distance, mode='constant')
    
    # Per ogni punto nell'immagine
    for y in range(min_distance, height + min_distance):
        for x in range(min_distance, width + min_distance):
            # Estrai la regione intorno al punto
            neighborhood = padded[y - min_distance:y + min_distance + 1,
                                  x - min_distance:x + min_distance + 1]
            
            # Se il punto è il massimo locale
            if padded[y, x] == np.max(neighborhood) and padded[y, x] > 0:
                peaks.append([y - min_distance, x - min_distance])
                
                # Se abbiamo abbastanza picchi, termina
                if num_peaks is not None and len(peaks) >= num_peaks:
                    return np.array(peaks)
    
    return np.array(peaks)

def stack_images(images, focus_ratio=0.3, blur_strength=0.7, randomness=0.5, num_regions=3, seed=None):
    """
    Sovrappone più immagini con effetti di focus selettivo
    Mantiene i colori originali senza effetto bianco e nero
    """
    if not images or len(images) == 0:
        return None, None
    
    # Se c'è una sola immagine, applica l'effetto direttamente
    if len(images) == 1:
        img_array = np.array(images[0])
        mask = create_focus_mask(img_array, num_regions, randomness, seed)
        return apply_selective_focus_to_image(img_array, mask, blur_strength), seed
    
    # Imposta seed per riproducibilità
    if seed is None:
        seed = random.randint(1, 9999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Allinea le immagini
    aligned_images = align_images(images)
    
    # Usa le dimensioni della prima immagine come riferimento
    height, width = aligned_images[0].shape[:2]
    
    # Crea una maschera complessiva
    combined_mask = np.zeros((height, width), dtype=np.uint8)
    
    # Per ogni immagine, crea una maschera di focus e combinala
    for i, img in enumerate(aligned_images):
        # Dimensioni potrebbero variare dopo l'allineamento
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # Crea una maschera per questa immagine
        # Utilizziamo un seed diverso per ogni immagine ma derivato dal seed principale
        img_seed = seed + i * 1000
        mask = create_focus_mask(img, num_regions, randomness, img_seed)
        
        # Combina con la maschera complessiva
        # Usiamo un fade-out per le immagini successive nello stack
        weight = 1.0 - (i / len(aligned_images)) * 0.5
        mask_weighted = (mask * weight).astype(np.uint8)
        combined_mask = cv2.addWeighted(combined_mask, 1.0, mask_weighted, 1.0, 0)
    
    # Normalizza la maschera combinata
    combined_mask = cv2.normalize(combined_mask, None, 0, 255, cv2.NORM_MINMAX)
    
    # Crea l'immagine risultato come media ponderata delle immagini
    result = np.zeros_like(aligned_images[0], dtype=np.float32)
    
    for i, img in enumerate(aligned_images):
        # Normalizza le dimensioni
        if img.shape[:2] != (height, width):
            img = cv2.resize(img, (width, height))
        
        # Peso maggiore per le prime immagini nello stack
        weight = 1.0 - (i / len(aligned_images)) * 0.3
        result += img.astype(np.float32) * weight
    
    # Normalizza il risultato
    result = result / np.sum([1.0 - (i / len(aligned_images)) * 0.3 for i in range(len(aligned_images))])
    result = result.astype(np.uint8)
    
    # Applica l'effetto di focus selettivo utilizzando la maschera combinata
    final_result = apply_selective_focus_to_image(result, combined_mask, blur_strength)
    
    return final_result, seed

def apply_selective_focus_to_image(image, mask, blur_strength):
    """
    Applica l'effetto di focus selettivo a un'immagine usando una maschera
    Mantiene i colori originali, varia solo la nitidezza/sfocatura
    """
    # Crea una versione sfocata dell'immagine
    blur_radius = int(max(5, 20 * blur_strength))
    blurred = cv2.GaussianBlur(image, (blur_radius*2+1, blur_radius*2+1), 0)
    
    # Normalizza la maschera a valori tra 0 e 1
    normalized_mask = mask.astype(float) / 255.0
    
    # Crea maschera 3D se necessario
    if len(image.shape) == 3:
        mask_3d = np.stack([normalized_mask, normalized_mask, normalized_mask], axis=2)
    else:
        mask_3d = normalized_mask
    
    # Combina l'immagine originale e quella sfocata in base alla maschera
    # Nelle aree dove la maschera è 1, usa l'immagine originale
    # Nelle aree dove la maschera è 0, usa l'immagine sfocata
    result = image * mask_3d + blurred * (1 - mask_3d)
    
    return result.astype(np.uint8)

def get_image_download_link(img_array, filename):
    """Genera un link per scaricare un'immagine"""
    # Converti numpy array in immagine PIL
    img = Image.fromarray(img_array)
    
    # Salva in buffer
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" class="download-button">📥 Scarica</a>'
    return href

def create_filename(base_name, focus, blur, random, seed):
    """Crea nome file con i parametri inclusi"""
    # Formatta i parametri con 2 decimali
    f_str = f"{focus:.2f}".replace('.', '_')
    b_str = f"{blur:.2f}".replace('.', '_')
    r_str = f"{random:.2f}".replace('.', '_')
    
    # Formatta il seed come 4 cifre
    seed_str = f"{seed % 10000:04d}"
    
    return f"{base_name}_f{f_str}_b{b_str}_r{r_str}_s{seed_str}.jpg"

# Main Streamlit app
def main():
    st.title("🔍 Selective Focus Stacking")
    st.markdown("Combinazione di più immagini con effetto di messa a fuoco selettiva")
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
            help="Carica più immagini per lo stacking. Il miglior risultato si ottiene con 2-5 immagini simili."
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
        
        num_regions = st.slider(
            "Numero di regioni:",
            min_value=1,
            max_value=8,
            value=3,
            step=1,
            help="Quante aree interessanti mettere a fuoco per immagine"
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
            
            for i, file in enumerate(uploaded_files):
                with image_cols[i % len(image_cols)]:
                    img = Image.open(file)
                    # Ridimensiona le immagini troppo grandi per evitare problemi di memoria
                    if max(img.size) > 1200:
                        ratio = 1200 / max(img.size)
                        new_size = (int(img.size[0] * ratio), int(img.size[1] * ratio))
                        img = img.resize(new_size, Image.LANCZOS)
                    images.append(img)
                    st.image(img, width=150, caption=file.name)
            
            # Elabora immagini quando viene premuto il pulsante
            if process_button:
                st.markdown("---")
                st.subheader("Risultati dello stacking")
                
                # Progress bar
                progress_bar = st.progress(0)
                status = st.empty()
                
                # Crea un nome base per il file combinando i nomi delle immagini
                if len(uploaded_files) == 1:
                    base_name = os.path.splitext(uploaded_files[0].name)[0]
                else:
                    base_name = f"stacked_{len(uploaded_files)}_images"
                
                try:
                    if generate_variants:
                        st.markdown(f"**Varianti di stacking**")
                        
                        # Crea griglia 2x2 per le varianti
                        for row in range(2):
                            cols = st.columns(2)
                            for col in range(2):
                                variant_idx = row * 2 + col
                                variant_seed = seed + variant_idx if seed else random.randint(1, 9999)
                                
                                with cols[col]:
                                    status.text(f"Generazione variante {variant_idx+1}/4...")
                                    progress_bar.progress(0.1 + variant_idx * 0.2)
                                    
                                    # Applica stacking con focus selettivo
                                    result, used_seed = stack_images(
                                        images,
                                        focus_ratio,
                                        blur_strength,
                                        randomness,
                                        num_regions,
                                        variant_seed
                                    )
                                    
                                    # Mostra risultato
                                    st.image(result, use_container_width=True)
                                    
                                    # Crea nome file con parametri
                                    filename = create_filename(
                                        f"{base_name}_variant_{variant_idx+1}",
                                        focus_ratio,
                                        blur_strength,
                                        randomness,
                                        used_seed
                                    )
                                    
                                    # Mostra info e link download
                                    st.caption(f"Seed: {used_seed}")
                                    st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                    else:
                        # Elabora una sola versione
                        status.text("Elaborazione stacking in corso...")
                        progress_bar.progress(0.2)
                        
                        # Applica stacking con focus selettivo
                        result, used_seed = stack_images(
                            images,
                            focus_ratio,
                            blur_strength,
                            randomness,
                            num_regions,
                            seed
                        )
                        
                        # Aggiorna progress bar
                        progress_bar.progress(0.8)
                        
                        # Mostra risultato
                        st.image(result, use_container_width=True)
                        
                        # Crea nome file con parametri
                        filename = create_filename(
                            base_name,
                            focus_ratio,
                            blur_strength,
                            randomness,
                            used_seed
                        )
                        
                        # Mostra info e link download
                        st.caption(f"Parametri: Focus={focus_ratio}, Blur={blur_strength}, Random={randomness}, Seed={used_seed}")
                        st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                    
                    # Pulisci status
                    progress_bar.progress(1.0)
                    status.empty()
                    progress_bar.empty()
                    
                    # Messaggio finale
                    st.success("✅ Stacking completato con successo!")
                    
                except Exception as e:
                    st.error(f"Si è verificato un errore durante l'elaborazione: {str(e)}")
                    st.info("Prova a modificare i parametri o a caricare immagini diverse.")
                    progress_bar.empty()
                    status.empty()
        else:
            # Messaggio iniziale
            st.info("👈 Carica due o più immagini simili per iniziare lo stacking.")
            
            # Info sull'app
            with st.expander("Informazioni sull'app"):
                st.markdown("""
                ## Selective Focus Stacking
                
                Questa app combina più immagini applicando un effetto di messa a fuoco selettiva.
                
                ### Come funziona:
                
                1. **Stacking delle immagini**: L'app allinea e combina più immagini in una sola
                
                2. **Focus selettivo**: Vengono identificate aree interessanti che rimangono nitide, 
                   mentre il resto viene sfocato
                
                3. **Preservazione dei colori**: Tutti i colori originali vengono mantenuti in tutta l'immagine,
                   solo la nitidezza viene alterata
                
                ### Consigli per ottenere i migliori risultati:
                
                - Carica da 2 a 5 immagini simili (idealmente della stessa scena)
                - Le immagini dovrebbero essere già allineate o molto simili
                - Regola i parametri per ottenere l'effetto desiderato
                - Usa il seed specifico per riprodurre lo stesso effetto più volte
                
                ### Parametri:
                
                - **Area a fuoco**: Dimensione delle aree a fuoco
                - **Intensità sfocatura**: Quanto sfocate saranno le aree fuori fuoco
                - **Casualità**: Aggiunge variazione nella scelta delle aree a fuoco
                - **Numero di regioni**: Quante aree interessanti mettere a fuoco
                - **Seed**: Valore che garantisce risultati riproducibili
                """)

if __name__ == "__main__":
    main()
