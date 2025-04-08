import streamlit as st
import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageChops, ImageEnhance
import io
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Selective Focus Stacking",
    page_icon="🔍",
    layout="wide"
)

# Funzioni di utilità per il processing delle immagini
def apply_focus_mask(img, center_x, center_y, focus_size, transition_size):
    """Crea e applica una maschera di focus con centro e dimensioni specificate"""
    width, height = img.size
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Disegna un'ellisse piena come area di focus
    draw.ellipse(
        (center_x - focus_size/2, center_y - focus_size/2,
         center_x + focus_size/2, center_y + focus_size/2),
        fill=255
    )
    
    # Sfuma i bordi della maschera per una transizione graduale
    mask = mask.filter(ImageFilter.GaussianBlur(radius=transition_size))
    
    return mask

def add_noise_to_mask(mask, intensity=0.5):
    """Aggiunge rumore casuale alla maschera per un effetto più naturale"""
    if intensity <= 0:
        return mask
        
    mask_array = np.array(mask)
    noise = np.random.normal(0, 25 * intensity, mask_array.shape)
    mask_array = np.clip(mask_array + noise, 0, 255).astype(np.uint8)
    return Image.fromarray(mask_array)

def apply_ghost_effect(img, mask, threshold=0.5, intensity=0.7):
    """Applica un effetto 'ghost' creativo alle aree fuori fuoco"""
    # Converti in RGBA per l'elaborazione
    original_mode = img.mode
    img = img.convert('RGBA')
    
    # Crea una copia per il risultato
    result = img.copy()
    
    # Converti in array per velocizzare l'elaborazione
    img_array = np.array(img)
    mask_array = np.array(mask)
    
    # Area fuori fuoco in base alla threshold
    out_of_focus = mask_array < (255 * threshold)
    
    # Calcola la versione desaturata (bianco e nero)
    r, g, b, a = img_array[:, :, 0], img_array[:, :, 1], img_array[:, :, 2], img_array[:, :, 3]
    gray = (r * 0.299 + g * 0.587 + b * 0.114).astype(np.uint8)
    
    # Effetto di desaturazione parziale
    r_new = r.copy()
    g_new = g.copy()
    b_new = b.copy()
    
    # Applica l'effetto desaturato solo alle aree fuori fuoco
    r_new[out_of_focus] = (r[out_of_focus] * (1 - intensity) + gray[out_of_focus] * intensity).astype(np.uint8)
    g_new[out_of_focus] = (g[out_of_focus] * (1 - intensity) + gray[out_of_focus] * intensity).astype(np.uint8)
    b_new[out_of_focus] = (b[out_of_focus] * (1 - intensity) + gray[out_of_focus] * intensity).astype(np.uint8)
    
    # Aggiungi una leggera tinta blu alle aree fuori fuoco per un effetto ethereo
    b_new[out_of_focus] = np.minimum(255, (b_new[out_of_focus] * 1.05)).astype(np.uint8)
    
    # Ricostruisci l'immagine
    result_array = np.stack([r_new, g_new, b_new, a], axis=2)
    result = Image.fromarray(result_array)
    
    # Ripristina il formato originale se necessario
    if original_mode == 'RGB':
        result = result.convert('RGB')
    
    return result

def stack_images(images, focus_ratio=0.3, blur_strength=0.7, randomness=0.5, ghost_threshold=0.5, seed=None):
    """
    Applica un effetto di stacking con focus selettivo su un insieme di immagini.
    
    Parameters:
    images (list): Lista di oggetti PIL.Image
    focus_ratio (float): Controlla la dimensione dell'area a fuoco (0.1-0.9)
    blur_strength (float): Intensità della sfocatura (0.1-1.0)
    randomness (float): Casualità nella posizione e forma dell'area a fuoco (0.0-1.0)
    ghost_threshold (float): Soglia per l'effetto ghost (0.0-1.0)
    seed (int, optional): Seed per la generazione casuale
    
    Returns:
    PIL.Image: Immagine risultante con effetto di stacking
    """
    if not images or len(images) == 0:
        return None
    
    # Se c'è una sola immagine, applica un effetto di focus selettivo semplice
    if len(images) == 1:
        return apply_single_image_effect(images[0], focus_ratio, blur_strength, randomness, ghost_threshold, seed)
    
    # Imposta il seed casuale
    if seed is None:
        seed = random.randint(1, 999999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Assicurati che tutte le immagini abbiano la stessa dimensione (usa la prima come riferimento)
    base_img = images[0]
    width, height = base_img.size
    
    # Dimensione del focus e posizione centrale di base
    focus_size = min(width, height) * (1.0 - focus_ratio * 0.8)
    base_center_x, base_center_y = width // 2, height // 2
    
    # Aggiungi casualità alla posizione del focus
    if randomness > 0:
        base_center_x += int(random.uniform(-width * 0.15, width * 0.15) * randomness)
        base_center_y += int(random.uniform(-height * 0.15, height * 0.15) * randomness)
    
    # Prepara l'immagine risultato
    result = None
    
    # Per ogni immagine nello stack
    for i, img in enumerate(images):
        # Ridimensiona l'immagine se necessario
        if img.size != (width, height):
            img = img.resize((width, height), Image.LANCZOS)
        
        # Crea una copia dell'immagine per l'elaborazione
        processed = img.copy()
        
        # Calcola il centro del focus per questa immagine con leggere variazioni
        if i > 0 and randomness > 0:
            center_x = base_center_x + int(random.uniform(-width * 0.05, width * 0.05) * randomness * i)
            center_y = base_center_y + int(random.uniform(-height * 0.05, height * 0.05) * randomness * i)
            curr_focus_size = focus_size * random.uniform(0.9, 1.1)
        else:
            center_x, center_y = base_center_x, base_center_y
            curr_focus_size = focus_size
        
        # Transizione più ampia per le immagini successive dello stack
        transition_size = max(curr_focus_size / 10, curr_focus_size / (10 - i * 0.5))
        
        # Crea la maschera di focus
        mask = apply_focus_mask(img, center_x, center_y, curr_focus_size, transition_size)
        
        # Aggiungi rumore alla maschera se richiesto
        if randomness > 0:
            mask = add_noise_to_mask(mask, randomness * (1 + i * 0.1))
        
        # Applica sfocatura se non è nella zona di focus
        blur_radius = int(max(1, 5 + 5 * blur_strength * (i + 1) / len(images)))
        blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
        
        # Combina immagine originale e sfocata usando la maschera
        processed = Image.composite(img, blurred, mask)
        
        # Applica effetto ghost se richiesto
        if ghost_threshold > 0:
            ghost_value = ghost_threshold * (1 - 0.1 * i / len(images))  # Varia leggermente per ogni immagine
            processed = apply_ghost_effect(processed, mask, ghost_value)
        
        # Mescola con il risultato cumulativo
        if result is None:
            result = processed.convert('RGBA')
        else:
            # Varia l'opacità per ogni livello
            alpha = 1.0 - (i / len(images)) * 0.4
            processed = processed.convert('RGBA')
            
            # Crea una maschera per il blending
            blend_mask = Image.new('L', (width, height), int(255 * alpha))
            result = Image.composite(processed, result, blend_mask)
    
    # Converti il risultato finale in RGB
    if result.mode == 'RGBA':
        result = result.convert('RGB')
    
    return result

def apply_single_image_effect(img, focus_ratio=0.3, blur_strength=0.7, randomness=0.5, ghost_threshold=0.5, seed=None):
    """Applica un effetto di focus selettivo a una singola immagine"""
    # Imposta il seed casuale
    if seed is None:
        seed = random.randint(1, 999999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Assicurati che l'immagine sia in RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    # Calcola la posizione del focus con casualità
    center_x, center_y = width // 2, height // 2
    focus_size = min(width, height) * (1.0 - focus_ratio * 0.8)
    
    # Aggiungi casualità alla posizione
    if randomness > 0:
        center_x += int(random.uniform(-width * 0.2, width * 0.2) * randomness)
        center_y += int(random.uniform(-height * 0.2, height * 0.2) * randomness)
    
    # Crea l'effetto di sfocatura
    blur_radius = int(max(1, 10 * blur_strength))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Crea la maschera di focus
    transition_size = max(1, focus_size / 10)
    mask = apply_focus_mask(img, center_x, center_y, focus_size, transition_size)
    
    # Aggiungi rumore alla maschera
    if randomness > 0:
        mask = add_noise_to_mask(mask, randomness)
    
    # Combina l'immagine originale e quella sfocata
    result = Image.composite(img, blurred, mask)
    
    # Applica l'effetto ghost
    if ghost_threshold > 0:
        result = apply_ghost_effect(result, mask, ghost_threshold)
    
    return result

def get_image_download_link(img, filename):
    """Genera un link per scaricare un'immagine"""
    buffered = io.BytesIO()
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" class="download-button">📥 Scarica</a>'
    return href

# Main Streamlit app
def main():
    st.title("🔍 Selective Focus Stacking")
    st.markdown("Crea effetti di messa a fuoco selettiva con stacking di immagini")
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
            help="Carica una o più immagini. Se carichi più immagini, verranno elaborate con stacking."
        )
        
        # Modalità
        mode = st.radio(
            "Modalità:",
            ["Standard", "Esplora varianti", "Campiona stili"]
        )
        
        # Parametri
        st.subheader("Parametri")
        
        focus_ratio = st.slider(
            "Area a fuoco:",
            min_value=0.1,
            max_value=0.9,
            value=0.3,
            step=0.05,
            help="Controlla la dimensione dell'area a fuoco (valori più alti = area più piccola)"
        )
        
        blur_strength = st.slider(
            "Intensità sfocatura:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Intensità dell'effetto sfocato nelle aree fuori fuoco"
        )
        
        randomness = st.slider(
            "Casualità:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Aggiunge casualità alla posizione e forma dell'area a fuoco"
        )
        
        ghost_threshold = st.slider(
            "Effetto Ghost:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Controlla l'intensità dell'effetto bianco e nero nelle zone sfocate"
        )
        
        # Seed
        seed_option = st.radio(
            "Seed:",
            ["Casuale", "Specifico"]
        )
        
        seed = None
        if seed_option == "Specifico":
            seed = st.number_input(
                "Valore seed:",
                min_value=1,
                max_value=999999,
                value=42
            )
        
        # Pulsante elabora
        process_button = st.button(
            "Elabora immagini",
            type="primary",
            use_container_width=True
        )
    
    with col2:
        if uploaded_files:
            # Mostra le immagini caricate
            st.subheader("Immagini caricate")
            
            image_cols = st.columns(min(4, len(uploaded_files)))
            images = []
            
            for i, file in enumerate(uploaded_files):
                with image_cols[i % len(image_cols)]:
                    img = Image.open(file)
                    images.append(img)
                    st.image(img, width=150, caption=file.name)
            
            # Elabora le immagini quando viene premuto il pulsante
            if process_button:
                st.markdown("---")
                st.subheader("Risultati")
                
                # Tracciamento del progresso
                progress_bar = st.progress(0)
                status = st.empty()
                
                # Elabora le immagini in base alla modalità
                if mode == "Standard":
                    status.text("Elaborazione con stacking...")
                    
                    # Usa tutte le immagini caricate per lo stacking
                    result = stack_images(
                        images,
                        focus_ratio,
                        blur_strength,
                        randomness,
                        ghost_threshold,
                        seed
                    )
                    
                    if result:
                        # Mostra il risultato
                        st.image(result, use_container_width=True)
                        
                        # Crea link per il download
                        current_seed = seed if seed else random.randint(1, 999999)
                        filename = f"stacked_focus_seed_{current_seed}.jpg"
                        st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                        
                        # Mostra informazioni sul seed
                        st.caption(f"Seed utilizzato: {current_seed}")
                
                elif mode == "Esplora varianti":
                    # Genera 4 varianti con diversi seed
                    variants = 4
                    seeds = []
                    results = []
                    
                    # Genera i seed se necessario
                    base_seed = seed if seed else random.randint(1, 999999)
                    for i in range(variants):
                        seeds.append(base_seed + i if seed else random.randint(1, 999999))
                    
                    # Elabora ogni variante
                    for i, current_seed in enumerate(seeds):
                        status.text(f"Elaborazione variante {i+1}/{variants}...")
                        
                        result = stack_images(
                            images,
                            focus_ratio,
                            blur_strength,
                            randomness,
                            ghost_threshold,
                            current_seed
                        )
                        
                        if result:
                            results.append((result, current_seed))
                        
                        progress_bar.progress((i + 1) / variants)
                    
                    # Mostra i risultati in una griglia 2x2
                    if results:
                        for row in range(2):
                            cols = st.columns(2)
                            for col in range(2):
                                idx = row * 2 + col
                                if idx < len(results):
                                    with cols[col]:
                                        result, current_seed = results[idx]
                                        st.image(result, use_container_width=True)
                                        st.caption(f"Seed: {current_seed}")
                                        
                                        # Link per il download
                                        filename = f"variant_{idx+1}_seed_{current_seed}.jpg"
                                        st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                
                elif mode == "Campiona stili":
                    # Crea 6 preset di stili diversi
                    presets = [
                        {"name": "Soft Focus", "focus": 0.2, "blur": 0.5, "random": 0.3, "ghost": 0.4},
                        {"name": "Dramatic", "focus": 0.6, "blur": 0.9, "random": 0.4, "ghost": 0.6},
                        {"name": "Wide Area", "focus": 0.1, "blur": 0.4, "random": 0.5, "ghost": 0.3},
                        {"name": "Tight Focus", "focus": 0.7, "blur": 0.8, "random": 0.4, "ghost": 0.5},
                        {"name": "Dreamy", "focus": 0.3, "blur": 0.6, "random": 0.7, "ghost": 0.3},
                        {"name": "Creative", "focus": 0.4, "blur": 0.7, "random": 0.8, "ghost": 0.2}
                    ]
                    
                    # Usa lo stesso seed per tutti gli stili
                    current_seed = seed if seed else random.randint(1, 999999)
                    results = []
                    
                    # Elabora ogni stile
                    for i, preset in enumerate(presets):
                        status.text(f"Elaborazione stile {i+1}/{len(presets)}...")
                        
                        result = stack_images(
                            images,
                            preset["focus"],
                            preset["blur"],
                            preset["random"],
                            preset["ghost"],
                            current_seed
                        )
                        
                        if result:
                            results.append((result, preset["name"]))
                        
                        progress_bar.progress((i + 1) / len(presets))
                    
                    # Mostra i risultati in una griglia 2x3
                    if results:
                        for row in range(2):
                            cols = st.columns(3)
                            for col in range(3):
                                idx = row * 3 + col
                                if idx < len(results):
                                    with cols[col]:
                                        result, style_name = results[idx]
                                        st.image(result, use_container_width=True)
                                        st.caption(f"{style_name} (Seed: {current_seed})")
                                        
                                        # Link per il download
                                        filename = f"{style_name.lower().replace(' ', '_')}_seed_{current_seed}.jpg"
                                        st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                
                # Pulisci i controlli di progresso
                progress_bar.empty()
                status.empty()
                
                # Messaggio finale
                st.success("✅ Elaborazione completata!")
        else:
            # Messaggio quando non ci sono immagini caricate
            st.info("👈 Carica una o più immagini per iniziare.")
            
            with st.expander("Informazioni su Selective Focus Stacking"):
                st.markdown("""
                ## Cos'è Selective Focus Stacking?
                
                Questa applicazione ti permette di creare effetti artistici di messa a fuoco selettiva sulle tue immagini, 
                combinando più immagini (stacking) per ottenere effetti creativi dove solo alcune parti dell'immagine 
                sono a fuoco mentre il resto è sfocato con un effetto "ghost".
                
                ### Come funziona:
                
                1. **Con una singola immagine:** Verrà applicato un effetto di messa a fuoco selettiva con posizionamento casuale dell'area a fuoco
                
                2. **Con più immagini:** Le immagini verranno combinate (stacking) mantenendo a fuoco diverse aree in base ai parametri specificati
                
                ### Modalità disponibili:
                
                - **Standard:** Crea una singola immagine con i parametri specificati
                - **Esplora varianti:** Genera 4 variazioni con diversi seed casuali
                - **Campiona stili:** Applica 6 stili predefiniti alle tue immagini
                
                ### Parametri:
                
                - **Area a fuoco:** Controlla la dimensione dell'area a fuoco
                - **Intensità sfocatura:** Quanto saranno sfocate le aree fuori fuoco
                - **Casualità:** Aggiunge variazione alla posizione dell'area a fuoco
                - **Effetto Ghost:** Controlla l'intensità dell'effetto in bianco e nero nelle zone sfocate
                
                ### Seed:
                
                Il seed è un valore numerico che garantisce risultati consistenti quando si usa lo stesso seed.
                """)

if __name__ == "__main__":
    main()
