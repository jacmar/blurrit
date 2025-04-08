import streamlit as st
import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw, ImageEnhance, ImageChops
import io
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Selective Focus Stacking",
    page_icon="🔍",
    layout="wide"
)

def apply_multiple_focus_points(img, num_points=3, focus_ratio=0.3, blur_strength=0.7, randomness=0.5, seed=None):
    """Applica più punti di focus casuali all'immagine"""
    # Imposta seed per risultati riproducibili
    if seed is None:
        seed = random.randint(1, 9999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Assicurati che l'immagine sia in RGB
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    width, height = img.size
    
    # Crea una maschera combinata
    combined_mask = Image.new('L', (width, height), 0)
    
    # Crea una versione sfocata dell'immagine
    blur_radius = int(max(1, 15 * blur_strength))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Crea diversi punti di focus casuali
    for i in range(num_points):
        # Determina posizione casuale del focus che NON sia al centro
        quadrant = random.randint(0, 3)  # 0=alto-sx, 1=alto-dx, 2=basso-sx, 3=basso-dx
        
        if quadrant == 0:  # Alto-sinistro
            center_x = random.randint(int(width * 0.1), int(width * 0.4))
            center_y = random.randint(int(height * 0.1), int(height * 0.4))
        elif quadrant == 1:  # Alto-destro
            center_x = random.randint(int(width * 0.6), int(width * 0.9))
            center_y = random.randint(int(height * 0.1), int(height * 0.4))
        elif quadrant == 2:  # Basso-sinistro
            center_x = random.randint(int(width * 0.1), int(width * 0.4))
            center_y = random.randint(int(height * 0.6), int(height * 0.9))
        else:  # Basso-destro
            center_x = random.randint(int(width * 0.6), int(width * 0.9))
            center_y = random.randint(int(height * 0.6), int(height * 0.9))
        
        # Aggiungi casualità aggiuntiva basata sul parametro randomness
        if randomness > 0:
            center_x += int(random.uniform(-width * 0.15, width * 0.15) * randomness)
            center_y += int(random.uniform(-height * 0.15, height * 0.15) * randomness)
        
        # Limita le coordinate all'interno dell'immagine
        center_x = max(0, min(width-1, center_x))
        center_y = max(0, min(height-1, center_y))
        
        # Dimensione del punto di focus
        # Varia la dimensione in base al parametro focus_ratio (più piccolo per valori più alti)
        focus_size = min(width, height) * (0.2 - focus_ratio * 0.1) * random.uniform(0.8, 1.2)
        
        # Crea una maschera per questo punto di focus
        point_mask = Image.new('L', (width, height), 0)
        draw = ImageDraw.Draw(point_mask)
        
        # Disegna un'ellisse
        draw.ellipse(
            (center_x - focus_size/2, center_y - focus_size/2,
             center_x + focus_size/2, center_y + focus_size/2),
            fill=255
        )
        
        # Sfuma i bordi per una transizione naturale
        transition_size = max(1, focus_size / 4)
        point_mask = point_mask.filter(ImageFilter.GaussianBlur(radius=transition_size))
        
        # Aggiungi rumore casuale alla maschera
        if randomness > 0:
            mask_array = np.array(point_mask)
            noise = np.random.normal(0, 20 * randomness, mask_array.shape)
            mask_array = np.clip(mask_array + noise, 0, 255).astype(np.uint8)
            point_mask = Image.fromarray(mask_array)
        
        # Combina con la maschera globale (usa il massimo per ogni pixel)
        combined_array = np.maximum(np.array(combined_mask), np.array(point_mask))
        combined_mask = Image.fromarray(combined_array)
    
    # Combina le immagini originale e sfocata usando la maschera combinata
    result = Image.composite(img, blurred, combined_mask)
    
    # Applica l'effetto ghost
    return apply_ghost_effect(img, blurred, combined_mask, 0.5), seed

def apply_ghost_effect(orig_img, blurred_img, mask, threshold=0.5):
    """
    Applica un effetto ghost più naturale: 
    - Aree a fuoco: Colori originali
    - Aree sfocate: Versione desaturata/b&w con leggera sfocatura
    """
    # Assicurati che le immagini siano in RGB
    orig_img = orig_img.convert('RGB')
    blurred_img = blurred_img.convert('RGB')
    
    # Crea una versione B&W dell'immagine sfocata
    bw_blurred = blurred_img.convert('L').convert('RGB')
    
    # Converti la maschera in array numpy per un controllo più preciso
    mask_array = np.array(mask)
    
    # Crea array booleano per le aree fuori fuoco (dove mask < threshold)
    out_of_focus = mask_array < (255 * threshold)
    
    # Converti immagini in array numpy
    orig_array = np.array(orig_img)
    bw_array = np.array(bw_blurred)
    
    # Crea array risultato partendo dall'immagine originale
    result_array = np.copy(orig_array)
    
    # Sostituisci le aree fuori fuoco con la versione B&W
    result_array[out_of_focus] = bw_array[out_of_focus]
    
    # Crea l'immagine risultato
    result = Image.fromarray(result_array)
    
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
    st.title("🔍 Selective Focus con Punti Multipli")
    st.markdown("Applica effetti di messa a fuoco selettiva con punti casuali")
    st.markdown("---")
    
    # Sidebar per i controlli
    with st.sidebar:
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
            help="Dimensione dell'area a fuoco (valori più alti = area più piccola)"
        )
        
        blur_strength = st.slider(
            "Sfocatura:",
            min_value=0.1,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Intensità della sfocatura nelle aree fuori fuoco"
        )
        
        randomness = st.slider(
            "Casualità:",
            min_value=0.0,
            max_value=1.0,
            value=0.5,
            step=0.05,
            help="Influenza sulla posizione e forma delle aree a fuoco"
        )
        
        ghost_threshold = st.slider(
            "Effetto Ghost:",
            min_value=0.1,
            max_value=0.9,
            value=0.5,
            step=0.05,
            help="Soglia per l'effetto bianco e nero nelle aree sfocate"
        )
        
        num_focus_points = st.slider(
            "Punti di focus:",
            min_value=1,
            max_value=5,
            value=3,
            step=1,
            help="Numero di punti di focus casuali da generare"
        )
        
        # Opzioni seed
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
    
    # Area principale
    if uploaded_files:
        # Mostra anteprima immagini caricate
        st.subheader("Immagini caricate")
        
        cols = st.columns(min(4, len(uploaded_files)))
        images = []
        filenames = []
        
        for i, file in enumerate(uploaded_files):
            with cols[i % len(cols)]:
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
            
            # Per ogni immagine caricata
            for img_index, (img, base_name) in enumerate(zip(images, filenames)):
                status.text(f"Elaborazione di {base_name}...")
                
                # Se genera varianti, creiamo una griglia
                if generate_variants:
                    st.markdown(f"**{base_name} - Varianti**")
                    
                    # Crea 2x2 griglia
                    for row in range(2):
                        cols = st.columns(2)
                        for col in range(2):
                            variant_idx = row * 2 + col
                            variant_seed = seed + variant_idx if seed else random.randint(1, 9999)
                            
                            with cols[col]:
                                # Applica l'effetto con punti di focus multipli
                                result, used_seed = apply_multiple_focus_points(
                                    img,
                                    num_focus_points,
                                    focus_ratio,
                                    blur_strength,
                                    randomness,
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
                                    ghost_threshold,
                                    used_seed
                                )
                                
                                # Mostra informazioni e link download
                                st.caption(f"Seed: {used_seed}")
                                st.markdown(get_image_download_link(result, filename), unsafe_allow_html=True)
                else:
                    # Elabora una singola variante
                    result, used_seed = apply_multiple_focus_points(
                        img,
                        num_focus_points,
                        focus_ratio,
                        blur_strength,
                        randomness,
                        seed
                    )
                    
                    # Mostra il risultato
                    st.markdown(f"**{base_name}**")
                    st.image(result, use_container_width=True)
                    
                    # Crea nome file con parametri
                    filename = create_filename(
                        base_name,
                        focus_ratio,
                        blur_strength,
                        randomness,
                        ghost_threshold,
                        used_seed
                    )
                    
                    # Mostra informazioni e link download
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
            ## Selective Focus con Punti Multipli
            
            Questa app permette di creare effetti di messa a fuoco selettiva con punti di focus multipli
            posizionati in modo casuale nell'immagine.
            
            ### Caratteristiche:
            
            - **Punti di focus multipli:** L'app genera automaticamente diversi punti di focus
            - **Effetto ghost:** Le aree fuori fuoco vengono convertite in bianco e nero
            - **Parametri regolabili:** Puoi controllare dimensione, sfocatura, casualità e intensità dell'effetto
            - **Seed riproducibile:** Usando lo stesso seed otterrai risultati identici
            
            ### Parametri:
            
            - **Area a fuoco:** Controlla la dimensione delle aree a fuoco
            - **Sfocatura:** Intensità della sfocatura nelle aree fuori fuoco
            - **Casualità:** Influisce sulla posizione e forma delle aree a fuoco
            - **Effetto Ghost:** Intensità dell'effetto bianco e nero nelle aree sfocate
            - **Punti di focus:** Numero di punti di focus casuali da generare
            """)

if __name__ == "__main__":
    main()
