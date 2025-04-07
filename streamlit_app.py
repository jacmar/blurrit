import streamlit as st
import os
import glob
import random
import numpy as np
from PIL import Image, ImageFilter, ImageEnhance, ImageDraw, ImageChops
import io
import base64
import traceback
from datetime import datetime

# Carica CSS personalizzato
local_css_file = os.path.join(os.path.dirname(__file__), "style.css")
if os.path.exists(local_css_file):
    with open(local_css_file) as f:
        st.markdown(f'<style>{f.read()}</style>', unsafe_allow_html=True)
# Set page configuration
st.set_page_config(
    page_title="Selective Focus - Generatore di Effetti",
    page_icon="🔍",
    layout="wide"
)

def apply_effect(img, params, seed):
    """Apply the selective focus effect to a single image"""
    random.seed(seed)
    np.random.seed(seed)
    
    # Assicurati che l'immagine sia in RGB o RGBA
    if img.mode not in ('RGB', 'RGBA'):
        img = img.convert('RGB')
    
    # Extract parameters
    focus_ratio = params["focus_ratio"]  # 0.1-0.9
    blur_strength = params["blur_strength"]  # 0.1-1.0
    randomness = params["randomness"]  # 0.0-1.0
    ghost_threshold = params["ghost_threshold"]  # 0.0-1.0
    
    # Create a blurred copy of the image
    blur_radius = int(20 * blur_strength)
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Create a mask for selective focus
    width, height = img.size
    mask = Image.new('L', (width, height), 0)
    
    # Center of the image
    center_x, center_y = width // 2, height // 2
    
    # Radius of the focused area, modified with focus_ratio
    focus_size = min(width, height) * (1.0 - focus_ratio * 0.8)
    
    # Add randomness to the focus center if requested
    if randomness > 0:
        center_x += int(random.uniform(-width * 0.2, width * 0.2) * randomness)
        center_y += int(random.uniform(-height * 0.2, height * 0.2) * randomness)
    
    # Create focus mask with gradual transition
    draw = ImageDraw.Draw(mask)
    draw.ellipse((center_x - focus_size/2, center_y - focus_size/2,
                  center_x + focus_size/2, center_y + focus_size/2), fill=255)
    
    # Blur the mask edges for smooth transition
    transition_blur = max(1, int(focus_size/12))  # Increased smoothness
    mask = mask.filter(ImageFilter.GaussianBlur(radius=transition_blur))
    
    # Apply randomness to the mask if requested
    if randomness > 0:
        mask_array = np.array(mask)
        noise = np.random.normal(0, 20 * randomness, mask_array.shape)  # Reduced noise intensity
        mask_array = np.clip(mask_array + noise, 0, 255).astype(np.uint8)
        mask = Image.fromarray(mask_array)
    
    # Combine the original image and the blurred one using the mask
    result = Image.composite(img, blurred, mask)
    
    # Apply "ghost" effect if threshold > 0
    if ghost_threshold > 0:
        # Usa il valore effettivo di ghost_threshold per evitare l'effetto "grigio"
        result = apply_ghost_effect(result, mask, ghost_threshold)
    
    return result

def apply_ghost_effect(img, mask, threshold):
    """Apply a creative 'ghost' effect to out-of-focus areas"""
    # Salva il formato originale dell'immagine
    original_mode = img.mode
    # Converti in RGBA per il processamento
    img = img.convert('RGBA')
    width, height = img.size
    result = img.copy()
    pixels = result.load()
    mask_pixels = mask.load()
    
    for y in range(height):
        for x in range(width):
            r, g, b, a = pixels[x, y]
            mask_value = mask_pixels[x, y]
            
            # If we're outside the focus area based on threshold
            if mask_value < 255 * threshold:
                # Partial desaturation effect
                gray = (r + g + b) // 3
                factor = 0.7  # Effect intensity
                
                # Mix between original color and gray
                r = int(r * (1 - factor) + gray * factor)
                g = int(g * (1 - factor) + gray * factor)
                b = int(b * (1 - factor) + gray * factor)
                
                # Now we use more natural effect - only slight blue tint
                b = min(255, int(b * 1.05))  # Reduced from 1.1 to 1.05
                
                pixels[x, y] = (r, g, b, a)
    
    # Se l'immagine originale era in RGB, riconverti per evitare problemi
    if original_mode == 'RGB':
        result = result.convert('RGB')
    
    return result

def process_image(img, mode, seed=None, params=None):
    """
    Process a single image with selective focus effect.
    
    Parameters:
    img (PIL.Image): Input image
    mode (str): Processing mode ('standard', 'explore', 'sample', 'refine')
    seed (int, optional): Random seed for reproducibility
    params (dict, optional): Custom parameters for the effect
    
    Returns:
    list: List of processed images (PIL.Image objects)
    """
    results = []
    
    # Set random seed
    if seed is None:
        seed = random.randint(1, 999999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Default parameters
    default_params = {
        "focus_ratio": 0.3,      # 0.1-0.9
        "blur_strength": 0.7,    # 0.1-1.0
        "randomness": 0.5,       # 0.0-1.0
        "ghost_threshold": 0.5   # 0.0-1.0
    }
    
    # Merge with custom parameters if provided
    if params:
        for key, value in params.items():
            if key in default_params:
                default_params[key] = value
    
    params = default_params
    
    # Process according to mode
    if mode == "standard" or mode == "refine":
        # Generate a single image with the specified parameters
        result = apply_effect(img, params, seed)
        results.append(result)
    
    elif mode == "explore":
        # Generate 4 variations with different seeds
        for i in range(4):
            current_seed = seed + i
            result = apply_effect(img, params, current_seed)
            results.append(result)
    
    elif mode == "sample":
        # Generate 6 artistic variations with the same seed
        presets = [
            {"focus_ratio": 0.2, "blur_strength": 0.5, "randomness": 0.3, "ghost_threshold": 0.4},
            {"focus_ratio": 0.3, "blur_strength": 0.8, "randomness": 0.6, "ghost_threshold": 0.5},
            {"focus_ratio": 0.4, "blur_strength": 0.9, "randomness": 0.4, "ghost_threshold": 0.6},
            {"focus_ratio": 0.5, "blur_strength": 0.7, "randomness": 0.7, "ghost_threshold": 0.3},
            {"focus_ratio": 0.6, "blur_strength": 0.6, "randomness": 0.5, "ghost_threshold": 0.4},
            {"focus_ratio": 0.7, "blur_strength": 0.5, "randomness": 0.8, "ghost_threshold": 0.2}
        ]
        
        for preset in presets:
            # Merge with default parameters
            params_copy = default_params.copy()
            for key, value in preset.items():
                params_copy[key] = value
                
            result = apply_effect(img, params_copy, seed)
            results.append(result)
    
    return results

def get_image_download_link(img, filename, text):
    """Generate a download link for an image"""
    buffered = io.BytesIO()
    # Converti l'immagine in RGB se è in modalità RGBA per evitare errori JPEG
    if img.mode == 'RGBA':
        img = img.convert('RGB')
    img.save(buffered, format="JPEG", quality=90)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" style="text-decoration:none;">{text}</a>'
    return href

# Main Streamlit app
def main():
    st.title("🔍 SELECTIVE FOCUS - GENERATORE DI EFFETTI")
    st.markdown("---")
    
    # Sidebar for navigation/settings
    with st.sidebar:
        st.header("Impostazioni")
        
        # Input handling
        st.subheader("Input")
        uploaded_files = st.file_uploader("Carica immagini", type=["jpg", "jpeg", "png"], accept_multiple_files=True)
        
        # Mode selection
        st.subheader("Modalità di elaborazione")
        mode = st.selectbox("Scegli modalità", 
                          ["Standard - Parametri personalizzati", 
                           "Esplora - 4 varianti con seed diversi", 
                           "Sample - 6 varianti artistiche", 
                           "Refine - Seed specifico con parametri personalizzati"],
                          index=0)
        
        # Convert to simple mode string
        mode_map = {
            "Standard - Parametri personalizzati": "standard",
            "Esplora - 4 varianti con seed diversi": "explore",
            "Sample - 6 varianti artistiche": "sample",
            "Refine - Seed specifico con parametri personalizzati": "refine"
        }
        mode_key = mode_map[mode]
        
        # Seed handling
        seed = None
        if mode_key in ["refine", "sample"]:
            seed_option = st.radio("Seed", ["Genera automaticamente", "Inserisci seed specifico"])
            if seed_option == "Inserisci seed specifico":
                seed = st.number_input("Seed:", min_value=1, max_value=999999, value=42)
        
        # Parameters
        params = {}
        if mode_key in ["standard", "refine"]:
            st.subheader("Parametri personalizzati")
            
            focus_ratio = st.slider("Focus ratio:", 0.1, 0.9, 0.3, 0.05, 
                                   help="Controlla la dimensione dell'area a fuoco (valori più alti = area più piccola)")
            
            blur_strength = st.slider("Blur strength:", 0.1, 1.0, 0.7, 0.05,
                                     help="Intensità dell'effetto sfocato nelle aree fuori fuoco")
            
            randomness = st.slider("Randomness:", 0.0, 1.0, 0.5, 0.05,
                                  help="Aggiunge casualità alla posizione e forma dell'area a fuoco")
            
            ghost_threshold = st.slider("Ghost threshold:", 0.0, 1.0, 0.5, 0.05,
                                       help="Soglia per l'effetto 'fantasma' nelle aree sfocate")
            
            params = {
                "focus_ratio": focus_ratio,
                "blur_strength": blur_strength,
                "randomness": randomness,
                "ghost_threshold": ghost_threshold
            }
        
        process_button = st.button("Elabora Immagini", type="primary")
    
    # Main content area
    if uploaded_files:
        # Display a preview of uploaded images
        st.subheader("Immagini caricate")
        image_cols = st.columns(min(4, len(uploaded_files)))
        
        for i, uploaded_file in enumerate(uploaded_files):
            with image_cols[i % len(image_cols)]:
                img = Image.open(uploaded_file)
                st.image(img, caption=uploaded_file.name, width=150)
        
        # Process images when button is clicked
        if process_button:
            st.markdown("---")
            st.subheader("Risultati")
            
            # Set up progress tracking
            progress_bar = st.progress(0)
            status_text = st.empty()
            
            # Process each image
            all_results = []
            for i, uploaded_file in enumerate(uploaded_files):
                img = Image.open(uploaded_file)
                status_text.text(f"Elaborazione di {uploaded_file.name}...")
                
                try:
                    # Process the image - note changed function name here
                    results = process_image(img, mode_key, seed, params)
                    all_results.append((uploaded_file.name, results))
                except Exception as e:
                    st.error(f"Errore durante l'elaborazione: {str(e)}")
                    st.error(f"Tipo di errore: {type(e).__name__}")
                    st.error(f"Traceback: {traceback.format_exc()}")
                    break
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
            
            # Clear progress indicators
            progress_bar.empty()
            status_text.empty()
            
            # Display results
            if all_results:
                st.subheader("Immagini elaborate")
                
                # Calculate how many columns to display based on the mode
                if mode_key == "standard" or mode_key == "refine":
                    num_cols = 1
                elif mode_key == "explore":
                    num_cols = 2
                else:  # sample
                    num_cols = 3
                
                # For each processed image
                for orig_name, results in all_results:
                    st.markdown(f"**Risultati per: {orig_name}**")
                    
                    # Create rows of results
                    for i in range(0, len(results), num_cols):
                        cols = st.columns(num_cols)
                        
                        for j in range(num_cols):
                            if i + j < len(results):
                                with cols[j]:
                                    result_img = results[i + j]
                                    st.image(result_img, use_container_width=True)
                                    
                                                    # Create a unique filename
                                    base_name = os.path.splitext(orig_name)[0]
                                    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                                    effect_type = "standard" if mode_key == "standard" else \
                                                 f"explore_{i+j+1}" if mode_key == "explore" else \
                                                 f"sample_{i+j+1}" if mode_key == "sample" else \
                                                 f"refine_{seed}"
                                    filename = f"{base_name}_{effect_type}.jpg"
                                    
                                    # Provide download link
                                    st.markdown(get_image_download_link(result_img, filename, "📥 Scarica"), unsafe_allow_html=True)
                    
                    st.markdown("---")
            else:
                st.warning("Nessun risultato generato a causa di errori.")
    else:
        # Show instructions when no images are uploaded
        st.info("👆 Carica una o più immagini dalla barra laterale per iniziare.")
        
        with st.expander("Informazioni su Selective Focus"):
            st.markdown("""
            # Benvenuto in Selective Focus
            
            Questo strumento ti permette di creare effetti artistici di messa a fuoco selettiva sulle tue immagini.
            
            ## Modalità disponibili:
            
            - **Standard**: Crea una singola immagine con parametri personalizzati
            - **Esplora**: Genera 4 varianti con seed diversi (parametri standard)
            - **Sample**: Genera 6 varianti artistiche con lo stesso seed
            - **Refine**: Usa un seed specifico con parametri personalizzati
            
            ## Parametri:
            
            - **Focus ratio**: Controlla la dimensione dell'area a fuoco
            - **Blur strength**: Intensità dell'effetto sfocato nelle aree fuori fuoco
            - **Randomness**: Aggiunge casualità alla posizione e forma dell'area a fuoco
            - **Ghost threshold**: Soglia per l'effetto 'fantasma' nelle aree sfocate
            """)

if __name__ == "__main__":
    main()
