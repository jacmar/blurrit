import streamlit as st
import os
import random
import numpy as np
from PIL import Image, ImageFilter, ImageDraw
import io
import base64
from datetime import datetime

# Set page configuration
st.set_page_config(
    page_title="Selective Focus - Generatore di Effetti",
    page_icon="🔍",
    layout="wide"
)

def apply_selective_focus(img, focus_ratio=0.3, blur_strength=0.7, randomness=0.5, seed=None):
    """Apply simple selective focus effect without ghost effect"""
    # Set random seed for reproducibility
    if seed is None:
        seed = random.randint(1, 999999)
    random.seed(seed)
    np.random.seed(seed)
    
    # Make sure image is in RGB mode
    if img.mode != 'RGB':
        img = img.convert('RGB')
    
    # Get dimensions
    width, height = img.size
    
    # Create a blurred copy
    blur_radius = max(1, int(20 * blur_strength))
    blurred = img.filter(ImageFilter.GaussianBlur(radius=blur_radius))
    
    # Create mask for focus area
    mask = Image.new('L', (width, height), 0)
    draw = ImageDraw.Draw(mask)
    
    # Calculate center and focus size
    center_x, center_y = width // 2, height // 2
    focus_size = min(width, height) * (1.0 - focus_ratio * 0.8)
    
    # Add randomness to focus position
    if randomness > 0:
        center_x += int(random.uniform(-width * 0.1, width * 0.1) * randomness)
        center_y += int(random.uniform(-height * 0.1, height * 0.1) * randomness)
    
    # Draw focus area
    draw.ellipse(
        (center_x - focus_size/2, center_y - focus_size/2,
         center_x + focus_size/2, center_y + focus_size/2),
        fill=255
    )
    
    # Blur the mask for a smooth transition
    mask = mask.filter(ImageFilter.GaussianBlur(radius=max(1, focus_size/15)))
    
    # Composite the original and blurred images
    result = Image.composite(img, blurred, mask)
    
    return result, seed

def get_image_download_link(img, filename):
    """Generate a download link for a processed image"""
    buffered = io.BytesIO()
    img.save(buffered, format="JPEG", quality=95)
    img_str = base64.b64encode(buffered.getvalue()).decode()
    href = f'<a href="data:file/jpg;base64,{img_str}" download="{filename}" class="download-button">📥 Scarica</a>'
    return href

def main():
    st.title("🔍 SELECTIVE FOCUS")
    st.markdown("Crea effetti di messa a fuoco selettiva sulle tue immagini")
    st.markdown("---")
    
    # Sidebar for controls
    with st.sidebar:
        st.header("Controlli")
        
        # Image uploader
        uploaded_files = st.file_uploader(
            "Carica immagini",
            type=["jpg", "jpeg", "png"],
            accept_multiple_files=True
        )
        
        # Mode selection
        mode = st.radio(
            "Modalità:",
            ["Standard", "Esplora varianti", "Campiona stili"]
        )
        
        # Parameters
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
            value=0.3,
            step=0.05,
            help="Aggiunge casualità alla posizione dell'area a fuoco"
        )
        
        # Seed options
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
        
        # Process button
        process_button = st.button(
            "Elabora immagini",
            type="primary",
            use_container_width=True
        )
    
    # Main content area
    if uploaded_files:
        # Show uploaded images
        st.subheader("Immagini caricate")
        
        # Display thumbnails in a grid
        cols = st.columns(min(4, len(uploaded_files)))
        for i, file in enumerate(uploaded_files):
            with cols[i % len(cols)]:
                img = Image.open(file)
                st.image(img, width=150, caption=file.name)
        
        # Process images when button is clicked
        if process_button:
            st.markdown("---")
            st.subheader("Risultati")
            
            # Progress tracking
            progress_bar = st.progress(0)
            status = st.empty()
            
            # Number of variants based on mode
            variants = 1
            if mode == "Esplora varianti":
                variants = 4
            elif mode == "Campiona stili":
                variants = 6
            
            # Process each image
            for i, file in enumerate(uploaded_files):
                status.text(f"Elaborazione di {file.name}...")
                
                # Open image
                img = Image.open(file)
                
                if mode == "Standard":
                    # Single image with current parameters
                    result, used_seed = apply_selective_focus(
                        img, focus_ratio, blur_strength, randomness, seed
                    )
                    
                    # Show result
                    st.markdown(f"**{file.name}** (Seed: {used_seed})")
                    st.image(result, use_container_width=True)
                    
                    # Create download link
                    file_name = f"{os.path.splitext(file.name)[0]}_focus.jpg"
                    st.markdown(get_image_download_link(result, file_name), unsafe_allow_html=True)
                
                elif mode == "Esplora varianti":
                    # Show 4 variants with different seeds
                    st.markdown(f"**{file.name}** - Varianti")
                    
                    # Create a grid of 2x2 images
                    rows = 2
                    for row in range(rows):
                        cols = st.columns(2)
                        for col in range(2):
                            variant_idx = row * 2 + col
                            variant_seed = seed if seed else random.randint(1, 999999) + variant_idx
                            
                            with cols[col]:
                                result, _ = apply_selective_focus(
                                    img, focus_ratio, blur_strength, randomness, variant_seed
                                )
                                st.image(result, use_container_width=True)
                                st.caption(f"Seed: {variant_seed}")
                                
                                # Create download link
                                file_name = f"{os.path.splitext(file.name)[0]}_variant_{variant_idx+1}.jpg"
                                st.markdown(get_image_download_link(result, file_name), unsafe_allow_html=True)
                
                elif mode == "Campiona stili":
                    # Create 6 preset styles
                    presets = [
                        {"name": "Soft Focus", "focus": 0.2, "blur": 0.5, "random": 0.1},
                        {"name": "Dramatic", "focus": 0.5, "blur": 0.9, "random": 0.2},
                        {"name": "Wide Focus", "focus": 0.1, "blur": 0.4, "random": 0.3},
                        {"name": "Tight Focus", "focus": 0.7, "blur": 0.8, "random": 0.2},
                        {"name": "Dreamy", "focus": 0.3, "blur": 0.6, "random": 0.5},
                        {"name": "Vivid", "focus": 0.4, "blur": 0.7, "random": 0.0}
                    ]
                    
                    st.markdown(f"**{file.name}** - Stili")
                    
                    # Using same seed for all styles
                    used_seed = seed if seed else random.randint(1, 999999)
                    
                    # Display in 2 rows of 3
                    for row in range(2):
                        cols = st.columns(3)
                        for col in range(3):
                            preset_idx = row * 3 + col
                            if preset_idx < len(presets):
                                preset = presets[preset_idx]
                                
                                with cols[col]:
                                    result, _ = apply_selective_focus(
                                        img, 
                                        preset["focus"],
                                        preset["blur"],
                                        preset["random"],
                                        used_seed
                                    )
                                    st.image(result, use_container_width=True)
                                    st.caption(f"{preset['name']}")
                                    
                                    # Create download link
                                    file_name = f"{os.path.splitext(file.name)[0]}_{preset['name'].lower().replace(' ', '_')}.jpg"
                                    st.markdown(get_image_download_link(result, file_name), unsafe_allow_html=True)
                
                # Update progress
                progress_bar.progress((i + 1) / len(uploaded_files))
                
                # Add separator between images
                if i < len(uploaded_files) - 1:
                    st.markdown("---")
            
            # Clear status and progress
            status.empty()
            progress_bar.empty()
            
            # Final message
            st.success(f"✅ Elaborazione completata! {len(uploaded_files)} immagini elaborate.")
    
    else:
        # Welcome message when no images are uploaded
        st.info("👆 Carica una o più immagini dalla barra laterale per iniziare.")
        
        with st.expander("Informazioni su Selective Focus"):
            st.markdown("""
            ## Cos'è Selective Focus?
            
            Selective Focus è uno strumento che ti permette di creare effetti di messa a fuoco selettiva 
            sulle tue immagini, simulando l'effetto di profondità di campo ottenuto con lenti fotografiche.
            
            ### Modalità disponibili:
            
            - **Standard**: Crea una singola immagine con i parametri specificati
            - **Esplora varianti**: Genera 4 varianti con diversi seed casuali
            - **Campiona stili**: Applica 6 stili predefiniti all'immagine
            
            ### Parametri:
            
            - **Area a fuoco**: Controlla la dimensione dell'area a fuoco
            - **Intensità sfocatura**: Quanto saranno sfocate le aree fuori fuoco
            - **Casualità**: Aggiunge variazione alla posizione dell'area a fuoco
            
            ### Seed:
            
            Il "seed" è un valore numerico che garantisce risultati consistenti.
            Usando lo stesso seed, otterrai lo stesso effetto ogni volta.
            """)

if __name__ == "__main__":
    main()
