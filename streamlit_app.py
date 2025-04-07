import streamlit as st
import os
import tempfile
import selective_focus_basic as sf

st.title("Selective Focus")

# Upload delle immagini
uploaded_files = st.file_uploader("Carica immagini", accept_multiple_files=True, type=['jpg', 'jpeg', 'png'])

if uploaded_files:
    st.write(f"Caricate {len(uploaded_files)} immagini")
    
    # Parametri
    col1, col2 = st.columns(2)
    
    with col1:
        mode = st.selectbox(
            "Modalità di elaborazione",
            ["standard", "explore", "sample", "refine"]
        )
    
    with col2:
        seed_option = st.radio("Seed", ["Automatico", "Personalizzato"])
        if seed_option == "Personalizzato":
            seed = st.number_input("Valore seed", min_value=1, max_value=999999, value=42)
        else:
            seed = None
    
    # Parametri personalizzati per modalità standard e refine
    if mode in ["standard", "refine"]:
        st.subheader("Parametri personalizzati")
        
        focus_ratio = st.slider("Focus Ratio", 0.1, 0.9, 0.3, 0.05)
        blur_strength = st.slider("Blur Strength", 0.1, 1.0, 0.7, 0.05)
        randomness = st.slider("Randomness", 0.0, 1.0, 0.5, 0.05)
        ghost_threshold = st.slider("Ghost Threshold", 0.0, 1.0, 0.5, 0.05)
        
        params = {
            "focus_ratio": focus_ratio,
            "blur_strength": blur_strength,
            "randomness": randomness,
            "ghost_threshold": ghost_threshold
        }
    else:
        params = {}
    
    if st.button("Elabora Immagini"):
        with tempfile.TemporaryDirectory() as input_dir, tempfile.TemporaryDirectory() as output_dir:
            # Salva le immagini caricate nella directory temporanea
            for file in uploaded_files:
                with open(os.path.join(input_dir, file.name), "wb") as f:
                    f.write(file.getbuffer())
            
            # Elabora le immagini
            with st.spinner("Elaborazione in corso..."):
                output_files = sf.process_images(input_dir, output_dir, mode, seed, params)
            
            # Mostra i risultati
            st.success("Elaborazione completata!")
            
            # Crea una griglia di immagini
            for i, output_file in enumerate(output_files):
                with open(output_file, "rb") as f:
                    st.download_button(
                        label=f"Scarica risultato {i+1}",
                        data=f,
                        file_name=os.path.basename(output_file),
                        mime="image/jpeg"
                    )
                st.image(output_file, caption=f"Risultato {i+1}")
