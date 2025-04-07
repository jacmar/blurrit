import streamlit as st
import os
import tempfile
import sys
from typing import Dict, Any, List, Optional, Union
import importlib.util

# Importa il modulo selective_focus_basic con un wrapper di sicurezza
try:
    import selective_focus_basic as sf
    sf_module = sf
except (ImportError, ModuleNotFoundError) as e:
    st.error(f"Errore nell'importazione del modulo: {e}")
    st.stop()

# Wrapper di sicurezza per process_images
def safe_process_images(input_dir: str, output_dir: str, mode: str, 
                       seed: Optional[int] = None, 
                       params: Optional[Dict[str, Any]] = None) -> List[str]:
    """
    Wrapper sicuro per process_images che gestisce i tipi di parametri.
    
    Args:
        input_dir: Directory con le immagini di input
        output_dir: Directory dove salvare le immagini elaborate
        mode: Modalità di elaborazione (standard, explore, sample, refine)
        seed: Seed per la generazione casuale (opzionale)
        params: Dizionario di parametri personalizzati (opzionale)
        
    Returns:
        Lista di percorsi ai file di output generati
    """
    # Assicurati che tutti i parametri necessari siano presenti
    safe_params = {} if params is None else params.copy()
    
    # Valori predefiniti per i parametri
    default_params = {
        'focus_ratio': 0.3,
        'blur_strength': 0.7,
        'randomness': 0.5,
        'ghost_threshold': 0.5
    }
    
    # Applica i valori predefiniti per i parametri mancanti
    for key, default_value in default_params.items():
        if key not in safe_params:
            safe_params[key] = default_value
    
    # Assicurati che tutti i valori siano float
    for key in safe_params:
        try:
            safe_params[key] = float(safe_params[key])
        except (TypeError, ValueError):
            st.warning(f"Parametro '{key}' non valido, uso il valore predefinito")
            safe_params[key] = default_params.get(key, 0.0)
    
    # Gestisci il seed
    safe_seed = seed
    if seed is not None:
        try:
            safe_seed = int(seed)
        except (TypeError, ValueError):
            st.warning("Seed non valido, generazione automatica")
            safe_seed = None
    
    # Ora chiama la funzione originale con parametri sicuri
    try:
        # Ottieni il nome effettivo della funzione process_images
        process_func = getattr(sf_module, 'process_images', None)
        
        # Se non esiste, cerca altre funzioni che potrebbero avere lo stesso scopo
        if process_func is None:
            potential_funcs = [
                'process_images', 'process_image', 'run', 'main', 
                'process', 'selective_focus'
            ]
            for func_name in potential_funcs:
                if hasattr(sf_module, func_name):
                    process_func = getattr(sf_module, func_name)
                    st.info(f"Utilizzo della funzione {func_name} invece di process_images")
                    break
        
        if process_func is None:
            st.error("Non riesco a trovare la funzione process_images nel modulo")
            return []
        
        # Chiamata alla funzione con gestione degli errori
        return process_func(input_dir, output_dir, mode, safe_seed, safe_params)
        
    except Exception as e:
        st.error(f"Errore durante l'elaborazione: {str(e)}")
        st.error(f"Tipo di errore: {type(e).__name__}")
        st.error(f"Dettaglio: {getattr(e, '__traceback__', 'Non disponibile')}")
        return []

# Interfaccia Streamlit
st.title("Selective Focus")

# Upload delle immagini
uploaded_files = st.file_uploader("Carica immagini", 
                                 accept_multiple_files=True, 
                                 type=['jpg', 'jpeg', 'png', 'tiff'])

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
            
            # Mostra le immagini caricate
            st.subheader("Immagini selezionate")
            image_cols = st.columns(min(3, len(uploaded_files)))
            for i, file in enumerate(uploaded_files):
                with image_cols[i % 3]:
                    st.image(file, caption=file.name, width=200)
            
            # Elabora le immagini usando il wrapper sicuro
            with st.spinner("Elaborazione in corso..."):
                output_files = safe_process_images(input_dir, output_dir, mode, seed, params)
            
            if output_files:
                # Mostra i risultati
                st.success(f"Elaborazione completata! Generati {len(output_files)} file.")
                
                # Mostra le immagini in una griglia
                st.subheader("Risultati")
                result_cols = st.columns(min(3, len(output_files)))
                
                for i, output_file in enumerate(output_files):
                    with result_cols[i % 3]:
                        # Leggi l'immagine elaborata
                        with open(output_file, "rb") as f:
                            output_data = f.read()
                        
                        # Mostra l'immagine
                        st.image(output_data, caption=f"Risultato {i+1}")
                        
                        # Pulsante di download
                        st.download_button(
                            label=f"Scarica {os.path.basename(output_file)}",
                            data=output_data,
                            file_name=os.path.basename(output_file),
                            mime="image/jpeg"
                        )
            else:
                st.error("Non sono stati generati file di output. Controlla i log per gli errori.")
else:
    st.info("Carica una o più immagini per iniziare")

# Informazioni aggiuntive
with st.expander("Informazioni"):
    st.write("""
    ## Selective Focus
    
    Questa applicazione permette di generare effetti di messa a fuoco selettiva, 
    combinando più immagini con diverse aree a fuoco.
    
    ### Modalità disponibili:
    - **Standard**: Crea una singola immagine con parametri personalizzati
    - **Explore**: Genera 4 varianti con seed diversi (parametri standard)
    - **Sample**: Genera 6 varianti artistiche con lo stesso seed
    - **Refine**: Usa un seed specifico con parametri personalizzati
    
    ### Parametri:
    - **Focus Ratio**: Controlla la dimensione dell'area a fuoco
    - **Blur Strength**: Intensità della sfocatura
    - **Randomness**: Casualità nella posizione e forma dell'area a fuoco
    - **Ghost Threshold**: Intensità dell'effetto di desaturazione nelle aree sfocate
    """)
