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

# Definizione degli stili di focus
FOCUS_STYLES = [
    {"name": "organic", "label": "Organico", "description": "Forme irregolari con transizioni morbide"},
    {"name": "dreamy", "label": "Sognante", "description": "Effetto onirico con sfocature ampie"},
    {"name": "motion", "label": "Movimento", "description": "Simula movimento con sfocatura direzionale"},
    {"name": "multi_point", "label": "Multi-punto", "description": "Numerosi piccoli punti a fuoco"},
    {"name": "tilt_shift", "label": "Tilt-shift", "description": "Effetto miniatura con banda a fuoco"}
]

def apply_organic_focus(depth_map, focus_points, width, height, radius, randomness):
    """Crea forme organiche con bordi molto irregolari e sfumati"""
    focus_mask = np.zeros_like(depth_map, dtype=np.float32)
    
    for x, y in focus_points:
        # Crea una forma molto più irregolare con molti vertici
        num_vertices = random.randint(12, 20)
        vertices = []
        
        for i in range(num_vertices):
            angle = 2 * np.pi * i / num_vertices
            # Raggio estremamente variabile
            r = radius * random.uniform(0.4, 2.0)
            vx = x + int(r * np.cos(angle))
            vy = y + int(r * np.sin(angle))
            
            # Aggiungi "rumore" ai vertici per renderli ancora più irregolari
            vx += int(radius * 0.4 * random.uniform(-1, 1))
            vy += int(radius * 0.4 * random.uniform(-1, 1))
            
            # Limita ai bordi dell'immagine
            vx = max(0, min(width-1, vx))
            vy = max(0, min(height-1, vy))
            vertices.append([vx, vy])
        
        # Disegna il poligono
        temp_mask = np.zeros_like(depth_map, dtype=np.float32)
        cv2.fillPoly(temp_mask, [np.array(vertices, dtype=np.int32)], 1.0)
        
        # Applica più passaggi di blur con kernel di dimensioni diverse
        # per ottenere transizioni ancora più morbide e naturali
        temp_mask = cv2.GaussianBlur(temp_mask, (41, 41), 0)
        temp_mask = cv2.GaussianBlur(temp_mask, (81, 81), 0)
        
        # Aggiungi alla maschera principale
        focus_mask = np.maximum(focus_mask, temp_mask)
    
    return focus_mask

def apply_dreamy_focus(depth_map, focus_points, width, height, radius, randomness):
    """Crea un effetto onirico con grandi aree sfumate e sovrapposte"""
    focus_mask = np.zeros_like(depth_map, dtype=np.float32)
    
    # Seleziona solo alcuni punti di focus
    num_points = max(1, min(len(focus_points), 3))
    selected_points = random.sample(focus_points, num_points) if len(focus_points) > num_points else focus_points
    
    for x, y in selected_points:
        # Raggio molto grande
        large_radius = int(radius * 4.0 * random.uniform(0.9, 1.3))
        
        # Crea una maschera temporanea con gradiente radiale
        temp_mask = np.zeros((height, width), dtype=np.float32)
        
        # Crea un gradiente basato sulla distanza
        for y_coord in range(height):
            for x_coord in range(width):
                # Calcola distanza
                dx = x_coord - x
                dy = y_coord - y
                
                # Aggiungi distorsione alla distanza
                if randomness > 0.3:
                    distortion = randomness * 50.0
                    dx += random.uniform(-distortion, distortion)
                    dy += random.uniform(-distortion, distortion)
                
                distance = np.sqrt(dx**2 + dy**2)
                
                # Formula del gradiente con decadimento esponenziale più lento
                gradient_value = np.exp(-distance**2 / (2 * (large_radius*1.2)**2))
                temp_mask[y_coord, x_coord] = max(temp_mask[y_coord, x_coord], gradient_value)
        
        # Sfuma ulteriormente i bordi
        temp_mask = cv2.GaussianBlur(temp_mask, (101, 101), 0)
        
        # Aggiungi alla maschera principale
        focus_mask = np.maximum(focus_mask, temp_mask)
    
    return focus_mask

def apply_motion_focus(depth_map, focus_points, width, height, radius, randomness):
    """Crea un effetto tipo motion blur"""
    focus_mask = np.zeros_like(depth_map, dtype=np.float32)
    
    if not focus_points:
        return focus_mask
    
    # Punto di partenza (uno dei punti focali)
    x, y = random.choice(focus_points)
    
    # Scegli una direzione casuale
    angle = random.uniform(0, 2 * np.pi)
    
    # Lunghezza e larghezza del motion blur
    length = int(min(width, height) * 0.4 * (0.7 + randomness))
    width_blur = int(radius * random.uniform(0.5, 1.5))
    
    # Crea punti lungo la traiettoria
    points = []
    for t in range(-length//2, length//2, max(3, length//20)):
        px = int(x + t * np.cos(angle))
        py = int(y + t * np.sin(angle))
        
        # Aggiungi variazione perpendicolare alla direzione
        if randomness > 0.3:
            perp_angle = angle + np.pi/2
            deviation = int(randomness * width_blur * random.uniform(-1, 1))
            px += int(deviation * np.cos(perp_angle))
            py += int(deviation * np.sin(perp_angle))
        
        # Assicurati che i punti siano all'interno dell'immagine
        px = max(0, min(width-1, px))
        py = max(0, min(height-1, py))
        points.append((px, py))
    
    # Crea la maschera di motion blur
    temp_mask = np.zeros_like(depth_map, dtype=np.float32)
    
    # Disegna una serie di cerchi sfumati lungo il percorso
    for i, (px, py) in enumerate(points):
        # Dimensione variabile lungo il percorso
        local_width = width_blur * (1 - 0.5 * abs(i - len(points)//2) / (len(points)//2 or 1))
        
        # Disegna un cerchio
        cv2.circle(temp_mask, (px, py), int(local_width), 1.0, -1)
    
    # Sfuma nella direzione del movimento
    ksize = max(3, min(151, int(length / 3)))
    # Assicura che sia dispari
    ksize = ksize + 1 if ksize % 2 == 0 else ksize
    
    # Applica blur direzionale
    temp_mask = cv2.GaussianBlur(temp_mask, (ksize, ksize), 0)
    
    # Aggiungi un po' di gaussian blur per ammorbidire i bordi
    temp_mask = cv2.GaussianBlur(temp_mask, (41, 41), 0)
    
    # Aggiungi alla maschera principale
    focus_mask = np.maximum(focus_mask, temp_mask)
    
    return focus_mask

def apply_multi_point_focus(depth_map, focus_points, width, height, radius, randomness):
    """Crea numerosi piccoli punti di focus sparsi"""
    focus_mask = np.zeros_like(depth_map, dtype=np.float32)
    
    # Genera molti punti aggiuntivi
    all_points = []
    
    # Prima aggiungi i punti di focus originali
    for x, y in focus_points:
        all_points.append((x, y))
    
    # Poi genera punti extra
    num_extra_points = int(20 + randomness * 30)
    
    # Genera sia punti raggruppati intorno ai punti focali
    # che alcuni completamente casuali
    for _ in range(num_extra_points):
        if random.random() < 0.7 and focus_points:  # 70% raggruppati
            # Scegli un punto focale casuale
            fx, fy = random.choice(focus_points)
            
            # Distanza dal punto focale
            dist = radius * random.uniform(0.5, 3.0)
            angle = random.uniform(0, 2 * np.pi)
            
            px = int(fx + dist * np.cos(angle))
            py = int(fy + dist * np.sin(angle))
        else:  # 30% completamente casuali
            px = random.randint(0, width-1)
            py = random.randint(0, height-1)
        
        # Limita ai bordi dell'immagine
        px = max(0, min(width-1, px))
        py = max(0, min(height-1, py))
        
        all_points.append((px, py))
    
    # Crea piccole aree di focus per ogni punto
    for px, py in all_points:
        # Raggio molto variabile
        local_radius = int(radius * 0.2 * random.uniform(0.3, 1.8))
        
        temp_mask = np.zeros_like(depth_map, dtype=np.float32)
        
        # Usa forme variabili (cerchi o ellissi)
        if random.random() < 0.7:  # Cerchio
            cv2.circle(temp_mask, (px, py), local_radius, 1.0, -1)
        else:  # Ellisse
            a = local_radius * random.uniform(0.8, 1.2)
            b = local_radius * random.uniform(0.8, 1.2)
            angle = random.uniform(0, 360)
            
            cv2.ellipse(
                temp_mask, 
                (px, py), 
                (int(a), int(b)), 
                angle, 0, 360, 1.0, -1
            )
        
        # Sfuma i bordi, con intensità variabile
        blur_size = int(local_radius * 4) | 1  # Assicura che sia dispari
        blur_size = max(3, min(blur_size, 41))
        temp_mask = cv2.GaussianBlur(temp_mask, (blur_size, blur_size), 0)
        
        # Intensità variabile
        if random.random() < 0.4:
            intensity = random.uniform(0.3, 1.0)
            temp_mask *= intensity
        
        # Aggiungi alla maschera principale
        focus_mask = np.maximum(focus_mask, temp_mask)
    
    return focus_mask
def apply_tilt_shift_focus(depth_map, focus_points, width, height, radius, randomness):
    """Crea un effetto tilt-shift con una banda a fuoco"""
    focus_mask = np.zeros_like(depth_map, dtype=np.float32)
    
    # Scegli un angolo per la banda
    angle = random.uniform(-30, 30) if randomness > 0.3 else 0
    
    # Centro dell'immagine come punto di riferimento
    center_x, center_y = width // 2, height // 2
    
    # Se ci sono punti focali, sposta il centro verso di essi
    if focus_points:
        avg_x = sum(x for x, _ in focus_points) / len(focus_points)
        avg_y = sum(y for _, y in focus_points) / len(focus_points)
        
        # Interpolazione tra centro immagine e media punti focali
        center_x = int(center_x * 0.3 + avg_x * 0.7)
        center_y = int(center_y * 0.3 + avg_y * 0.7)
    
    # Larghezza della banda a fuoco
    band_width = min(width, height) * (0.1 + randomness * 0.2)
    
    # Crea la maschera con una banda sfumata
    for y in range(height):
        for x in range(width):
            # Coordinate relative al centro
            rel_x = x - center_x
            rel_y = y - center_y
            
            # Rotazione delle coordinate
            angle_rad = angle * np.pi / 180
            rot_x = rel_x * np.cos(angle_rad) - rel_y * np.sin(angle_rad)
            rot_y = rel_x * np.sin(angle_rad) + rel_y * np.cos(angle_rad)
            
            # Distanza dalla linea centrale (usando solo y dopo la rotazione)
            distance = abs(rot_y)
            
            # Valore di maschera basato sulla distanza
            mask_value = np.exp(-distance**2 / (2 * (band_width/2)**2))
            
            # Aggiungi un po' di rumore se richiesto
            if randomness > 0.5:
                noise = randomness * 0.2 * random.uniform(-1, 1)
                mask_value = max(0, min(1, mask_value + noise))
            
            focus_mask[y, x] = mask_value
    
    # Aggiungi ulteriore sfumatura
    focus_mask = cv2.GaussianBlur(focus_mask, (81, 81), 0)
    
    return focus_mask

# Dizionario di funzioni di stile
FOCUS_STYLE_FUNCTIONS = {
    "organic": apply_organic_focus,
    "dreamy": apply_dreamy_focus,
    "motion": apply_motion_focus,
    "multi_point": apply_multi_point_focus,
    "tilt_shift": apply_tilt_shift_focus
}

def simulate_depth_map(image, blur_range=(3, 25), randomness=0.5, ghosting=0.3, seed=None, focus_style="organic"):
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
    
    # 4. Crea una maschera di messa a fuoco basata sui punti selezionati e lo stile scelto
    
    # Calcola il raggio di base per le aree di focus
    radius = min(width, height) / (6 + randomness * 4)
    
    # Applica lo stile di focus scelto
    if focus_style in FOCUS_STYLE_FUNCTIONS:
        focus_mask = FOCUS_STYLE_FUNCTIONS[focus_style](depth_map, focus_points, width, height, radius, randomness)
    else:
        # Fallback allo stile organico
        focus_mask = apply_organic_focus(depth_map, focus_points, width, height, radius, randomness)
    
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
    ghost_images = []
    if ghosting > 0:
        # Crea copie spostate dell'immagine originale
        
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
    if ghosting > 0 and ghost_images:
        for ghost_img, weight in ghost_images:
            # Mescola il ghost con il risultato
            result = result * (1.0 - weight) + ghost_img * weight
    
    # Converti il risultato in un'immagine PIL
    result_img = Image.fromarray(result.astype(np.uint8))
    
    # Opzionalmente, restituisci anche la depth map per visualizzazione/debug
    depth_map_img = Image.fromarray(depth_map)
    focus_mask_img = Image.fromarray((focus_mask * 255).astype(np.uint8))
    
    return result_img, depth_map_img, focus_mask_img, seed, focus_style

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

def create_filename(base_name, focus_style, blur_min, blur_max, random, ghosting, seed):
    """Crea nome file con i parametri inclusi e lo stile"""
    b_min = f"{blur_min:.1f}".replace('.', '_')
    b_max = f"{blur_max:.1f}".replace('.', '_')
    r_str = f"{random:.2f}".replace('.', '_')
    g_str = f"{ghosting:.2f}".replace('.', '_')
    seed_str = f"{seed % 10000:04d}"
    return f"{base_name}_{focus_style}_b{b_min}-{b_max}_r{r_str}_g{g_str}_s{seed_str}.jpg"

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
    
    # Nome progetto
    project_name = st.text_input(
        "Nome del progetto:",
        value="selective_focus",
        help="Questo nome verrà usato come base per tutti i file generati"
    )
    
    # Upload immagini
    uploaded_files = st.file_uploader(
        "Carica immagini",
        type=["jpg", "jpeg", "png"],
        accept_multiple_files=True,
        help="Carica una o più immagini. Se carichi più immagini, verranno combinate."
    )
    
    # Parametri
    st.subheader("Parametri")
    
    # Seleziona stile di focus
    focus_style = st.selectbox(
        "Stile di focus:",
        options=[style["name"] for style in FOCUS_STYLES],
        format_func=lambda x: next((style["label"] for style in FOCUS_STYLES if style["name"] == x), x),
        help="Seleziona lo stile dell'effetto di messa a fuoco"
    )
    
    # Mostra descrizione dello stile selezionato
    selected_style = next((style for style in FOCUS_STYLES if style["name"] == focus_style), None)
    if selected_style:
        st.caption(selected_style["description"])
    
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
        
        if uploaded_files and len(uploaded_files) > 1:
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
                    result, depth_map, focus_mask, used_seed, used_style = simulate_depth_map(
                        combined_img,
                        (min_blur, max_blur),
                        randomness,
                        ghosting,
                        seed,
                        focus_style
                    )
                    
                    # Mostra risultato
                    st.image(result, use_container_width=True)
                    
                    # Mostra i parametri utilizzati
                    st.info(f"""
                    **Parametri utilizzati:**
                    - Stile: {next((style["label"] for style in FOCUS_STYLES if style["name"] == used_style), used_style)}
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
                    filename = create_filename(
                        project_name,
                        used_style,
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
                    
                    # Crea 4 varianti UNICHE con seed e stili diversi
                    variant_images = []
                    variant_filenames = []
                    
                    # Crea 2x2 grid for visualization
                    variant_cols1 = st.columns(2)
                    variant_cols2 = st.columns(2)
                    
                    # Usa diversi stili per le varianti
                    style_names = [style["name"] for style in FOCUS_STYLES]
                    
                    for i, col in enumerate([*variant_cols1, *variant_cols2]):
                        with col:
                            # Ogni variante ha un seed diverso, parametri diversi, E uno stile diverso
                            variant_seed = (seed + i * 159) if seed else random.randint(1, 9999)
                            
                            # Scegli uno stile diverso per ogni variante
                            variant_style = style_names[i % len(style_names)]
                            
                            # Varia MOLTO di più i parametri
                            variant_randomness = min(1.0, max(0.1, randomness * random.uniform(0.6, 1.8)))
                            variant_ghosting = min(1.0, max(0.0, ghosting * random.uniform(0.5, 2.0)))
                            variant_min_blur = max(1, min_blur + random.randint(-4, 6))
                            variant_max_blur = max(variant_min_blur + 10, max_blur + random.randint(-8, 12))
                            
                            # Applica l'effetto con stile diverso
                            result, _, _, used_seed, used_style = simulate_depth_map(
                                combined_img,
                                (variant_min_blur, variant_max_blur),
                                variant_randomness,
                                variant_ghosting,
                                variant_seed,
                                variant_style
                            )
                            
                            # Salva per download multiplo
                            variant_images.append(result)
                            variant_filename = create_filename(
                                f"{project_name}_variant_{i+1}",
                                used_style,
                                variant_min_blur,
                                variant_max_blur,
                                variant_randomness,
                                variant_ghosting,
                                used_seed
                            )
                            variant_filenames.append(variant_filename)
                            
                            # Mostra risultato
                            st.image(result, use_container_width=True)
                            
                            # Ottieni l'etichetta dello stile per la visualizzazione
                            style_label = next((style["label"] for style in FOCUS_STYLES if style["name"] == used_style), used_style)
                            
                            # Mostra i parametri utilizzati
                            st.info(f"""
                            **Variante {i+1}:**
                            - Stile: {style_label}
                            - Sfocatura: min={variant_min_blur}, max={variant_max_blur}
                            - Casualità: {variant_randomness:.2f}
                            - Ghosting: {variant_ghosting:.2f}
                            - Seed: {used_seed}
                            """)
                            
                            # Mostra il link per il download
                            st.markdown(get_image_download_link(result, variant_filename), unsafe_allow_html=True)
                    
                    # Aggiungi un link per scaricare tutte le varianti in un file ZIP
                    st.markdown("---")
                    st.markdown(get_zip_download_link(variant_images, variant_filenames, f"{project_name}_varianti.zip"), unsafe_allow_html=True)
                    
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
            
            ### Stili di focus disponibili:
            
            - **Organico**: Crea forme irregolari con transizioni molto morbide
            - **Sognante**: Effetto onirico con ampie aree sfumate
            - **Movimento**: Simula il movimento con sfocatura direzionale
            - **Multi-punto**: Numerosi piccoli punti a fuoco sparsi
            - **Tilt-shift**: Effetto miniatura con una banda a fuoco

            ### Suggerimenti:
            
            - Carica immagini con elementi interessanti distribuiti in diverse aree
            - Prova diversi stili di focus per ottenere effetti completamente diversi
            - L'effetto ghosting simula la sovrapposizione di scatti leggermente diversi
            - Se carichi più immagini simili, l'algoritmo creerà un effetto di stacking
            """)
