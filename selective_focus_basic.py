#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import os
import cv2
import numpy as np
import random
import time
import sys
import argparse

def process_images(input_folder, output_folder, focus_ratio=0.3, blur_strength=0.7, randomness=0.5, ghost_threshold=0.5, seed=None, variant_name=None):
    """
    Elabora le immagini nella cartella di input e crea un'immagine con effetto di messa a fuoco selettiva.
    """
    # Imposta il seed se fornito
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed)
    
    # Verifica che la cartella di input esista
    if not os.path.exists(input_folder):
        print(f"ERRORE: La directory di input '{input_folder}' non esiste.")
        return None
    
    # Crea la cartella di output se non esiste
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
        print(f"Creata directory di output: {output_folder}")
    
    # Trova le immagini nella cartella di input
    image_files = []
    for file in os.listdir(input_folder):
        ext = os.path.splitext(file)[1].lower()
        if ext in ['.jpg', '.jpeg', '.png', '.tiff', '.JPG', '.JPEG', '.PNG', '.TIFF']:
            image_files.append(os.path.join(input_folder, file))
    
    if not image_files:
        print(f"Nessuna immagine trovata in {input_folder}")
        return None
    
    print(f"Trovate {len(image_files)} immagini")
    
    # Carica le immagini
    images = []
    for path in image_files:
        img = cv2.imread(path)
        if img is not None:
            images.append(img)
            print(f"Caricata immagine: {os.path.basename(path)}")
        else:
            print(f"Impossibile caricare {path}")
    
    if not images:
        print("Nessuna immagine caricata con successo")
        return None
    
    # Assicurati che tutte le immagini abbiano le stesse dimensioni
    if len(images) > 1:
        target_h, target_w = images[0].shape[:2]
        for i in range(1, len(images)):
            h, w = images[i].shape[:2]
            if h != target_h or w != target_w:
                images[i] = cv2.resize(images[i], (target_w, target_h))
                print(f"Ridimensionata immagine {i+1} a {target_w}x{target_h}")
    
    # Dimensioni dell'immagine
    h, w = images[0].shape[:2]
    
    # Crea una mappa di messa a fuoco
    focus_mask = np.zeros((h, w), dtype=np.float32)
    
    # Aggiungi regioni casuali a fuoco
    num_regions = 2 + int(randomness * 5)
    for _ in range(num_regions):
        cx = random.randint(0, w-1)
        cy = random.randint(0, h-1)
        radius = int(min(w, h) * (0.1 + random.random() * 0.2))
        
        y, x = np.mgrid[:h, :w]
        dist = np.sqrt((x - cx)**2 + (y - cy)**2)
        region = np.exp(-(dist**2) / (2 * radius**2))
        focus_mask = np.maximum(focus_mask, region)
    
    # Aggiungi una regione centrale a fuoco
    cx, cy = w // 2, h // 2
    radius = int(min(w, h) * focus_ratio * 0.5)
    y, x = np.mgrid[:h, :w]
    dist = np.sqrt((x - cx)**2 + (y - cy)**2)
    center_region = np.exp(-(dist**2) / (2 * radius**2))
    focus_mask = np.maximum(focus_mask, center_region)
    
    # Normalizza la maschera
    if np.max(focus_mask) > 0:
        focus_mask = focus_mask / np.max(focus_mask)
    
    # Sfuma i bordi della maschera
    kernel_size = int(21 * blur_strength)
    if kernel_size % 2 == 0:
        kernel_size += 1
    kernel_size = max(3, min(kernel_size, 45))  # Limita tra 3 e 45
    focus_mask = cv2.GaussianBlur(focus_mask, (kernel_size, kernel_size), 0)
    
    # Crea l'immagine risultato
    result = np.zeros_like(images[0], dtype=np.float32)
    
    # Aggiungi l'immagine originale nelle aree a fuoco
    result += focus_mask[:, :, np.newaxis] * images[0].astype(np.float32)
    
    # Pesi totali applicati
    weights_sum = focus_mask.copy()
    
    # Applica le altre immagini nelle aree sfocate
    if len(images) > 1:
        for i in range(1, len(images)):
            # Calcola differenze con l'immagine base
            gray1 = cv2.cvtColor(images[0], cv2.COLOR_BGR2GRAY)
            gray2 = cv2.cvtColor(images[i], cv2.COLOR_BGR2GRAY)
            
            diff = np.abs(gray1.astype(np.float32) - gray2.astype(np.float32))
            max_diff = np.max(diff) if np.max(diff) > 0 else 1
            diff_norm = diff / max_diff
            
            # Controlla l'effetto ghost con la soglia
            ghost_mask = np.clip(1.0 - diff_norm + ghost_threshold, 0, 1)
            
            # Crea una maschera per questa immagine
            img_mask = np.zeros((h, w), dtype=np.float32)
            
            # Aggiungi regioni casuali
            num_areas = 2 + int(randomness * 3)
            for _ in range(num_areas):
                cx = random.randint(0, w-1)
                cy = random.randint(0, h-1)
                rx = int(min(w, h) * (0.1 + random.random() * 0.3))
                ry = int(min(w, h) * (0.1 + random.random() * 0.3))
                
                # Crea una forma ellittica
                y, x = np.mgrid[:h, :w]
                angle = random.random() * np.pi
                x_rot = (x - cx) * np.cos(angle) + (y - cy) * np.sin(angle)
                y_rot = -(x - cx) * np.sin(angle) + (y - cy) * np.cos(angle)
                dist = (x_rot / rx)**2 + (y_rot / ry)**2
                area = np.exp(-dist)
                
                img_mask = np.maximum(img_mask, area)
            
            # Normalizza la maschera
            if np.max(img_mask) > 0:
                img_mask = img_mask / np.max(img_mask)
            
            # Sfuma i bordi
            img_mask = cv2.GaussianBlur(img_mask, (kernel_size, kernel_size), 0)
            
            # Applica solo nelle aree disponibili
            # (non a fuoco e controllando ghost)
            available = (1.0 - weights_sum) * ghost_mask
            final_mask = img_mask * available
            
            # Sfoca l'immagine con intensità variabile
            blurred = cv2.GaussianBlur(
                images[i].astype(np.float32), 
                (max(3, int(31 * blur_strength) | 1), max(3, int(31 * blur_strength) | 1)), 
                0
            )
            
            # Aggiungi al risultato
            result += final_mask[:, :, np.newaxis] * blurred
            
            # Aggiorna i pesi totali
            weights_sum += final_mask
    
    # Riempi le aree rimanenti con l'immagine originale sfocata
    remaining = 1.0 - weights_sum
    if np.max(remaining) > 0:
        original_blur = cv2.GaussianBlur(
            images[0].astype(np.float32), 
            (max(3, int(31 * blur_strength) | 1), max(3, int(31 * blur_strength) | 1)), 
            0
        )
        result += remaining[:, :, np.newaxis] * original_blur
    
    # Aggiungi effetto di movimento
    if randomness > 0:
        flow_x = np.zeros((h, w), dtype=np.float32)
        flow_y = np.zeros((h, w), dtype=np.float32)
        
        # Aggiungi punti di distorsione
        num_points = 3 + int(randomness * 7)
        for _ in range(num_points):
            cx = random.randint(0, w-1)
            cy = random.randint(0, h-1)
            intensity = random.uniform(5, 15) * blur_strength
            radius = random.randint(w//10, w//3)
            angle = random.uniform(0, 2 * np.pi)
            dx = np.cos(angle)
            dy = np.sin(angle)
            
            y, x = np.mgrid[:h, :w]
            dist = np.sqrt((x - cx)**2 + (y - cy)**2)
            factor = intensity * np.exp(-(dist**2) / (2 * radius**2))
            
            flow_x += factor * dx
            flow_y += factor * dy
        
        # Crea la mappa di distorsione
        map_x = np.float32(np.array([[i for i in range(w)] for _ in range(h)]))
        map_y = np.float32(np.array([[i] * w for i in range(h)]))
        
        map_x = map_x + flow_x
        map_y = map_y + flow_y
        
        # Assicurati che i valori siano validi
        map_x = np.clip(map_x, 0, w-1)
        map_y = np.clip(map_y, 0, h-1)
        
        # Applica la distorsione
        result = cv2.remap(result, map_x, map_y, cv2.INTER_LINEAR)
    
    # Convertiti in uint8
    result = np.clip(result, 0, 255).astype(np.uint8)
    
    # Aggiungi il seed al nome del file se utilizzato
    seed_suffix = f"_seed{seed}" if seed is not None else ""
    variant_suffix = f"_{variant_name}" if variant_name else ""
    
    # Salva il risultato
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    filename = f"selective_focus_fr{focus_ratio:.2f}_bs{blur_strength:.2f}_r{randomness:.2f}_g{ghost_threshold:.2f}{seed_suffix}{variant_suffix}_{timestamp}.jpg"
    output_path = os.path.join(output_folder, filename)
    cv2.imwrite(output_path, result)
    
    print(f"Immagine salvata in: {output_path}")
    return output_path

def explore_mode(input_folder, output_folder, num_seeds=4):
    """
    Modalità esplorazione: genera diverse versioni con seed diversi e parametri standard
    """
    # Parametri standard
    focus_ratio = 0.3
    blur_strength = 0.7
    randomness = 0.5
    ghost_threshold = 0.5
    
    # Genera seed casuali
    seeds = [random.randint(1000, 9999) for _ in range(num_seeds)]
    
    print(f"Generazione di {num_seeds} varianti con seed diversi...")
    for seed in seeds:
        print(f"Elaborazione con seed {seed}...")
        process_images(
            input_folder, 
            output_folder, 
            focus_ratio, 
            blur_strength, 
            randomness, 
            ghost_threshold, 
            seed
        )

def sample_mode(input_folder, output_folder, seed):
    """
    Modalità sample: genera 6 varianti significative con lo stesso seed
    """
    if seed is None:
        seed = random.randint(1000, 9999)
        print(f"Seed casuale generato: {seed}")
    
    # Definizione delle 6 varianti
    variants = [
        # nome, focus_ratio, blur_strength, randomness, ghost_threshold
        ("base", 0.5, 0.5, 0.5, 0.5),  # Valori medi
        ("dettaglio", 0.7, 0.3, 0.5, 0.5),  # Alto focus, bassa sfocatura
        ("sogno", 0.3, 0.7, 0.5, 0.5),  # Basso focus, alta sfocatura
        ("movimento", 0.5, 0.5, 0.7, 0.5),  # Alta randomness
        ("fantasma", 0.5, 0.5, 0.5, 0.7),  # Alto ghost threshold
        ("contrasto", 0.7, 0.5, 0.5, 0.7)   # Alto focus, alto ghost
    ]
    
    print(f"Generazione di 6 varianti significative con seed {seed}...")
    for name, fr, bs, r, g in variants:
        print(f"Elaborazione variante '{name}'...")
        process_images(
            input_folder,
            output_folder,
            fr, bs, r, g,
            seed,
            name
        )

def main():
    parser = argparse.ArgumentParser(description='Crea un\'immagine con messa a fuoco selettiva')
    parser.add_argument('--input_dir', default='input', help='Directory contenente le immagini di input (default: "input")')
    parser.add_argument('--output_dir', default='output', help='Directory dove salvare il risultato (default: "output")')
    
    # Aggiungi argomento per la modalità
    parser.add_argument('--mode', '-m', choices=['standard', 'explore', 'refine', 'sample'], default='standard',
                      help='Modalità: standard (singola immagine con parametri specificati), explore (genera 4 varianti con seed diversi), refine (usa un seed specifico), sample (genera 6 varianti con lo stesso seed)')
    
    parser.add_argument('--focus_ratio', type=float, default=0.3, help='Rapporto delle aree a fuoco (0.1-0.9)')
    parser.add_argument('--blur_strength', type=float, default=0.7, help='Intensità della sfocatura (0.1-1.0)')
    parser.add_argument('--randomness', type=float, default=0.5, help='Livello di casualità (0.0-1.0)')
    parser.add_argument('--ghost_threshold', type=float, default=0.5, help='Soglia per l\'effetto ghosting (0.0-1.0). Più alto = più ghosting')
    
    # Aggiungi argomento per il seed
    parser.add_argument('--seed', '-s', type=int, default=None, help='Seed per la generazione casuale (per modalità refine e sample)')
    
    args = parser.parse_args()
    
    # Esegui la modalità appropriata
    if args.mode == 'explore':
        print("Modalità esplorazione: generazione di 4 varianti con seed diversi")
        explore_mode(args.input_dir, args.output_dir)
    elif args.mode == 'sample':
        print("Modalità sample: generazione di 6 varianti significative con lo stesso seed")
        sample_mode(args.input_dir, args.output_dir, args.seed)
    else:
        print(f"Elaborazione immagini da: {args.input_dir}")
        print(f"Parametri: focus_ratio={args.focus_ratio}, blur_strength={args.blur_strength}, randomness={args.randomness}, ghost_threshold={args.ghost_threshold}")
        
        if args.mode == 'refine' and args.seed is None:
            print("ERRORE: La modalità refine richiede un seed specifico. Usa --seed per specificarlo.")
            return
        
        process_images(
            args.input_dir,
            args.output_dir,
            args.focus_ratio,
            args.blur_strength,
            args.randomness,
            args.ghost_threshold,
            args.seed if args.mode == 'refine' else None
        )

if __name__ == "__main__":
    main()
