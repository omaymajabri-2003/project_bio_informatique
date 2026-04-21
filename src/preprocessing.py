# Emplacement : src/preprocessing.py

import pandas as pd
import os

def parse_adn_file(file_path):
    """
    Lit le fichier adn_data.txt et retourne un DataFrame avec 'sequence' et 'label'.
    """
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"Le fichier {file_path} est introuvable.")

    with open(file_path, 'r') as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]

    # Lecture des métadonnées (selon la structure de votre fichier)
    num_classes = int(lines[0])  # ex: 2
    class_sizes = [int(lines[i+1]) for i in range(num_classes)] # ex: [40, 47]
    
    data = []
    current_line = 1 + num_classes # On commence après les tailles de classes

    for i in range(num_classes):
        label = lines[current_line] # Lit le label (0 ou 1)
        current_line += 1
        
        # Récupère le nombre exact de séquences pour cette classe
        for _ in range(class_sizes[i]):
            sequence = lines[current_line]
            data.append({'sequence': sequence, 'label': int(label)})
            current_line += 1

    return pd.DataFrame(data)

def clean_sequences(df):
    """
    Nettoyage basique : suppression des doublons et vérification des caractères.
    """
    # Suppression des doublons
    initial_len = len(df)
    df = df.drop_duplicates(subset=['sequence'])
    if len(df) < initial_len:
        print(f"INFO: {initial_len - len(df)} doublons supprimés.")
    
    # On s'assure que les séquences sont en majuscules
    df['sequence'] = df['sequence'].str.upper()
    
    return df