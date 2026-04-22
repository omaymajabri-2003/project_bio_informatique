import pandas as pd
import os

def clean_sequence_file(raw_file_path):
    """
    Lit le fichier brut, extrait les métadonnées, sépare les classes
    et retourne un DataFrame propre.
    """
    with open(raw_file_path, 'r') as f:
        # Lire toutes les lignes et enlever les espaces/sauts de ligne
        lines = [line.strip() for line in f if line.strip()]

    # 1. Extraction des métadonnées (Lignes 1 à 4 du fichier)
    # nb_classes = int(lines[0]) # Non utilisé ici mais présent
    len_class_0 = int(lines[1])
    len_class_1 = int(lines[2])
    
    # 2. Extraction des séquences
    # La classe 0 commence après le marqueur '0' (qui est à l'index 3)
    start_class_0 = 4
    end_class_0 = start_class_0 + len_class_0
    sequences_0 = lines[start_class_0 : end_class_0]
    
    # La classe 1 commence après le marqueur '1'
    # Le marqueur '1' se trouve juste après la fin de la classe 0
    idx_marker_1 = end_class_0 
    start_class_1 = idx_marker_1 + 1
    end_class_1 = start_class_1 + len_class_1
    sequences_1 = lines[start_class_1 : end_class_1]

    # 3. Création du DataFrame professionnel
    data_0 = pd.DataFrame({'sequence': sequences_0, 'target': 0})
    data_1 = pd.DataFrame({'sequence': sequences_1, 'target': 1})
    
    df = pd.concat([data_0, data_1], ignore_index=True)
    
    # Nettoyage de sécurité : mise en majuscule
    df['sequence'] = df['sequence'].str.upper()
    
    return df

def save_processed_data(df, output_path):
    """Sauvegarde le DataFrame en CSV"""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_csv(output_path, index=False)