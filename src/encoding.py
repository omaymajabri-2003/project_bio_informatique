# Emplacement : src/encoding.py
import pandas as pd
from collections import Counter

def get_textual_decomposition(df, k):
    """Représentation visuelle : 'MPA PAT ATS...'"""
    decomposed = []
    for seq in df['sequence']:
        kmers = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        decomposed.append(" ".join(kmers))
    return pd.DataFrame({'division_k': decomposed, 'target' : df['target']})

def get_numeric_matrices(df, k):
    """
    Génère les matrices numériques en gardant l'ORDRE D'APPARITION.
    Exemple k=2 : MP, PA, AT, TS, SS, SI...
    """
    all_sequences_counts = []
    vocabulaire_ordonne = []
    deja_vus = set()

    # 1. Identifier tous les motifs dans l'ordre de lecture
    for seq in df['sequence']:
        kmers_in_seq = [seq[i:i+k] for i in range(len(seq) - k + 1)]
        
        # Compter les occurrences pour cette ligne
        all_sequences_counts.append(Counter(kmers_in_seq))
        
        # Remplir le vocabulaire dans l'ordre d'apparition sans doublons
        for kmer in kmers_in_seq:
            if kmer not in deja_vus:
                vocabulaire_ordonne.append(kmer)
                deja_vus.add(kmer)

    # 2. Construction de la matrice d'OCCURRENCE avec cet ordre
    print(f"Construction de la matrice ({len(vocabulaire_ordonne)} colonnes)...")
    data_occ = []
    for counts in all_sequences_counts:
        # On suit l'ordre de 'vocabulaire_ordonne' pour les colonnes
        ligne = [counts.get(kmer, 0) for kmer in vocabulaire_ordonne]
        data_occ.append(ligne)
    
    X_occurrence = pd.DataFrame(data_occ, columns=vocabulaire_ordonne)

    # 3. Construction de la matrice BOOLÉENNE
    X_boolean = (X_occurrence > 0).astype(int)

    return X_occurrence, X_boolean