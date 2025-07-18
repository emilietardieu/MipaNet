"""
Script de lancement pour l'entraînement flexible du modèle MIPANet.
"""
import os
from pathlib import Path
from train import train
from config import get_config

def main():
    """
    Fonction principale pour lancer l'entraînement avec différentes configurations.
    """

    data_types=['irc', 'mnh']

    config = get_config(data_types)

    if not config['TGCC']['use_TGCC']:
        data_path = Path(r"/home/etardieu2/Documents/my_data/ForMat2_small")
    else:
        data_path = os.path.join(os.environ["MYSCRATCH"], "my_data", "ForMat_2000")
        data_path = Path(data_path)

    train(
        root=data_path,
        data_types=data_types,
    )

main()