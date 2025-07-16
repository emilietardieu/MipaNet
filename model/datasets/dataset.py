"""
==========
ForMat = Dataset flexible pour différentes combinaisons de data_types
==========
Ce module définit la classe ForMat, qui peut gérer différentes combinaisons de data_types.

La classe gère :
- le chargement dynamique des images selon les data_types spécifiés
- leur transformation pour l'entraînement, la validation ou le test
"""
from PIL import Image
from pathlib import Path

from .base import BaseDataset
from .data_type import data_type

class ForMat(BaseDataset):
    """
    Classe pour gérer un dataset flexible avec différentes combinaisons de data_types.
    """
    NUM_CLASS = 3  # classe sol (0), classe forêt (1), classe vieille forêt (2)

    def __init__(self, root, split, data_types=['irc', 'mnh', 'biom']):
        """
        Initialise le dataset flexible.
            :param str root: Chemin vers le répertoire racine du dataset.
            :param str split: Indique si le dataset est utilisé pour l'entraînement ('train') ou le test ('test').
            :param list data_types: Liste des types de données à utiliser (ex: ['irc', 'mnh'], ['irc', 'biom'], etc.)
        """
        # Validation des data_types
        available_types = set(data_type.keys())
        requested_types = set(data_types)
        if not requested_types.issubset(available_types):
            invalid_types = requested_types - available_types
            raise ValueError(f"Types de données non supportés: {invalid_types}. Types disponibles: {available_types}")
        
        self.data_types = data_types
        self.transforms = {dt: data_type[dt]['transforms'] for dt in data_types}
        
        # Appel du constructeur de la classe parente BaseDataset
        super(ForMat, self).__init__(root, split)

        # Initialisation des chemins
        _dataset_root = Path(root) / 'data'
        _mask_dir = _dataset_root / 'MASK/3classes'

        # Définition des fichiers texte train/test
        _split_f = _dataset_root / ('train.txt' if self.mode == 'train' else 'test.txt')

        # Chargement des chemins des fichiers
        self.data_paths = {dt: [] for dt in data_types}
        self.masks = []
        
        with open(_split_f, "r") as lines:
            for line in lines:
                line = line.strip()
                
                # Charger les chemins pour chaque type de données
                for dt in data_types:
                    data_dir = _dataset_root / dt.upper()
                    data_path = data_dir / f"{line}.tif"
                    assert data_path.is_file(), f"Fichier manquant: {data_path}"
                    self.data_paths[dt].append(data_path)
                
                # Charger le masque
                _mask = _mask_dir / f"{line}.tif"
                assert _mask.is_file(), f"Masque manquant: {_mask}"
                self.masks.append(_mask)

        # Vérification que tous les types de données ont le même nombre d'échantillons
        lengths = [len(self.data_paths[dt]) for dt in data_types]
        lengths.append(len(self.masks))
        assert all(l == lengths[0] for l in lengths), "Nombre d'échantillons incohérent entre les types de données"

    def __getitem__(self, index):
        """
        Récupère un échantillon du dataset à l'index spécifié.
            :param int index: Index de l'échantillon à récupérer.
            :return: Un tuple contenant les images et le masque (si mode train/val)
            :rtype: tuple
        """
        # Chargement des données
        data_tensors = []
        for dt in self.data_types:
            img = Image.open(self.data_paths[dt][index])
            img_tensor = self.transforms[dt](img)
            data_tensors.append(img_tensor)
        
        # Nom du fichier (basé sur le premier type de données)
        file_name = self.data_paths[self.data_types[0]][index].name
        
        # Mode test : pas de masque
        if self.mode == 'test':
            return tuple(data_tensors + [file_name])
        
        # Mode train/val : avec masque
        if self.mode == 'train' or self.mode == 'val':
            _target = Image.open(self.masks[index])
            _target = self._target_transform(_target)
            return tuple(data_tensors + [_target, file_name])
        
        else:
            raise ValueError(f"Mode '{self.mode}' non reconnu. Utilisez 'train', 'val' ou 'test'.")

    def __len__(self):
        """
        Retourne le nombre total d'échantillons dans le dataset.
            :return: Nombre d'échantillons dans le dataset.
            :rtype: int
        """
        return len(self.masks)