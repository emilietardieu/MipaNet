
from .dataset import ForMat

datasets = {
    'format': ForMat
}

def get_dataset(name, root, split, data_types=None, **kwargs):
    """
    Fonction pour créer un dataset.
    
    :param name: nom du dataset
    :param root: chemin racine
    :param split: split (train/val/test)
    :param data_types: liste des types de données (pour le dataset flexible)
    :param kwargs: autres arguments (pour compatibilité avec l'ancien système)
    :return: instance du dataset
    """
    if name.lower() == 'format':
        if data_types is None:
            data_types = ['irc', 'mnh', 'biom']  # valeur par défaut
        return datasets[name](root, split, data_types=data_types)
    else:
        # Ancien système (format3)
        return datasets[name](root, split, **kwargs)