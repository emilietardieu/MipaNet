import numpy as np
import torch
import torch.utils.data as data

class BaseDataset(data.Dataset):
    """
    Classe de base pour les datasets.
        : param str root: Chemin racine du dataset.
        : param str split: Type de division (train, val, test).
        : param irc_transform (callable, optionnel): Transformation à appliquer aux images IRC.
        : param mnh_transform (callable, optionnel): Transformation à appliquer aux images MNH.
    """
    def __init__(self, root, split, irc_transform=None, mnh_transform=None, biom_transform=None):
        self.root = root
        self.split = split
        self.mode = split
        self.irc_transform = irc_transform
        self.mnh_transform = mnh_transform
        self.biom_transform = biom_transform

    @property
    def num_class(self):
        """
        Retourne le nombre de classes dans le dataset.
            :return: Nombre de classes.
            :rtype: int
        """
        return self.NUM_CLASS

    def _target_transform(self, mask):
        """
        Transforme le masque en tenseurs PyTorch.
            :param PIL.Image irc: Image IRC.
            :param PIL.Image mnh: Image MNH.
            :param PIL.Image mask: Masque d'entrée.
            :return: Tuple contenant l'image IRC, l'image MNH et le masque transformés
            :rtype: tuple
        """
        mask = torch.from_numpy(np.array(mask)).long()
        return mask
