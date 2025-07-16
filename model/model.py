from torch import nn

from model.net import Decoder
from model.net.encoder import Res18

class MipaNet(nn.Module):
    """
    Modèle MIPANet qui s'adapte au nombre et à la nature des data_types.
    """
    def __init__(self, n_classes, data_types=['irc', 'mnh', 'biom'], **encoder_kwargs):
        super().__init__()
        
        self.data_types = data_types
        self.num_branches = len(data_types)
        
        # Créer l'encodeur
        self.encoder = Res18(
            data_types=data_types,
            **encoder_kwargs
        )
        
        # Créer le décodeur
        self.decoder = Decoder(
            n_classes=n_classes,
            fuse_feats=self.encoder.fuse_feats,
            feats="x",
        )
    
    def forward(self, *inputs):
        """
        Forward pass du modèle MIPANet.
        """
        if len(inputs) != self.num_branches:
            raise ValueError(f"Attendu {self.num_branches} entrées pour {self.data_types}, reçu {len(inputs)}")
        
        feats = self.encoder(*inputs)
        feats = self.decoder(feats)
        return tuple(feats)

def get_mipanet(dataset, encoder_class, **encoder_kwargs):
    from .datasets import datasets
    model = MipaNet(datasets[dataset.lower()].NUM_CLASS, encoder_class, **encoder_kwargs) 
    return model

def get_mipanet(dataset, data_types=['irc', 'mnh', 'biom', 'histo'], **encoder_kwargs):
    """
    Créer un modèle MIPANet.
    
    :param dataset: nom du dataset
    :param data_types: liste des types de données à utiliser
    :param encoder_kwargs: arguments pour l'encodeur
    :return: modèle MIPANet
    """
    from .datasets import datasets
    model = MipaNet(
        datasets[dataset.lower()].NUM_CLASS, 
        data_types=data_types,
        **encoder_kwargs
    ) 
    return model