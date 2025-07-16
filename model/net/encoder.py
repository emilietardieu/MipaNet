"""
Module d'encodeur flexible pour le modèle MIPANet.
L'encodeur peut traiter un nombre variable de branches selon les data_types spécifiés.
"""
import torch.nn as nn

from .fuse import FUSE_MODULE_DICT
from .util import get_resnet18
from ..datasets.data_type import data_type

class Res18(nn.Module):
    """
    Encodeur basé sur ResNet-18 pour traiter différentes combinaisons de data_types.
    """
    def __init__(self, data_types=['irc', 'mnh', 'biom'], pass_rff=None, first_fusions="PAM", last_fusion="MIM", use_tgcc=False):
        super().__init__()
        
        self.data_types = data_types
        self.num_branches = len(data_types)
        
        # Validation des types de fusion
        allowed_fusions = ["PAM", "MIM"]
        if first_fusions not in allowed_fusions or last_fusion not in allowed_fusions:
            raise ValueError(f"Les fusions autorisées sont {allowed_fusions}. Vous avez fourni {first_fusions} et {last_fusion}.")

        # Configuration pass_rff par défaut
        if pass_rff is None:
            pass_rff = tuple([False] * self.num_branches)
        
        if len(pass_rff) != self.num_branches:
            raise ValueError(f"pass_rff doit avoir {self.num_branches} éléments pour {data_types}")
        
        self.pass_rff = pass_rff
        self.first_fusions = first_fusions
        self.last_fusion = last_fusion

        # Initialisation des branches ResNet-18 pour chaque data_type
        self.branches = nn.ModuleDict()
        for dt in data_types:
            num_channels = data_type[dt]['num_channels']
            branch_base = get_resnet18(input_dim=num_channels, use_tgcc=use_tgcc)
            self.branches[f'{dt}_base'] = branch_base
            
            # Première couche adaptée aux canaux d'entrée
            self.branches[f'{dt}_layer0'] = nn.Sequential(
                branch_base.conv1,
                branch_base.bn1,
                branch_base.relu
            )
            self.branches[f'{dt}_inpool'] = branch_base.maxpool
            
            # Couches ResNet 1-4
            for i in range(1, 5):
                self.branches[f'{dt}_layer{i}'] = branch_base.__getattr__(f'layer{i}')

        # Initialisation des blocs de fusion pour chaque niveau
        self.fuse_feats = [64, 64, 128, 256, 512]
        self.fuse_modules = nn.ModuleDict()
        
        for i in range(len(self.fuse_feats)):
            if i == 4:
                # Dernière fusion
                if self.last_fusion == "PAM":
                    self.fuse_modules[f'fuse{i}'] = FUSE_MODULE_DICT[self.last_fusion](
                        in_channels=self.fuse_feats[i], 
                        num_branches=self.num_branches
                    )
                else:  # MIM
                    self.fuse_modules[f'fuse{i}'] = FUSE_MODULE_DICT[self.last_fusion](
                        in_channels=self.fuse_feats[i]
                    )
            else:
                # Fusions intermédiaires
                self.fuse_modules[f'fuse{i}'] = FUSE_MODULE_DICT[self.first_fusions](
                    in_channels=self.fuse_feats[i], 
                    num_branches=self.num_branches
                )

    def forward(self, *inputs):
        """
        Passage avant de l'encodeur.
        """
        if len(inputs) != self.num_branches:
            raise ValueError(f"Attendu {self.num_branches} entrées, reçu {len(inputs)}")
        
        # Stockage des caractéristiques par niveau et par branche
        features = {}
        
        # Niveau 0 : première couche convolutionnelle
        level0_outputs = []
        for i, dt in enumerate(self.data_types):
            x = self.branches[f'{dt}_layer0'](inputs[i])
            features[f'{dt}_l0'] = x
            level0_outputs.append(x)
        
        # Niveau 1 : après pooling + layer1
        level1_outputs = []
        for i, dt in enumerate(self.data_types):
            x = self.branches[f'{dt}_inpool'](features[f'{dt}_l0'])
            x = self.branches[f'{dt}_layer1'](x)
            features[f'{dt}_l1'] = x
            level1_outputs.append(x)
        
        # Fusion niveau 1
        if self.first_fusions == "PAM":
            fused_result = self.fuse_modules['fuse1'](*level1_outputs)
            features['x1'] = fused_result[0]
            for i, dt in enumerate(self.data_types):
                features[f'{dt}_r1'] = fused_result[i + 1]
        else:  # MIM
            fused_result = self.fuse_modules['fuse1'](level1_outputs[0], level1_outputs[1])
            features['x1'] = fused_result
            for i, dt in enumerate(self.data_types):
                features[f'{dt}_r1'] = level1_outputs[i]
        
        # Niveaux 2, 3, 4
        for level in range(2, 5):
            level_outputs = []
            for i, dt in enumerate(self.data_types):
                # Utiliser les caractéristiques raffinées si pass_rff est True
                if self.pass_rff[i]:
                    x = features[f'{dt}_r{level-1}']
                else:
                    x = features[f'{dt}_l{level-1}']
                
                x = self.branches[f'{dt}_layer{level}'](x)
                features[f'{dt}_l{level}'] = x
                level_outputs.append(x)
            
            # Fusion
            if level == 4 and self.last_fusion == "MIM":
                # Dernière fusion avec MIM
                fused_result = self.fuse_modules[f'fuse{level}'](level_outputs[0], level_outputs[1])
                features[f'x{level}'] = fused_result
                for i, dt in enumerate(self.data_types):
                    features[f'{dt}_r{level}'] = level_outputs[i]
            else:
                # Fusion avec PAM
                fused_result = self.fuse_modules[f'fuse{level}'](*level_outputs)
                features[f'x{level}'] = fused_result[0]
                for i, dt in enumerate(self.data_types):
                    features[f'{dt}_r{level}'] = fused_result[i + 1]

        # Construire le dictionnaire de sortie final
        output_features = {}
        
        # Caractéristiques fusionnées
        for level in range(1, 5):
            output_features[f'x{level}'] = features[f'x{level}']
        
        # Caractéristiques individuelles des branches (pour compatibilité)
        for dt in self.data_types:
            for level in range(1, 5):
                # Utiliser la première lettre du data_type comme clé
                key = f'{dt[0]}{level}'
                output_features[key] = features[f'{dt}_l{level}']
        
        return output_features


class Encoder(nn.Module):
    """
    Classe principale de l'encodeur.
    """
    def __init__(self, encoder_class, **encoder_kwargs):
        super().__init__()
        self.encoder = encoder_class(**encoder_kwargs)

    def forward(self, *inputs):
        """
        Passage avant de l'encodeur.
        """
        return self.encoder(*inputs)