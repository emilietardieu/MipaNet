"""
Module de décodeur pour le modèle MIPANet à deux branches.

IRB_block (inverted residual block): 
C’est une implémentation classique de bloc type MobileNetV2
    Conv1x1 pour étendre l’espace des canaux
    Depthwise Conv3x3 pour capturer des motifs spatiaux
    Conv1x1 pour réduire les canaux
"""
import torch.nn as nn

from .util import IRB_Block, LearnedUpUnit

class Decoder(nn.Module):
    """
    Classe principale du décodeur
        :attibut int n_classes: Nombre de classes de sortie.
        :attibut list fuse_feats: Liste des dimensions des caractéristiques extraites du backbone.
        :attibut str feats: Préfixe pour les noms des caractéristiques (x, l, d..)
    """
    def __init__(self, n_classes, fuse_feats, feats='x'):
        super().__init__()

        self.feats = feats

        # Inverser les dimensions des caractéristiques pour le décodage
        decoder_feats = fuse_feats[-2:0:-1] #[256, 128, 64]

        # Blocs de fusion
        for i in range(len(decoder_feats)):
            self.add_module('refine%d' % i, Level_Fuse_Module())

        # Blocs de Upsampling
        for i in range(len(decoder_feats)):
            self.add_module('up%d' % i, 
                IRB_Up_Block(decoder_feats[i])
            )

        # Couches auxiliaires
        for i in range(len(decoder_feats)):
            self.add_module('aux%d' % i, 
                nn.Conv2d(decoder_feats[i], n_classes, kernel_size=1, stride=1, padding=0, bias=True),
            )

        # Couches de fusion finale
        self.out_conv = nn.Sequential(
            nn.Conv2d(min(decoder_feats), n_classes, kernel_size=1, stride=1, padding=0, bias=True)
        )
        self.out_up = nn.Sequential(
            LearnedUpUnit(n_classes),
            LearnedUpUnit(n_classes)
        )

    def forward(self, in_feats):
        f1 = in_feats[f'{self.feats}1']
        f2 = in_feats[f'{self.feats}2']
        f3 = in_feats[f'{self.feats}3']
        f4 = in_feats[f'{self.feats}4']


        feats, aux0 = self.up0(f4)
        feats = self.refine0(feats, f3)

        feats, aux1 = self.up1(feats)
        feats = self.refine1(feats, f2)

        feats, aux2 = self.up2(feats)
        feats = self.refine2(feats, f1)

        aux3 = self.out_conv(feats)

        out_feats = [self.out_up(aux3), self.aux2(aux2), self.aux1(aux1), self.aux0(aux0)]

        return out_feats

class Level_Fuse_Module(nn.Module):
    """
    Module de fusion.
    Fusion par addition des skip connection avec le décodeur
    """
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        """
        Fusionne les caractéristiques x et y.
            :param torch.Tensor x: Caractéristiques du décodeur.
            :param torch.Tensor y: Caractéristiques de la skip connection.
            :returns: Caractéristiques fusionnées.
            :rtype: torch.Tensor
        """
        return x + y


class IRB_Up_Block(nn.Module):
    def __init__(self, in_feats):
        super().__init__()
        self.conv_unit = nn.Sequential(
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, 2*in_feats),
            IRB_Block(2*in_feats, in_feats)
        )
        self.up_unit = LearnedUpUnit(in_feats) #UpSampling + Conv

    def forward(self, x):
        """
        :param torch.Tensor x: Caractéristiques d'entrée.
        :returns: Caractéristiques upsamplées et caractéristiques intermédiaires.
        :rtype: tuple(torch.Tensor, torch.Tensor)
        """
        feats = self.conv_unit(x)
        return (self.up_unit(feats), feats)