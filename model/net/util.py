import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.models import resnet18


# Utilisé dans l'encodeur
def get_resnet18(input_dim, f_path='./model/utils/resnet18-5c106cde.pth', use_tgcc=False):
    """
    Chargement du modèle ResNet-18 avec des poids pré-entraînés.
    ResNet-18 est conçu pour 3 canaux d'entrée (RGB), on l'adapter pour 1, 2 ou 4 canaux.
        :param int input_dim: Dimension d'entrée (1, 2, 3 ou 4 canaux).
        :param str f_path: Chemin vers le fichier de poids pré-entraînés.
        :param bool use_tgcc: Si True, utilise le chemin TGCC pour les poids.
    """
    assert input_dim in (1, 2, 3, 4)

    # Chargement du modèle ResNet-18 avec des poids pré-entraînés sur ImageNet
    model = resnet18(weights=None)

    # si on utilise TGCC
    if use_tgcc:
        f_path = '/ccc/cont003/dsku/blanchet/home/user/inp/tardieue/MY_MIPANet_2_branches/model/utils/resnet18-5c106cde.pth'

    # Vérification de l'existence du fichier de poids
    if not os.path.exists(f_path):
        print(f"Le fichier de poids pré-entraînés n'existe pas à l'emplacement : {f_path}")
        raise FileNotFoundError('The pretrained model cannot be found.')
    
    # ResNet-18 est conçu pour 3 canaux d'entrée, donc on modifie la première couche
    if input_dim != 3:
        model.conv1 = nn.Conv2d(input_dim, 64, kernel_size=7, stride=2, padding=3, bias=False)
        weights = torch.load(f_path, weights_only=False)
        for k, v in weights.items():
            weights[k] = v.data
        conv1_ori = weights['conv1.weight'] # poids originaux pour 3 canaux
        conv1_new = torch.zeros((64, input_dim, 7, 7), dtype=torch.float32) # Taille du nouveau poids a construire

        if input_dim == 4:
            conv1_new[:, :3, :, :] = conv1_ori # On copie les 3 premiers canaux
            conv1_new[:,  3, :, :] = conv1_ori[:,  1, :, :] # Pour le quatrième canal, on copie le second canal (stratégie empirique)
        elif input_dim == 2:
            conv1_new[:,  0, :, :] = conv1_ori[:,  1, :, :] # Premier canal : on copie le canal vert
            conv1_new[:,  1, :, :] = conv1_ori[:,  2, :, :] # Deuxième canal : on copie le canal bleu
        else:  # input_dim == 1
            conv1_new[:,  0, :, :] = conv1_ori[:,  1, :, :] # On garde seulement le canal vert

        weights['conv1.weight'] = conv1_new
        model.load_state_dict(weights, strict=False)

    else:
        # On garde les poids originaux pour 3 canaux
        model.load_state_dict(torch.load(f_path,weights_only=False), strict=False)

    return model

# Utilisé dans le décodeur
class ConvBnAct(nn.Sequential):
    """
    Cette classe hérite de nn.Sequential et crée une séquence de couches :
    1. Convolution 2D
    2. Normalisation (par défaut BatchNorm2d)
    3. Activation (par défaut ReLU)
    Si `act` est False, la couche d'activation est remplacée par nn.Identity pour ne pas ajouter d'activation.
        :attribut int in_feats: Nombre de canaux d'entrée.
        :attribut int out_feats: Nombre de canaux de sortie.
        :attribut int kernel: Taille du noyau de convolution.
        :attribut int stride: Pas de la convolution.
        :attribut int pad: Padding
        :attribut bool bias: Si True, ajoute un biais à la convolution.
        :attribut dict conv_args: Arguments supplémentaires pour la couche de convolution.
        :attribut norm_layer: Couche de normalisation (par défaut nn.BatchNorm2d).
        :attribut bool act: Si True, ajoute une activation après la normalisation.
        :attribut act_layer: Couche d'activation (par défaut nn.ReLU(inplace=True)).
    """
    def __init__(
            self, in_feats, out_feats, kernel=3, stride=1, pad=1, bias=False, conv_args = {},
            norm_layer=nn.BatchNorm2d, act=True, act_layer=nn.ReLU(inplace=True)
            ):
        
        super().__init__()
        self.add_module('conv', nn.Conv2d(
            in_feats, out_feats, kernel_size=kernel, stride=stride,
            padding=pad, bias=bias, **conv_args)
            )
        self.add_module('bn', norm_layer(out_feats))
        self.add_module('act', act_layer if act else nn.Identity())

class ResidualBasicBlock(nn.Module):
    """
    Bloc de base résiduel
    """
    def __init__(self, in_feats, out_feats=None):
        super().__init__()
        self.conv_unit = nn.Sequential(
            ConvBnAct(in_feats, in_feats),
            ConvBnAct(in_feats, in_feats, act=False)
        )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        """
        Passage avant du bloc résiduel.
        Passage de l'entrée dans deux couches ConvBnAct, puis addition de l'entrée à la sortie.
            :param torch.Tensor x: Entrée du bloc résiduel.
            :returns: Sortie du bloc résiduel après l'addition.
            :rtype: torch.Tensor
        """
        out = self.conv_unit(x)
        return self.relu(x + out)

class IRB_Block(nn.Module):
    """
    Inverted Residual Block (IRB) utilisé dans les architectures MobileNet.
    Il est composé de trois couches :
    1. Convolution point-wise (1x1) pour augmenter la dimensionnalité
    2. Convolution depth-wise (3x3) pour capturer les caractéristiques spatiales
    3. Convolution point-wise (1x1) pour réduire la dimensionnal
        :param int in_feats: Nombre de canaux d'entrée.
        :param int out_feats: Nombre de canaux de sortie. Si None, il est égal à in_feats.
        :param str act: Type d'activation à utiliser. 'idt' pour nn.Identity, 'relu' pour nn.ReLU6.
        :param int expand_ratio: Ratio d'expansion pour le nombre de canaux intermédiaires.
    """
    def __init__(self, in_feats, out_feats=None, act='idt', expand_ratio=6):
        super().__init__()
        mid_feats = round(in_feats * expand_ratio)
        out_feats = in_feats if out_feats is None else out_feats
        act_layer = nn.Identity() if act == 'idt' else nn.ReLU6(inplace=True)
        self.idt = (in_feats == out_feats)
        self.irb = nn.Sequential(
                # point-wise conv
                nn.Conv2d(in_feats, mid_feats, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(mid_feats),
                nn.ReLU6(inplace=True),
                # mnh-wise conv
                nn.Conv2d(mid_feats, mid_feats, kernel_size=3, stride=1, padding=1, groups=mid_feats, bias=False),
                nn.BatchNorm2d(mid_feats),
                nn.ReLU6(inplace=True),
                # point-wise conv
                nn.Conv2d(mid_feats, out_feats, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_feats),
                act_layer
            )

    def forward(self, x):
        """
        Passage avant de l'IRB.
        Si `self.idt`, on ajoute l'entrée `x` à la sortie de l'IRB.
        Sinon, on retourne seulement la sortie de l'IRB.
            :param torch.Tensor x: Entrée de l'IRB.
            :returns: Sortie de l'IRB, éventuellement additionnée à l'entrée.
            :rtype: torch.Tensor
        """
        return (x + self.irb(x)) if self.idt else self.irb(x)

class LearnedUpUnit(nn.Module):
    """
    Unité de Upsampling avec une couche de convolution.
        :param int in_feats: Nombre de canaux d'entrée.
    """
    def __init__(self, in_feats):
        super().__init__()
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.depth_conv = nn.Conv2d(in_feats, in_feats, kernel_size=3, stride=1, padding=1, groups=in_feats, bias=False)

    def forward(self, x):
        """
        Passage avant de l'unité de Upsampling.
            :param torch.Tensor x: Entrée de l'unité de Upsampling.
            :returns: Sortie de l'unité de Upsampling après la convolution depth-wise.
            :rtype: torch.Tensor
        """
        x = self.up(x)
        x = self.depth_conv(x)
        return x

