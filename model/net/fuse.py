"""
Module de fusion pour MIPANet
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

# PAM
class ChannelAttention(nn.Module):
    """
    Module d'attention par canal.
        :attributs int in_channels : Nombre de canaux des caractéristiques d'entrée.
    """
    def __init__(self, in_channels):
        super(ChannelAttention, self).__init__()
        self.in_channels = in_channels

        self.to_avg_pool = nn.AdaptiveAvgPool2d(2)  
        self.to_max_pool = nn.MaxPool2d(kernel_size=2)  
        self.to_conv = nn.Conv2d(self.in_channels, self.in_channels, kernel_size=1)
        self.to_sigmoid = nn.Sigmoid() 

    def forward(self, x):
        """
        Passage avant du module ChannelAttention.
            :param torch.Tensor x : Caractéristiques d'entrée.
            :returns: Caractéristiques re-pondérées.
            :rtype: torch.Tensor
        """
        avg_pooled_x = self.to_avg_pool(x)
        max_pooled_x = self.to_max_pool(avg_pooled_x)
        weighted_x = self.to_conv(max_pooled_x)
        attention_x = self.to_sigmoid(weighted_x)
        out_x = x * attention_x + x

        return out_x

#PAM
class SpatialAttention(nn.Module):
    """
    Module d'attention spatiale.
        :attributs int in_channels : Nombre de canaux des caractéristiques d'entrée.
    """
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        self.relu = nn.ReLU()

    def forward(self, x):
        """
        Passage avant du module SpatialAttention.
            :param torch.Tensor x : Caractéristiques d'entrée.
            :returns: Caractéristiques re-pondérées.
            :rtype: torch.Tensor
        """
        # Spatial attention
        spatial_attention = self.sigmoid(self.conv1(x))
        out_x = x * self.relu(spatial_attention)

        return out_x

#PAM
class to_Attention(nn.Module):
    """
    Module combinant l'attention par canal et l'attention spatiale.
    Adapté pour supporter un nombre variable de branches.
        :attributs int in_channels : Nombre de canaux des caractéristiques d'entrée.
        :attributs int num_branches : Nombre de branches à fusionner.
    """
    def __init__(self, in_channels, num_branches=3):
        super().__init__()
        self.num_branches = num_branches
        self.Self1 = ChannelAttention(in_channels)
        self.Self2 = SpatialAttention(in_channels)

    def forward(self, *inputs):
        """
        Passage avant du module to_Attention.
            :param torch.Tensor inputs : Caractéristiques d'entrée de chaque branche.
            :returns: Caractéristiques fusionnées et individuelles.
            :rtype: tuple
        """
        if len(inputs) != self.num_branches:
            raise ValueError(f"Attendu {self.num_branches} entrées, reçu {len(inputs)}")
        
        # Appliquer l'attention par canal à chaque entrée
        attended_outputs = []
        for x in inputs:
            attended_outputs.append(self.Self1(x))
        
        # Fusionner toutes les sorties
        result = sum(attended_outputs)
        
        return tuple([result] + attended_outputs)
    

#MIM
class CrossAttention(nn.Module):
    """
    Module d'attention croisée 
    Ce module applique une attention croisée multi-tête entre une entrée de requête (x_q) 
    et une entrée clé/valeur (x_kv), avec un résiduel sur x_kv.
        :attribut int in_channels : Nombre de canaux des caractéristiques d'entrée.
        :attribut int heads : Nombre de têtes d'attention.
        :attribut int dim_head : Dimension de chaque tête d'attention.
        :attribut float dropout : Taux de dropout sur la sortie.
    """
    def __init__(self, in_channels, heads=8, dim_head=64, dropout=0.1):
        super().__init__()

        self.heads = heads
        self.dim_head = dim_head
        self.inner_dim = heads * dim_head  # dimension totale après projection multi-tête
        self.scale = dim_head ** -0.5  # facteur de normalisation pour l'attention

        # Projections linéaires pour Q, K et V (requête, clé, valeur)
        self.to_q = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.to_k = nn.Linear(in_channels, self.inner_dim, bias=False)
        self.to_v = nn.Linear(in_channels, self.inner_dim, bias=False)

        # Projection finale (optionnelle) pour ramener à la dimension d'origine
        self.to_out = nn.Sequential(
            nn.Linear(self.inner_dim, in_channels),
            nn.Dropout(dropout)
        ) if self.inner_dim != in_channels else nn.Identity()

    def forward(self, x_q, x_kv):
        """
        Passage avant du module CrossAttention
            :param torch.Tensor x_q : Entrée requête (query), de forme [B, C, H, W].
            :param torch.Tensor x_kv : Entrée clé/valeur (key/value),
            :returns: Résultat de l'attention croisée, de forme [B, C, H, W].
            :rtype: torch.Tensor
        """
        B, C, H, W = x_q.shape  # B: batch, C: canaux, HxW: spatial

        # Mise à plat spatiale : (B, C, H, W) → (B, N, C), avec N = H * W
        x_q = x_q.view(B, C, -1).transpose(1, 2)   # [B, N, C]
        x_kv = x_kv.view(B, C, -1).transpose(1, 2) # [B, N, C]
        N = x_q.size(1)  # Nombre total de positions spatiales

        # Projections Q, K, V
        q = self.to_q(x_q)  # [B, N, heads * dim_head]
        k = self.to_k(x_kv)
        v = self.to_v(x_kv)

        # Découpe en têtes : [B, N, H*D] → [B, H, N, D]
        q = q.view(B, N, self.heads, self.dim_head).transpose(1, 2)  # [B, H, N, D]
        k = k.view(B, N, self.heads, self.dim_head).transpose(1, 2)
        v = v.view(B, N, self.heads, self.dim_head).transpose(1, 2)

        # Attention scores : produit scalaire entre Q et K transposé
        attn_scores = torch.matmul(q, k.transpose(-2, -1)) * self.scale  # [B, H, N, N]

        # Normalisation par softmax
        attn_probs = F.softmax(attn_scores, dim=-1)  # [B, H, N, N]

        # Pondération des valeurs : A x V
        out = torch.matmul(attn_probs, v)  # [B, H, N, D]

        # Fusion des têtes : [B, H, N, D] → [B, N, H*D]
        out = out.transpose(1, 2).contiguous().view(B, N, self.inner_dim)

        # Projection finale (si nécessaire) : [B, N, H*D] → [B, N, C]
        out = self.to_out(out)

        # Reshape inverse : [B, N, C] → [B, C, H, W]
        out = out.transpose(1, 2).view(B, C, H, W)

        # Ajout résiduel de x_kv
        return out + x_kv


FUSE_MODULE_DICT = {
    'PAM': to_Attention,
    'MIM': CrossAttention
}