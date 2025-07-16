import torch.nn as nn
import torch.nn.functional as F

class SegmentationLoss(nn.CrossEntropyLoss):
    """
    Classe SegmentationLoss qui étend nn.CrossEntropyLoss pour des tâches de segmentation sémantique
        :attribut int nclass: Nombre de classes pour la segmentation sémantique.
        :attribut bool aux: Si True, utilise la perte auxiliaire, utilisée pour des dsorties intermédiaires du décodeur
        :attribut float aux_weight: Poids pour la perte auxiliaire.
        :attribut torch.Tensor weight: Pondération des classes pour la CrossEntropyLoss.
        :attribut int ignore_index: Valeur dans les masques de segmentation qu’on ignore (ex: pixels sans annotation).
    """
    def __init__(self,
                 nclass=-1,
                 aux=True,
                 aux_weight=0.4,
                 weight=None,
                 ignore_index=-1
                 ):
        super(SegmentationLoss, self).__init__(weight, None, ignore_index)
        self.aux = aux
        self.nclass = nclass
        self.aux_weight = aux_weight

    def forward(self, *inputs):
        """
        Passage avant de la fonction de perte.
            :param inputs: Entrées pour la fonction de perte, généralement les caractéristiques de sortie et la cible.
            :returns: La perte calculée.
            :rtype: torch.Tensor
        """
        # On redimensionne le target pour qu’il corresponde à la taille de chaque sortie auxiliaire.
        if self.aux: 
            out_feats, target = inputs[0], inputs[-1]
            aux_feats, aux_loss = inputs[1:-1], []
            for aux in aux_feats: 
                _, _, h, w = aux.size()
                aux_target = F.interpolate(target.unsqueeze(1).float(), size=(h, w)).long().squeeze(1)
                aux_loss.append(super(SegmentationLoss, self).forward(aux, aux_target))
            # On calcule la perte CrossEntropyLoss sur chaque auxiliaire, puis on les moyenne
            # Donc on combine les deux pertes : la principale + une pondération des pertes auxiliaires.
            loss1 = super(SegmentationLoss, self).forward(out_feats, target)
            loss2 = sum(aux_loss) / len(aux_loss)
            return loss1 + self.aux_weight * loss2
        # Pas de perte auxiliaire, on calcule juste la CrossEntropyLoss sur la sortie principale.
        else:
            out_feats, target = inputs[0], inputs[-1]
            return super(SegmentationLoss, self).forward(out_feats, target)