"""
Ce module gère la configuration flexible du projet.
"""
from model.net.encoder import Res18

def get_config(data_types):
    """
    Génère un dictionnaire de configuration pour l'entraînement, la phase de test et l'encodeur
    """

    # Dictionnaire de configuration modèle
    CONFIG = {

        'training': {
            'data_type': data_types,
            'wandb_activate': False,
            'commentaire': "",

            'epochs': 6,
            'batch_size': 1,
            'lr': 0.0005,

            'lr_scheduler': 'poly',
            'momentum': 0.9,

            'use_cuda': False,

            'dataset': "format",
            'data_types': ['irc', 'mnh'],
            'workers': 4,
            'train_split': 'train',
            'export': True,
            'aux_weight': 0.4,
            'class_weight': 1,
            'weight_decay': 0.0001,
            'seed': 42,
        },

        'testing': {
            'use_cuda': False,
            'dataset': 'format',
            'data_types': ['irc', 'mnh'],
            'batch_size': 16,
            'workers': 4
        },

        'encoder': {
            'encoder_class': Res18,
            'data_types': ['irc', 'mnh'],
            'pass_rff': None,
            'first_fusions': 'PAM',
            'last_fusion': 'PAM'
        },
        'TGCC': {
            'use_TGCC': True,
            'TGCC_path': '/ccc/cont003/dsku/blanchet/home/user/inp/tardieue/MY_MIPANet_2_branches'
        }
    }
    if CONFIG['TGCC']['use_TGCC']:
        CONFIG['use_cuda'] = True
    else :
        CONFIG['use_cuda'] = False


    return CONFIG