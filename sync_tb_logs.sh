#!/bin/bash

# À adapter avec ton identifiant et chemin exact sur le cluster
CLUSTER_USER=tardieue
CLUSTER_HOST=gen15621@v100
CLUSTER_LOG_DIR=/ccc/scratch/cont003/gen15621/tardieue/my_data/ForMat_11_small/MIPANet_2_branches/train/tensorboard_logs


# Dossier local où tu veux enregistrer
LOCAL_DIR=/home/etardieu2/Documents/my_data/ForMat2_small/MIPANet_2_branches/train

# Crée le dossier si besoin
mkdir -p $LOCAL_DIR

# Copie les logs
scp -r etardieu2@constantina.ensiacet.fr tardieue@irene-fr.ccc.cea.fr:${CLUSTER_LOG_DIR} $LOCAL_DIR
