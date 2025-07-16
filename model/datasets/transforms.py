"""
Toutes les transformations nécessaires pour les différentes données du projet.
"""
import torch
import torchvision.transforms as transform

irc_mean = [.392, .234, .250]
irc_std = [.145, .082, .094]

mnh_mean = [0.0508]
mnh_std = [0.0442]

biom_mean = [0.5502]
biom_std = [0.3004]

# Modifié pour ne prendre que le premier canal (le deuxième étant constant)
histo_mean = [0.472421]
histo_std = [0.175449]

# Transformation des données : conversion en tenseurs et normalisation
irc_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(irc_mean, irc_std)])

mnh_transform = transform.Compose([
        transform.ToTensor(),
        transform.Lambda(lambda x: x.to(torch.float)),
        transform.Normalize(mnh_mean, mnh_std)
    ])

biom_transform = transform.Compose([
    transform.ToTensor(),
    transform.Normalize(biom_mean, biom_std)
    ])

# Transformation pour prendre seulement le premier canal des images histogramme
histo_transform = transform.Compose([
    transform.ToTensor(),
    transform.Lambda(lambda x: x[0:1, :, :]),  # Prendre seulement le premier canal
    transform.Normalize(histo_mean, histo_std)
    ])



transforms = {'irc_transform': irc_transform, 'mnh_transform' : mnh_transform,'biom_transform' : biom_transform, 'histo_transform' : histo_transform}