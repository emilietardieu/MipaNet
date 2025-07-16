from .transforms import transforms

data_type = {
    "irc" : {
        "num_channels": 3,
        "transforms": transforms['irc_transform']
    },
    "mnh": {
        "num_channels": 1,
        "transforms": transforms['mnh_transform']
    },
    "biom": {
        "num_channels": 1,
        "transforms": transforms['biom_transform']
    },
    "histo": {
        "num_channels": 1,
        "transforms": transforms['histo_transform']     
    }
}