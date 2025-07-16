
def get_param_ids(module_list):
    """
    Récupère tous les paramètres des modules, prend leur identifiant mémoire, retourne une liste plate de tous ces identifiants
    Sert à Différencier les groupes de paramètres pour l’optimiseur (paramètres de base/paramètres ajoutés)
        :param list module_list: Liste de modules (ici resnet))
        :return: Liste des identifiants des paramètres des modules.
        :rtype: list
    """
    param_ids = []
    for mo in module_list:
        ids = list(map(id, mo.parameters()))  # Récupère les identifiants des paramètres du module
        param_ids = param_ids + ids  # Ajoute les identifiants à la liste
    return param_ids


