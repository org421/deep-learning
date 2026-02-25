"""
Module de corrélation utilisé dans CMSeg-Net.

Ce module permet de mesurer la similarité entre les pixels
afin de détecter les zones copiées-collées (copy-move).
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class ZeroWindow:
    """
    Classe qui applique une fenêtre gaussienne sur la corrélation.

    Le but est de réduire l'auto-corrélation, c'est-à-dire éviter
    qu'un pixel soit trop corrélé avec lui-même.
    """
    def __init__(self):
        # Dictionnaire pour stocker les fenêtres déjà calculées
        # afin d'éviter de les recalculer plusieurs fois
        self.store = {}

    def __call__(self, x_in, h, w, rat_s=0.1):
        # Sigma correspond à la taille de la fenêtre gaussienne
        sigma = h * rat_s, w * rat_s
        b, c, h2, w2 = x_in.shape
        
        # Clé utilisée pour identifier une fenêtre déjà calculée
        key = str(x_in.shape) + str(rat_s)
        
        if key not in self.store:
            # Création des indices de lignes et de colonnes
            ind_r = torch.arange(h2).float()
            ind_c = torch.arange(w2).float()
            ind_r = ind_r.view(1, 1, -1, 1).expand_as(x_in)
            ind_c = ind_c.view(1, 1, 1, -1).expand_as(x_in)

            # Calcul des coordonnées centrales pour chaque pixel
            c_indices = torch.from_numpy(np.indices((h, w))).float()
            c_ind_r = c_indices[0].reshape(-1)
            c_ind_c = c_indices[1].reshape(-1)

            cent_r = c_ind_r.reshape(1, c, 1, 1).expand_as(x_in)
            cent_c = c_ind_c.reshape(1, c, 1, 1).expand_as(x_in)

            # Fonction gaussienne utilisée pour créer la fenêtre
            def fn_gauss(x, u, s):
                return torch.exp(-(x - u) ** 2 / (2 * s ** 2))
            
            # Calcul de la fenêtre gaussienne selon les lignes et colonnes
            gaus_r = fn_gauss(ind_r, cent_r, sigma[0])
            gaus_c = fn_gauss(ind_c, cent_c, sigma[1])
            
            # Création du masque final (inversion de la gaussienne)
            out_g = 1 - gaus_r * gaus_c
            out_g = out_g.to(x_in.device)
            
            # Sauvegarde du masque pour réutilisation
            self.store[key] = out_g
        else:
            # Récupération du masque déjà calculé
            out_g = self.store[key]
        
        # Application du masque sur la matrice de corrélation
        out = out_g * x_in
        return out


def get_topk(x, k=10, dim=-3):
    """
    Récupère les k valeurs les plus élevées
    le long d'une dimension donnée.
    """
    val, _ = torch.topk(x, k=k, dim=dim)
    return val


class Corr(nn.Module):
    """
    Module principal de corrélation (CoSA).

    Il calcule l'auto-corrélation entre tous les pixels d'une image.
    Les régions copiées-collées présentent en général
    de fortes corrélations entre elles.
    
    Args:
        topk: nombre de corrélations les plus fortes conservées
    """
    def __init__(self, topk=3):
        super().__init__()
        
        # Nombre de corrélations maximales à conserver
        self.topk = topk
        
        # Module utilisé pour supprimer l'auto-corrélation
        self.zero_window = ZeroWindow()
        
        # Paramètre appris pour amplifier la corrélation
        self.alpha = nn.Parameter(torch.tensor(5., dtype=torch.float32))

    def forward(self, x):
        b, c, h1, w1 = x.shape
        h2 = h1
        w2 = w1

        # Normalisation des features pour stabiliser la corrélation
        xn = F.normalize(x, p=2, dim=-3)
        
        # Calcul de la matrice de corrélation entre tous les pixels
        x_aff_o = torch.matmul(
            xn.permute(0, 2, 3, 1).view(b, -1, c), 
            xn.view(b, c, -1)
        )

        # Suppression de l'auto-corrélation grâce à la ZeroWindow
        x_aff = self.zero_window(
            x_aff_o.view(b, -1, h1, w1), h1, w1, rat_s=0.05
        ).reshape(b, h1 * w1, h2 * w2)
        
        # Application d'un double softmax pour mettre en valeur
        # les corrélations les plus importantes
        x_c = (
            F.softmax(x_aff * self.alpha, dim=-1) *
            F.softmax(x_aff * self.alpha, dim=-2)
        )
        x_c = x_c.reshape(b, h1, w1, h2, w2)

        # Extraction des top-k corrélations les plus fortes
        xc_o = x_c.view(b, h1 * w1, h2, w2)
        val = get_topk(xc_o, k=self.topk, dim=-3)

        return val
