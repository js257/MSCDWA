import torch

DEVICE = torch.device("cuda:0")

"""
# MOSEI SETTING
ACOUSTIC_DIM = 74
VISUAL_DIM = 35
TEXT_DIM = 768
"""

#SETTING
ACOUSTIC_DIM = 74   #MOSI:74  MOSEI:74  
VISUAL_DIM = 27 #MOSI:27 MOSEI:35
TEXT_DIM = 768
FUSION_DIM = 512  #MOSI:512  MOSEI: 256
A_reduced_dim = 40 # MOSI: 40, MOSEI: 32
V_reduced_dim = 20  # MOSI: 20, MOSEI: 27
A_PCA_dir = '/media/pca_pkl/MOSI_a-40-pca.pkl'
V_PCA_dir = '/media/pca_pkl/MOSI_v-20-pca.pkl'


