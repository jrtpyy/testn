# testn


import torch
import torch.nn as nn
import numpy as np


_K_A_ = 0.00074424
_K_B_ = 0
_B_A_ = 0.00000154
_B_B_ = 0.00161675
_B_C_ = 0

def noise_model(iso):

    K = _K_A_ * iso + _K_B_
    B = _B_A_*iso*iso + _B_B_*iso + _B_C_
    
    a = K/255.0
    b = B /255.0/255.0
    return a,b

def add_poisson_gauss_noise(I, bl,a,b):
    
    #div a to get photon
    photon = (I-bl)/a
    
    #to match the poisson noise value level, b is variance ,so div a**2
    std_gauss = np.sqrt(max(b,0)/(a**2))
    
    overExp_region = np.array(I > 0.98)
    I_add_noise = np.random.poisson(lam = photon,size = np.shape(I))  + np.random.normal(loc=bl/a,scale=std_gauss,size=np.shape(I))
    
    #*a to Restore Image Values 
    I_add_noise = I_add_noise * a
    I_add_noise = I * overExp_region + (1.0-overExp_region)*I_add_noise
    
    return I_add_noise
    
    
    
    
