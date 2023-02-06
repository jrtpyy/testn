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
    
    
def addNoise(I,bl,K,B):

    i8 = I*255.0
    
    sigma2 = K*(i8 - bl) + B
    
    mu = np.zeros_like(I)
    std = np.sqrt(sigma2)
    noise = np.random.normal(mu, std)
    
    I_addn = i8 + noise
    
    noise_map = K*(I_addn - bl) + B
    
    I_addn = I_addn/255.0
    overExp_region = np.array(I > 0.98)
    I_addn = I * overExp_region + (1.0-overExp_region)*I_addn
    
    
    #noise_map = noise_map/255.0/255.0  #match the image data range 0~1
    return I_addn
    
    
    
    
bl = 1.0/16.0
iso = 6400

K,B,a,b = noise_model(iso)
print("noise model : ",K,B,a,b)

for luma in range (0,20,1):
    I = np.zeros((256,256))
    I = I + luma/255.0 + bl
    I_noise = add_poisson_gauss_noise(I, bl,a,b)
    
    I_noise_old = addNoise(I,bl,K,B)
    
    tmp = np.clip(I*255,0,255)
    out_file_pth = "./tmp/I_%d.bmp"%(luma)
    cv2.imwrite(out_file_pth, tmp)
    
    tmp = np.clip(I_noise*255,0,255)
    out_file_pth = "./tmp/I_%d_noise.bmp"%(luma)
    cv2.imwrite(out_file_pth, tmp)
    
    
    I_noise_vst = vst(I_noise, bl,a,b)
    I_noise_vst_old = vst(I_noise_old, bl,a,b)
    
    noise_var = np.var(I_noise_vst)
    noise_var_old = np.var(I_noise_vst_old)
    
    print("luma,noise_var = ",luma,noise_var,noise_var_old,noise_var/noise_var_old)

    
    
    
    

def vst(I, bl,a,b):
    g = bl
    z_a = (I-g)/a
    z_b = b/(a**2)
    I_vst = 2*np.sqrt(np.maximum((I-g)/a+3.0/8.0+z_b,0))
    
    
    return I_vst
    
    
    
    
    
