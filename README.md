
def sharpen(L: np.ndarray, lambda_: float, kernel: np.ndarray, eps: float = 1e-3,lf_blr:float = 0.4,lf_sp_strength:float = 0.5):


    L_blur = refine_illumination_map_linear_decomp(L, lambda_, kernel, eps, lf_blr)
    
    #decomp hf
    hf = L - L_blur
    hf_blur0 = cv2.GaussianBlur(hf, (5, 5), 1.5)
    hf0 = hf - hf_blur0
    hf_blur1 = cv2.GaussianBlur(hf0, (3, 3), 0.8)
    hf1 = hf0 - hf_blur1
    
    #sharpen lf and apply overshoot controlling
    h,w = np.shape(L)
    lf_r = 2
    lf_d = 2*lf_r + 1
    L_blur_lf = cv2.GaussianBlur(L_blur, (lf_d, lf_d), 1.5)
    
    ref_r = lf_r+1
    ref_d = 2*ref_r+1
    refs = np.zeros((h,w,ref_d*ref_d))
    L_blur_pad = np.pad(L_blur, ((ref_r, ref_r),(ref_r,ref_r)), 'reflect')
    L_blur_pad = np.expand_dims(L_blur_pad, axis=-1)
    for j in range (ref_d):
        for i in range (ref_d):
            idx = j*ref_d + i
            refs[:,:,idx:idx+1] = L_blur_pad[j:j+h,i:i+w,:]
    
    max_v = np.max(refs,axis=-1)
    min_v = np.min(refs,axis=-1)
    
    L_blur_hf = L_blur - L_blur_lf
    L_blur_sp = L_blur + L_blur_hf*lf_sp_strength
    L_blur_sp = np.where(L_blur_sp > max_v,max_v,L_blur_sp)
    L_blur_sp = np.where(L_blur_sp < min_v,min_v,L_blur_sp)
    
    #apply sharpen
    L_sp = L_blur_sp + (hf_blur0*1.1 + hf_blur1*1.35 + hf1 * 1.6)
    

    return L_sp  
