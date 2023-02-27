
def sharpen(L: np.ndarray, lambda_: float, kernel: np.ndarray, eps: float = 1e-3,lf_blr:float = 0.4,lf_sp_strength:float = 0.5):


    h,w = np.shape(L)
    L_blur = refine_illumination_map_linear(L, 1.0, lambda_, kernel, eps)
    
    #decomp hf
    hf = L - L_blur
    hf_blur0 = cv2.GaussianBlur(hf, (5, 5), 1.5)
    hf0 = hf - hf_blur0
    hf_blur1 = cv2.GaussianBlur(hf0, (3, 3), 0.8)
    hf1 = hf0 - hf_blur1
    
    #sharpen lf and apply overshoot controlling

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
    
    #gradient region detect
    L_blur_lf = cv2.GaussianBlur(L_blur, (5, 5), 1.5)
    noise = np.random.randn()
    gradient0 = (var_calc(L_blur - L_blur_lf,2) + 0.01)*255/(var_calc(L_blur,2) + 0.01)
    
    hf_refine = refine_illumination_map_linear(hf+0.5, 1.0, lambda_, kernel, eps)
    hf_refine_dn = cv2.resize(hf_refine,(w//2,h//2))
    gradient1 = var_calc(hf_refine_dn,2)
    gradient1 = cv2.resize(gradient1,(w,h))
    
    gradient_f = gradient1*255*4/(gradient0 + 0.1)
    
    #############################
    tmp = np.clip((L**(1.0/2.2))*255,0,255)
    out_file_pth = "./tmp/L.bmp"
    cv2.imwrite(out_file_pth, tmp)
    
    tmp = np.clip((L_blur**(1.0/2.2))*255,0,255)
    out_file_pth = "./tmp/L_blur.bmp"
    cv2.imwrite(out_file_pth, tmp)
    
    tmp = np.clip(hf*255 + 128,0,255)
    out_file_pth = "./tmp/hf.bmp"
    cv2.imwrite(out_file_pth, tmp)
    
    tmp = np.clip((hf_refine-0.5)*255 + 128,0,255)
    out_file_pth = "./tmp/hf_refine.bmp"
    cv2.imwrite(out_file_pth, tmp)
    
    tmp = np.clip(gradient0,0,255)
    out_file_pth = "./tmp/gradient0.bmp"
    cv2.imwrite(out_file_pth, tmp)
    
    tmp = np.clip(gradient1,0,255)
    out_file_pth = "./tmp/gradient1.bmp"
    cv2.imwrite(out_file_pth, tmp)
    
    tmp = np.clip(gradient_f,0,255)
    out_file_pth = "./tmp/gradient_f.bmp"
    cv2.imwrite(out_file_pth, tmp)
    
    #############################
    

    return L_sp  
