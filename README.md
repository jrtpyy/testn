# 3p
import numpy as np
import cv2
from scipy.spatial import distance
from scipy.ndimage.filters import convolve
from scipy.sparse import diags, csr_matrix
from scipy.sparse.linalg import spsolve
# project
from utils import get_sparse_neighbor



def create_spacial_affinity_kernel(spatial_sigma: float, size: int = 15):
    """Create a kernel (`size` * `size` matrix) that will be used to compute the he spatial affinity based Gaussian weights.

    Arguments:
        spatial_sigma {float} -- Spatial standard deviation.

    Keyword Arguments:
        size {int} -- size of the kernel. (default: {15})

    Returns:
        np.ndarray - `size` * `size` kernel
    """
    kernel = np.zeros((size, size))
    for i in range(size):
        for j in range(size):
            kernel[i, j] = np.exp(-0.5 * (distance.euclidean((i, j), (size // 2, size // 2)) ** 2) / (spatial_sigma ** 2))

    return kernel


def compute_smoothness_weights(L: np.ndarray, x: int, kernel: np.ndarray, eps: float = 1e-3):
    """Compute the smoothness weights used in refining the illumination map optimization problem.

    Arguments:
        L {np.ndarray} -- the initial illumination map to be refined.
        x {int} -- the direction of the weights. Can either be x=1 for horizontal or x=0 for vertical.
        kernel {np.ndarray} -- spatial affinity matrix

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability. (default: {1e-3})

    Returns:
        np.ndarray - smoothness weights according to direction x. same dimension as `L`.
    """
    Lp = cv2.Sobel(L, cv2.CV_64F, int(x == 1), int(x == 0), ksize=1)
    T = convolve(np.ones_like(L), kernel, mode='constant')
    T = T / (np.abs(convolve(Lp, kernel, mode='constant')) + eps)
    return T / (np.abs(Lp) + eps)


def fuse_multi_exposure_images(im: np.ndarray, under_ex: np.ndarray, over_ex: np.ndarray,
                               bc: float = 1, bs: float = 1, be: float = 1):
    """perform the exposure fusion method used in the DUAL paper.

    Arguments:
        im {np.ndarray} -- input image to be enhanced.
        under_ex {np.ndarray} -- under-exposure corrected image. same dimension as `im`.
        over_ex {np.ndarray} -- over-exposure corrected image. same dimension as `im`.

    Keyword Arguments:
        bc {float} -- parameter for controlling the influence of Mertens's contrast measure. (default: {1})
        bs {float} -- parameter for controlling the influence of Mertens's saturation measure. (default: {1})
        be {float} -- parameter for controlling the influence of Mertens's well exposedness measure. (default: {1})

    Returns:
        np.ndarray -- the fused image. same dimension as `im`.
    """
    merge_mertens = cv2.createMergeMertens(bc, bs, be)
    images = [np.clip(x * 255, 0, 255).astype("uint8") for x in [im, under_ex, over_ex]]
    fused_images = merge_mertens.process(images)
    return fused_images


def refine_illumination_map_linear(L: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """Refine the illumination map based on the optimization problem described in the two papers.
       This function use the sped-up solver presented in the LIME paper.

    Arguments:
        L {np.ndarray} -- the illumination map to be refined.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3}).

    Returns:
        np.ndarray -- refined illumination map. same shape as `L`.
    """
    # compute smoothness weights
    L_blur = cv2.GaussianBlur(L, (7, 7), 2.5) 
    L_hf = L - L_blur
    print(np.shape(L_hf),np.shape(kernel))
    wx = compute_smoothness_weights(L_hf, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L_hf, x=0, kernel=kernel, eps=eps)

    n, m = L.shape
    L_1d = L.copy().flatten()

    # compute the five-point spatially inhomogeneous Laplacian matrix
    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    # solve the linear system
    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

    # gamma correction
    L_refined = np.clip(L_refined, eps, 1) ** gamma

    return L_refined
    


#use hf for smoothness weights calc, avoid contour artifacts in gradient regions
#used for sharpen
def refine_illumination_map_linear_decomp(L: np.ndarray,  lambda_: float, kernel: np.ndarray, eps: float = 1e-3, lf_blr:float = 0.4):
    """Refine the illumination map based on the optimization problem described in the two papers.
       This function use the sped-up solver presented in the LIME paper.

    Arguments:
        L {np.ndarray} -- the illumination map to be refined.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3}).

    Returns:
        np.ndarray -- refined illumination map. same shape as `L`.
    """
    # compute smoothness weights
    L_blur = cv2.GaussianBlur(L, (7, 7), 2.0) 
    L_hf = L - L_blur
    print(np.shape(L_hf),np.shape(kernel))
    wx = compute_smoothness_weights(L_hf, x=1, kernel=kernel, eps=eps)
    wy = compute_smoothness_weights(L_hf, x=0, kernel=kernel, eps=eps)

    n, m = L.shape
    L_1d = L.copy().flatten()

    # compute the five-point spatially inhomogeneous Laplacian matrix
    row, column, data = [], [], []
    for p in range(n * m):
        diag = 0
        for q, (k, l, x) in get_sparse_neighbor(p, n, m).items():
            weight = wx[k, l] if x else wy[k, l]
            row.append(p)
            column.append(q)
            data.append(-weight)
            diag += weight
        row.append(p)
        column.append(p)
        data.append(diag)
    F = csr_matrix((data, (row, column)), shape=(n * m, n * m))

    # solve the linear system
    Id = diags([np.ones(n * m)], [0])
    A = Id + lambda_ * F
    L_refined = spsolve(csr_matrix(A), L_1d, permc_spec=None, use_umfpack=True).reshape((n, m))

    hf = L - L_refined
    hf_blur = cv2.GaussianBlur(hf, (7, 7), 2.0)
    L_refined = L_refined + hf_blur * lf_blr #recover some contrast info

    return L_refined


def var_calc(I,r):
    d = 2*r+1
    mean = cv2.boxFilter(I,-1,(d,d))
    sqr_mean = cv2.boxFilter(I*I,-1,(d,d))
    var = sqr_mean - mean*mean
    var = np.maximum(var,0)
    std = np.sqrt(var)*4096
    return std
    

def sharpen(L: np.ndarray, lambda_: float, kernel: np.ndarray, eps: float = 1e-3,lf_sp_strength:float = 1.1):
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
    L_sp = L_blur_sp + (hf_blur0*1.0 + hf_blur1*1.35 + hf1 * 1.6)
    
    #gradient region detect
    noise = np.random.randn(h,w)
    L_blur_t = L_blur + noise/255.0
    L_blur_lf = cv2.GaussianBlur(L_blur_t, (5, 5), 1.5)
    
    gradient0 = (var_calc(L_blur_t - L_blur_lf,2) + 0.1)*255/(var_calc(L_blur_t,2) + 0.1)
    
    hf_refine = refine_illumination_map_linear(hf+0.5, 1.0, lambda_, kernel, eps)
    hf_refine_dn = cv2.resize(hf_refine,(w//2,h//2))
    gradient1 = var_calc(hf_refine_dn,2)
    gradient1 = cv2.resize(gradient1,(w,h))
    
    gradient_f = gradient1*128*2/(gradient0 + 0.1)
    
    p0 = [10,0.0]
    p1 = [45,1.0]
    ones = np.ones_like(gradient_f)
    gL = p0[0] * ones
    gH = p1[0] * ones
    mask_v = np.maximum(gradient_f,gL)
    mask_v = np.minimum(gradient_f,gH)
    blr = p0[1] + (mask_v - p0[0])*(p1[1] - p0[1])/(p1[0] - p0[0])
    
    L_sp = L_sp * (1.0 - blr) + L * blr
    
    #############################
    tmp = np.clip(blr*255,0,255)
    out_file_pth = "./tmp/blr.bmp"
    cv2.imwrite(out_file_pth, tmp)
    
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
    
    
def correct_underexposure(im: np.ndarray, gamma: float, lambda_: float, kernel: np.ndarray, eps: float = 1e-3):
    """correct underexposudness using the retinex based algorithm presented in DUAL and LIME paper.

    Arguments:
        im {np.ndarray} -- input image to be corrected.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem.
        kernel {np.ndarray} -- spatial affinity matrix.

    Keyword Arguments:
        eps {float} -- small constant to avoid computation instability (default: {1e-3})

    Returns:
        np.ndarray -- image underexposudness corrected. same shape as `im`.
    """

    # first estimation of the illumination map
    L = np.max(im, axis=-1)
    # illumination refinement

    L_refined_gamma = refine_illumination_map_linear(L, gamma, lambda_, kernel, eps)

    # correct image underexposure
    L_refined_3d = np.repeat(L_refined_gamma[..., None], 3, axis=-1)
    im_corrected = im / L_refined_3d
    return im_corrected

# TODO: resize image if too large, optimization take too much time


def enhance_image_exposure(im: np.ndarray, gamma: float, lambda_: float, dual: bool = True, sigma: int = 3,
                           bc: float = 1, bs: float = 1, be: float = 1, eps: float = 1e-3, sp: float = 0.5):
    """Enhance input image, using either DUAL method, or LIME method. For more info, please see original papers.

    Arguments:
        im {np.ndarray} -- input image to be corrected.
        gamma {float} -- gamma correction factor.
        lambda_ {float} -- coefficient to balance the terms in the optimization problem (in DUAL and LIME).

    Keyword Arguments:
        dual {bool} -- boolean variable to indicate enhancement method to be used (either DUAL or LIME) (default: {True})
        sigma {int} -- Spatial standard deviation for spatial affinity based Gaussian weights. (default: {3})
        bc {float} -- parameter for controlling the influence of Mertens's contrast measure. (default: {1})
        bs {float} -- parameter for controlling the influence of Mertens's saturation measure. (default: {1})
        be {float} -- parameter for controlling the influence of Mertens's well exposedness measure. (default: {1})
        eps {float} -- small constant to avoid computation instability (default: {1e-3})

    Returns:
        np.ndarray -- image exposure enhanced. same shape as `im`.
    """
    # create spacial affinity kernel
    kernel = create_spacial_affinity_kernel(sigma)

    # correct underexposudness
    im_normalized = im.astype(float) / 255.
    under_corrected = correct_underexposure(im_normalized, gamma, lambda_, kernel, eps)

    if dual:
        # correct overexposure and merge if DUAL method is selected
        inv_im_normalized = 1 - im_normalized
        over_corrected = 1 - correct_underexposure(inv_im_normalized, gamma, lambda_, kernel, eps)

        # fuse images
        im_corrected = fuse_multi_exposure_images(im_normalized, under_corrected, over_corrected, bc, bs, be)
    else:
        im_corrected = under_corrected
    
    
    if sp >= 0:
        g = im_normalized[:,:,1:2]
        #rdiff = im_corrected[:,:,0:1] - g
        #bdiff = im_corrected[:,:,2:3] - g

        g_sp = sharpen(g[:,:,0], lambda_, kernel, eps,sp)
        g_sp = np.expand_dims(g_sp,axis=-1)
        hf = g_sp - g
        im_corrected = im_corrected + hf
        
        
    # convert to 8 bits and returns
    return np.clip(im_corrected * 255, 0, 255).astype("uint8")
