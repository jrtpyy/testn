
#apply gamma for all data including  negative number
def apply_gamma(x,gamma_in = 1.0/3.5,b_in = 0.0031308):
    
        
    gamma = nn.Parameter(torch.tensor(gamma_in),requires_grad = False)
    b = nn.Parameter(torch.tensor(b_in),requires_grad = False)
    
    a = 1./(1./(b**gamma*(1.-gamma))-1.)
    k0 = (1+a)*gamma*b**(gamma-1.)
    
    mask0 = x < b
    mask1 = x > 1.0
    
    out = torch.zeros_like(x)
    out[mask0] = k0*x
    out[~mask0] = (1+a)*torch.pow(torch.maximum(x[~mask0],b),gamma).to(x)-a
    
    k1 = (1+a)*gamma
    out[mask1] = k1*x-k1+1

    return out  
    
    
    
