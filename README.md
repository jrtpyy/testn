

#calc ratio for the fine detail(complex detail) pixels 
class fine_detail_w(nn.Module):
    def __init__(self,channels = 1,eps = 2):
        super(fine_detail_w, self).__init__()
        self.eps = eps/255.0
        self.gf_d = 5
        self.gaussianfilter = GaussianSmoothing(channels, self.gf_d,1.2)
        
    def get_mask(self,I):
        with torch.no_grad():
            t = torch.clamp(x - 1.0/16.0,0,1.0)**(1.0/2.2)
                        
            r = self.gf_d//2
            tmp = F.pad(I,(r,r,r,r), mode='reflect')
            lf = self.gaussianfilter(tmp)
            
            hf = I - lf
            
            h_x = hf.size()[2]
            w_x = hf.size()[3]

            h_tv = hf[:,:,1:,:]-hf[:,:,:h_x-1,:]
            w_tv = hf[:,:,:,1:]-hf[:,:,:,:w_x-1]
            h_tv = F.pad(h_tv,pad = (0,0,0,1),mode='constant',value = 0)
            w_tv = F.pad(w_tv,pad = (0,1,0,0),mode='constant',value = 0)
            
            grad = torch.minimum(torch.abs(h_tv),torch.abs(w_tv))
            
            zeros = torch.zeors_like(grad)
            ones = zeros +1
            
            num = torch.where(grad > self.eps,ones,zeros)
            
            w = torch.mean(num,dim = (1,2,3))
            
        return w


    def forward(self, x, mul = 10):
        w = self.get_mask(x.detach()).detach()
        w = torch.clamp(w*mul,0,30)
        return w
        
        
        
class exceptional_mask_calc(nn.Module):
    def __init__(self,channels = 1,eps = 0.01):
        super(exceptional_mask_calc, self).__init__()
        self.gf_d = 7
        self.gaussianfilter = GaussianSmoothing(channels, self.gf_d,1.5)
        
        self.kernel_size = 5
        self.boxfilter0 = boxfilter(channels, kernel_size)
        
        self.eps = eps
        
    def get_mask(self,net_r,label):
        with torch.no_grad():
        
            ##################################################
            t = torch.clamp(net_r - 1.0/16.0,0,1.0)**(1.0/2.2)
                        
            r = self.gf_d//2
            tmp = F.pad(net_r,(r,r,r,r), mode='reflect')
            lf = self.gaussianfilter(tmp)
            
            hf = net_r - lf
            
            
            r = self.kernel_size//2
            input = F.pad(hf, (r, r, r, r), mode='reflect')
            mean = self.boxfilter0(input)
            
            input = F.pad(hf*hf, (r, r, r, r), mode='reflect')
            sqr_mean = self.boxfilter0(input)
            var = sqr_mean - mean*mean
            
            zeros_map = torch.zeros_like(var)
            var_net = torch.maximum(var,zeros_map)
            
            ##################################################
            t = torch.clamp(label - 1.0/16.0,0,1.0)**(1.0/2.2)
                        
            r = self.gf_d//2
            tmp = F.pad(label,(r,r,r,r), mode='reflect')
            lf = self.gaussianfilter(tmp)
            
            hf = label - lf
            
            
            r = self.kernel_size//2
            input = F.pad(hf, (r, r, r, r), mode='reflect')
            mean = self.boxfilter0(input)
            
            input = F.pad(hf*hf, (r, r, r, r), mode='reflect')
            sqr_mean = self.boxfilter0(input)
            var = sqr_mean - mean*mean
            
            zeros_map = torch.zeros_like(var)
            var_label = torch.maximum(var,zeros_map)
            
            
            ################
            t = var_net/(var_label + self.eps)
            
            ones = torch.ones_like(t)
            
            t = torch.maximum(t,ones)
            
            
            
        return t


    def forward(self,net_r,label):
        exceptional_mask = self.get_mask(net_r.detach(),labeldetach()).detach()
        
        return exceptional_mask    
