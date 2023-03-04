
class TVLoss(nn.Module):
    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]
        count_h = self._tensor_size(x[:,:,1:,:])
        count_w = self._tensor_size(x[:,:,:,1:])
        h_tv = torch.pow((x[:,:,1:,:]-x[:,:,:h_x-1,:]),2).sum()
        w_tv = torch.pow((x[:,:,:,1:]-x[:,:,:,:w_x-1]),2).sum()
        return self.TVLoss_weight*2*(h_tv/count_h+w_tv/count_w)/batch_size

    def _tensor_size(self,t):
        return t.size()[1]*t.size()[2]*t.size()[3]
        
        
        
        import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
from math import exp

def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size//2)**2/float(2*sigma**2)) for x in range(window_size)])
    return gauss/gauss.sum()

def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window

def _ssim(img1, img2, window, window_size, channel, size_average = True):
    mu1 = F.conv2d(img1, window, padding = window_size//2, groups = channel)
    mu2 = F.conv2d(img2, window, padding = window_size//2, groups = channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1*mu2

    sigma1_sq = F.conv2d(img1*img1, window, padding = window_size//2, groups = channel) - mu1_sq
    sigma2_sq = F.conv2d(img2*img2, window, padding = window_size//2, groups = channel) - mu2_sq
    sigma12 = F.conv2d(img1*img2, window, padding = window_size//2, groups = channel) - mu1_mu2

    C1 = 0.01**2
    C2 = 0.03**2

    ssim_map = ((2*mu1_mu2 + C1)*(2*sigma12 + C2))/((mu1_sq + mu2_sq + C1)*(sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)

class SSIM(torch.nn.Module):
    def __init__(self, window_size = 11, size_average = True):
        super(SSIM, self).__init__()
        self.window_size = window_size
        self.size_average = size_average
        self.channel = 1
        self.window = create_window(window_size, self.channel)

    def forward(self, img1, img2):
        (_, channel, _, _) = img1.size()

        if channel == self.channel and self.window.data.type() == img1.data.type():
            window = self.window
        else:
            window = create_window(self.window_size, channel)
            
            if img1.is_cuda:
                window = window.cuda(img1.get_device())
            window = window.type_as(img1)
            
            self.window = window
            self.channel = channel


        return _ssim(img1, img2, window, self.window_size, channel, self.size_average)

def ssim(img1, img2, window_size = 11, size_average = True):
    (_, channel, _, _) = img1.size()
    window = create_window(window_size, channel)
    
    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)
    
    return _ssim(img1, img2, window, window_size, channel, size_average)


#################
def get_mask(self,x,y):
        with torch.no_grad():
            b,c,h,w = x.size()
            mask = np.zeros((b,1*w*h))
            diff = torch.abs(x-y).sum(dim=1).view(b,-1)
            diff_sort = [diff[i].sort(descending=True) for i in range (b)]
            diff_np = diff.cpu().numpy()
            hard_th_idx = int(self.p0*w*h)
            for i in range(b):
                hard_th = diff_sort[i][0][hard_th_idx].item()
                mask[i] = diff_np[i] > hard_th
                
            rand_hart_th_idx = int(self.p1*w*h)
            mask_rand = np.zeros((b,1*w*h))
            for i in range(b):
                mask_rand[i,0:rand_hart_th_idx] = 1
                np.random.shuffle(mask_rand[i])
                
            mask = mask + mask_rand
            mask = mask.reshape(b,-1,h,w)
        return torch.from_numpy(mask).cuda()


    def forward(self, x, y,weight):
        mask = self.get_mask(x.detach(),y.detach())
        loss = torch.mean(torch.abs(x-y)*mask*weight)
        return loss
        
        
####################
def forward(ctx, input):
        ctx.save_for_backward(input)
        i = input.clone()
        ge = torch.ge(input,0).float()
        output = i + (ge.data - i.data)

        return output


    @staticmethod
    def backward(ctx, grad_output): 
        input, = ctx.saved_tensors
        input_grad = grad_output.clone()
        
        ones = torch.ones_like(input)
        zeros = torch.zeros_like(input)
        output_grad = torch.maximum(zeros, torch.abs(ones - input) ) * input_grad

        return output_grad
