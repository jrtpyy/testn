
def warp_torch(x,xymap,w,h):
    
    vgrid_x = 2.0 * xymap[:, 0,:, :] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * xymap[:, 1,:, :] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode='nearest', padding_mode='reflection')
         
    return  output   


    def __init__(self,TVLoss_weight=1):
        super(TVLoss,self).__init__()
        self.TVLoss_weight = TVLoss_weight

    def forward(self,x):
        batch_size = x.size()[0]
        h_x = x.size()[2]
        w_x = x.size()[3]

        h_tv = x[:,:,1:,:]-x[:,:,:h_x-1,:]
        w_tv = x[:,:,:,1:]-x[:,:,:,:w_x-1]
        h_tv = torch.pad(h_tv,pad = (0,0,0,1),mode='constant',value = 0)
        w_tv = torch.pad(w_tv,pad = (0,1,0,0),mode='constant',value = 0)
        
        out = torch.stack((h_tv,w_tv),dim = -1)
        return out
