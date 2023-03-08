
def warp_torch(x,xymap,w,h):
    
    vgrid_x = 2.0 * xymap[:, 0,:, :] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * xymap[:, 1,:, :] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode='nearest', padding_mode='reflection')
         
    return  output   
