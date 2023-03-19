vgrid_x = 2.0 * xymap[:, 0,:, :] / max(w - 1, 1) - 1.0
    vgrid_y = 2.0 * xymap[:, 1,:, :] / max(h - 1, 1) - 1.0
    vgrid_scaled = torch.stack((vgrid_x, vgrid_y), dim=3)
    output = F.grid_sample(x, vgrid_scaled, mode='nearest', padding_mode='reflection')



#for conv2(conv1)
def merge_conv(conv1_weight,conv1_bias,conv2_weight,conv2_bias):

    #k1 k1 out mid * k2 k2 mid in -> k k out in
	  weight3 = torch.matmul(conv2_weight.permute(2,3,0,1),conv1_weight.permute(2,3,0,1))
	  weight3 = weight3.permute(2,3,0,1)
	  if(conv1_bias != NULL):
	      bias3 = torch.matmul(conv2_weight.sum(2,3),conv1_bias) + conv2_bias
	  else:
	      bias3 = conv2_bias
	      
	  return weight3,bias3
	 
def test1():
    conv1 = nn.Conv2d(in_channels=16, out_channels=16, kernel_size=3,stride=1, padding=1)# out_chn,in_chn, k ,k for transpose conv
	conv2 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1)
	
	conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=3,stride=1, padding=1)
	weight3,bias3 = merge_conv(conv1.weight,conv1.bias,conv2.weight,conv2.bias)
	
	conv3.weight.data.copy_(weight3)
	conv3.bias.data.copy_(bias3)
	
	x = torch.randn(1,16,4,4)
	
	y0 = conv2(conv1(x))
	y1 = conv3(x)
	
	print(y0)
	print(y1)
	
def test2():#for ConvTranspose2d
    conv1 = nn.ConvTranspose2d(16, 16, kernel_size=2, stride=2) # in_chn, out_chn,k ,k for transpose conv
	conv2 = nn.Conv2d(in_channels=16, out_channels=1, kernel_size=1)
	
	conv3 = nn.Conv2d(in_channels=16, out_channels=4, kernel_size=1,stride=1, padding=1)
	weight,bias = merge_conv(conv1.weight.permute(1,0,2,3),conv1.bias,conv2.weight,conv2.bias)
	
	weight3 = torch.cat([weight[:,:,0::2,0::2],weight[:,:,0::2,1::2],weight[:,:,1::2,0::2],weight[:,:,1::2,1::2]],dim = 0)
	
	conv3.weight.data.copy_(weight3)
	conv3.bias.data.copy_(bias)
	
	x = torch.randn(1,16,4,4)
	
	y0 = conv2(conv1(x))
	y1 = conv3(x)
	
	print(y0)
	print(y1)	
	
