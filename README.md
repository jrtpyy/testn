

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
	 
     
class xx(nn.Module):
    def __init__(self, in_ch,out_ch,kernel_size = 3, stride = 1,padding = 1, merge = False):
        super(xx, self).__init__()
        
        
        self.conv0 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=1,stride=stride)
        
        self.conv1 = nn.Conv2d(in_channels=out_ch, out_channels=out_ch, kernel_size=kernel_size,stride = stride, padding=padding)
        
        self.conv2 = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,stride = stride, padding=padding)
        
        
        if merge:
            conv_merge = nn.Conv2d(in_channels=in_ch, out_channels=out_ch, kernel_size=kernel_size,stride = stride, padding=padding)
            weight3,bias3 = merge_conv(conv0.weight,conv0.bias,conv1.weight,conv1.bias)
            
            weight3 = weight3 + self.conv2.weight.data
            bias3 = bias3 + self.conv2.bias.data
            
            conv_merge.weight.data.copy_(weight3)
            conv_merge.bias.data.copy_(bias3)
            
            self.conv_merge = conv_merge
            

    def forward(self, x):
        if merge:
            t = self.conv_merge(x)
        else:
            y0 = self.conv0(x)
            y1 = self.conv1(y0)
            y2 = self.conv2(x)
            
            y = y1 + y2
        return y
