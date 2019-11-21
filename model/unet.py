import torch
from torch import nn

## Needs work.

class CoarseUNetGenerator(nn.Module):
    def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
        super(CoarseUNetGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        
        ## Contracting Path (left side)
        # Double 3x3 Convolutions (unpadded) 
        # each followed by a RELU and a 2x2 Max Pooling operation
        # with stride 2 for downsampling.
        
        self.downConv1 = doubleConv(5, 64)
        self.downConv2 = doubleConv(64, 128)
        self.downConv3 = doubleConv(128, 256)
        self.downConv4 = doubleConv(256, 512)
        
        self.maxPool = nn.MaxPool2d(2, stride=2)
        
        ## Expansive Path (right side)
        self.upConv4 = doubleConv(512 + 512, 512)
        self.upConv3 = doubleConv(512 + 256, 256)
        self.upConv2 = doubleConv(256 + 128, 128)
        self.upConv1 = doubleConv(128, 64)
        self.outConv = doubleConv(64, 5)
        
        self.upSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
    def forward(self, input, mask):
        
        # For indicating the boundaries of images
        ones = torch.ones(input.size(0), 1, input.size(2), input.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
        
        temp = torch.cat([input, ones, mask], dim=1)
        downConv1 = self.downConv1(temp)
        input = self.maxPool(downConv1)
        
        downConv2 = self.downConv2(input)
        input = self.maxPool(downConv2)
        
        downConv3 = self.downConv3(input)
        input = self.maxPool(downConv3)
        
        downConv4 = self.downConv4(input)
        input = self.maxPool(downConv4)
        
        input = self.upSample(input)
        input = torch.cat([input, downConv4], dim=1)
        
        upConv4 = self.upConv4(input)
        input = self.upSample(upConv4)
        input = torch.cat([input, downConv3], dim=1)
        
        upConv3 = self.upConv3(input)
        input = self.upSample(upConv3)
        input = torch.cat([input, downConv2], dim=1)
        
        upConv2 = self.upConv2(input)
        input = self.upSample(upConv2)
        input = torch.cat([input, downConv1], dim=1)
        
        input = self.upConv1(input)
        
        input_stage_1 = torch.clamp(input, -1., 1.)     
                
        return input_stage_1
        
        
        
    

def doubleConv(in_channel, out_channel, batch_norm=True):
    conv_block = []
    
    conv_block.append(nn.Conv2d(in_channel, out_channel, 3, padding=0))
    conv_block.append(nn.ReLU(inplace=True))
    
    if batch_norm:
        conv_block.append(nn.BatchNorm2d(out_channel))
    
    conv_block.append(nn.Conv2d(out_channel, out_channel, 3, padding=0))
    conv_block.append(nn.ReLU(inplace=True))
    
    if batch_norm:
        conv_block.append(nn.BatchNorm2d(out_channel))
        
    return nn.Sequential(*conv_block)