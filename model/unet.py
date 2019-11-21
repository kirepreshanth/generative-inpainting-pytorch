import torch
from torch import nn
from model.cxt_att import ContextualAttention

##https://github.com/a7med12345/Cycle-GAN-with-Unet-as-GENERATOR/blob/f3d93f7aab5083b20e70d3ecb720c4ac31e6e630/MyUnet.py#L5

class ConvBlock(torch.nn.Module):
    def __init__(self, input_channel, output_channel,batch_normalization=True):
        super(ConvBlock, self).__init__()

        self.conv1 = torch.nn.Conv2d(input_channel, output_channel, 3, padding=1)
        self.bn1 = torch.nn.BatchNorm2d(output_channel)
        self.conv2 = torch.nn.Conv2d(output_channel,output_channel,3,padding=1)
        self.bn2 = torch.nn.BatchNorm2d(output_channel)
        self.relu = torch.nn.ReLU()
        self.batch_normalization = batch_normalization

    def forward(self,x):
        x = self.conv1(x)
        if self.batch_normalization:
            x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        if self.batch_normalization:
            x = self.bn2(x)

        x=self.relu(x)

        return x


class DownSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(DownSample, self).__init__()
        self.down_sample = torch.nn.MaxPool2d(factor, factor)

    def forward(self,x):
        return self.down_sample(x)


class UpSample(torch.nn.Module):
    def __init__(self, factor=2):
        super(UpSample, self).__init__()
        self.up_sample = torch.nn.Upsample(scale_factor = factor, mode='bilinear')

    def forward(self,x):
        return self.up_sample(x)


class CropConcat(torch.nn.Module):
    def __init__(self,crop = True):
        super(CropConcat, self).__init__()
        self.crop = crop

    def do_crop(self,x, tw, th):
        b,c,w, h = x.size()
        x1 = int(round((w - tw) / 2.))
        y1 = int(round((h - th) / 2.))
        return x[:,:,x1:x1 + tw, y1:y1 + th]

    def forward(self,x,y):
        b, c, h, w = y.size()
        if self.crop:
            x = self.do_crop(x,h,w)
        return torch.cat((x,y),dim=1)


class UpBlock(torch.nn.Module):
    def __init__(self,input_channel, output_channel,batch_normalization=True,downsample = False):
        super(UpBlock, self).__init__()
        self.downsample = downsample
        self.conv = ConvBlock(input_channel,output_channel,batch_normalization=batch_normalization)
        self.downsampling = DownSample()

    def forward(self,x):
        x1 = self.conv(x)
        if self.downsample:
            x = self.downsampling(x1)
        else:
            x = x1
        return x,x1

class DownBlock(torch.nn.Module):
    def __init__(self,input_channel, output_channel,batch_normalization=True,Upsample = False):
        super(DownBlock, self).__init__()
        self.Upsample = Upsample
        self.conv = ConvBlock(input_channel,output_channel,batch_normalization=batch_normalization)
        self.upsampling = UpSample()
        self.crop = CropConcat()

    def forward(self,x,y):
        if self.Upsample:
            x = self.upsampling(x)
        x = self.crop(y,x)
        x = self.conv(x)
        return x


class CoarseUNetGenerator(torch.nn.Module):
    def __init__(self,input_channel,output_channel, use_cuda=True, device_ids=None):
        super(CoarseUNetGenerator, self).__init__()
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        
        #Down Blocks
        self.conv_block1 = ConvBlock(input_channel,64)
        self.conv_block2 = ConvBlock(64,128)
        self.conv_block3 = ConvBlock(128,256)
        self.conv_block4 = ConvBlock(256,512)
        self.conv_block5 = ConvBlock(512,1024)

        #Up Blocks
        self.conv_block6 = ConvBlock(1024+512, 512)
        self.conv_block7 = ConvBlock(512+256, 256)
        self.conv_block8 = ConvBlock(256+128, 128)
        self.conv_block9 = ConvBlock(128+64, 64)

        #Last convolution
        self.last_conv = torch.nn.Conv2d(64,input_channel,1)

        self.crop = CropConcat()

        self.downsample = DownSample()
        self.upsample =   UpSample()

    def forward(self,x, mask):
        # For indicating the boundaries of images
        ones = torch.ones(x.size(0), 1, x.size(2), x.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
            
        x1 = self.conv_block1(x)#torch.cat([x, ones, mask], dim=1))
        x = self.downsample(x1)
        x2 = self.conv_block2(x)
        x= self.downsample(x2)
        x3 = self.conv_block3(x)
        x= self.downsample(x3)
        x4 = self.conv_block4(x)
        x = self.downsample(x4)
        x5 = self.conv_block5(x)

        x = self.upsample(x5)
        x = self.crop(x4, x)
        x = self.conv_block6(x)

        x = self.upsample(x)
        x = self.crop(x3,x)
        x = self.conv_block7(x)

        x= self.upsample(x)
        x= self.crop(x2,x)
        x = self.conv_block8(x)

        x = self.upsample(x)
        x = self.crop(x1,x)
        x = self.conv_block9(x)


        x = self.last_conv(x)

        return x

class FineUNetGenerator(torch.nn.Module):
    def __init__(self,input_channel,output_channel, use_cuda=True, device_ids=None):
        super(FineUNetGenerator, self).__init__()
        
        self.use_cuda = use_cuda
        self.device_ids = device_ids
        
        #Down Blocks
        self.conv_block1 = ConvBlock(input_channel,64)
        self.conv_block2 = ConvBlock(64,128)
        self.conv_block3 = ConvBlock(128,256)
        self.conv_block4 = ConvBlock(256,512)
        self.conv_block5 = ConvBlock(512,1024)

        # Attention Branch
        self.pmconv_block1 = ConvBlock(input_channel,64)
        self.pmconv_block2 = ConvBlock(64,128)
        self.pmconv_block3 = ConvBlock(128,256)
        self.pmconv_block4 = ConvBlock(256,512)
        self.pmconv_block5 = ConvBlock(512,1024)
        self.contextul_attention = ContextualAttention(ksize=3, stride=1, rate=2, fuse_k=3, softmax_scale=10,
                                                       fuse=True, use_cuda=self.use_cuda, device_ids=self.device_ids)

        self.pmconv_block6 = ConvBlock(1024,1024)
        
        #Up Blocks
        self.conv_block6 = ConvBlock(1024+512, 512)
        self.conv_block7 = ConvBlock(512+256, 256)
        self.conv_block8 = ConvBlock(256+128, 128)
        self.conv_block9 = ConvBlock(128+64, 64)

        #Last convolution
        self.last_conv = torch.nn.Conv2d(64,input_channel,1)

        self.crop = CropConcat()

        self.downsample = DownSample()
        self.upsample =   UpSample()

    def forward(self, xin, x_stage1, mask):
        x1_inpaint = x_stage1 * mask + xin * (1. - mask)
        # For indicating the boundaries of images
        ones = torch.ones(xin.size(0), 1, xin.size(2), xin.size(3))
        if self.use_cuda:
            ones = ones.cuda()
            mask = mask.cuda()
            
        xnow = torch.cat([x1_inpaint, ones, mask], dim=1)
        x1 = self.conv_block1(x1_inpaint)#xnow)
        x = self.downsample(x1)
        x2 = self.conv_block2(x)
        x= self.downsample(x2)
        x3 = self.conv_block3(x)
        x= self.downsample(x3)
        x4 = self.conv_block4(x)
        x = self.downsample(x4)
        x5 = self.conv_block5(x)
        x_hallu = x5
        
        # attention branch        
        
        x5 = self.pmconv_block1(x1_inpaint)
        x5 = self.pmconv_block2(x5)
        x5 = self.pmconv_block3(x5)
        x5 = self.pmconv_block4(x5)
        x5 = self.pmconv_block5(x5)
        x5, offset_flow = self.contextul_attention(x5, x5, mask)
        x5 = self.pmconv_block6(x5)
        pm = x5
        x = torch.cat([x_hallu, pm], dim=1)
        # Right side
        
        x = self.upsample(x5)
        x = self.crop(x4, x)
        x = self.conv_block6(x)

        x = self.upsample(x)
        x = self.crop(x3,x)
        x = self.conv_block7(x)

        x= self.upsample(x)
        x= self.crop(x2,x)
        x = self.conv_block8(x)

        x = self.upsample(x)
        x = self.crop(x1,x)
        x = self.conv_block9(x)


        x = self.last_conv(x)

        return x, offset_flow


## Needs work.

# class CoarseUNetGenerator(nn.Module):
#     def __init__(self, input_dim, cnum, use_cuda=True, device_ids=None):
#         super(CoarseUNetGenerator, self).__init__()
#         self.use_cuda = use_cuda
#         self.device_ids = device_ids
        
#         ## Contracting Path (left side)
#         # Double 3x3 Convolutions (unpadded) 
#         # each followed by a RELU and a 2x2 Max Pooling operation
#         # with stride 2 for downsampling.
        
#         self.downConv1 = doubleConv(5, 64)
#         self.downConv2 = doubleConv(64, 128)
#         self.downConv3 = doubleConv(128, 256)
#         self.downConv4 = doubleConv(256, 512)
        
#         self.maxPool = nn.MaxPool2d(2, stride=2)
        
#         ## Expansive Path (right side)
#         self.upConv4 = doubleConv(512 + 512, 512)
#         self.upConv3 = doubleConv(512 + 256, 256)
#         self.upConv2 = doubleConv(256 + 128, 128)
#         self.upConv1 = doubleConv(128, 64)
#         self.outConv = doubleConv(64, 5)
        
#         self.upSample = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
    
#     def forward(self, input, mask):
        
#         # For indicating the boundaries of images
#         ones = torch.ones(input.size(0), 1, input.size(2), input.size(3))
#         if self.use_cuda:
#             ones = ones.cuda()
#             mask = mask.cuda()
        
#         temp = torch.cat([input, ones, mask], dim=1)
#         downConv1 = self.downConv1(temp)
#         input = self.maxPool(downConv1)
        
#         downConv2 = self.downConv2(input)
#         input = self.maxPool(downConv2)
        
#         downConv3 = self.downConv3(input)
#         input = self.maxPool(downConv3)
        
#         downConv4 = self.downConv4(input)
#         input = self.maxPool(downConv4)
        
#         input = self.upSample(input)
#         input = torch.cat([input, downConv4], dim=1)
        
#         upConv4 = self.upConv4(input)
#         input = self.upSample(upConv4)
#         input = torch.cat([input, downConv3], dim=1)
        
#         upConv3 = self.upConv3(input)
#         input = self.upSample(upConv3)
#         input = torch.cat([input, downConv2], dim=1)
        
#         upConv2 = self.upConv2(input)
#         input = self.upSample(upConv2)
#         input = torch.cat([input, downConv1], dim=1)
        
#         input = self.upConv1(input)
        
#         input_stage_1 = torch.clamp(input, -1., 1.)     
                
#         return input_stage_1  
        

# def doubleConv(in_channel, out_channel, batch_norm=True):
#     conv_block = []
    
#     conv_block.append(nn.Conv2d(in_channel, out_channel, 3, padding=0))
#     conv_block.append(nn.ReLU(inplace=True))
    
#     if batch_norm:
#         conv_block.append(nn.BatchNorm2d(out_channel))
    
#     conv_block.append(nn.Conv2d(out_channel, out_channel, 3, padding=0))
#     conv_block.append(nn.ReLU(inplace=True))
    
#     if batch_norm:
#         conv_block.append(nn.BatchNorm2d(out_channel))
        
#     return nn.Sequential(*conv_block)