import math
import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F

#===============================================================================================================================
#Our upscale and downscale blocks
#===============================================================================================================================
class up_block(nn.Module):
    def __init__(self, ich, mch, och, scale=2):
        super(up_block, self).__init__()

        if scale == 2:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, och, 3, stride=1, padding=1))
        elif scale == 4:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1))
        elif scale == 8:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.ConvTranspose2d(ich, mch, kernel_size=6, stride=4, padding=1))
        
        self.up_act = nn.ReLU(inplace=True)
        self.ex_det = nn.Sequential(
            nn.Conv2d(och, mch, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.ReLU(inplace=True),
            nn.Conv2d(mch, och, 3, stride=1, padding=1))
    
    def forward(self, x):
        x_up = self.up_conv(x)
        x_up_det = self.ex_det(self.up_act(x_up))
        
        x_sr = x_up + x_up_det
        
        return x_sr, x_up_det

class down_block(nn.Module):
    def __init__(self, ich, mch, och, scale=2):
        super(down_block, self).__init__()

        if scale == 2:
            self.down_conv = nn.Sequential(
                nn.Conv2d(ich, mch, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, och, 3, stride=1, padding=1))
        elif scale == 4:
            self.down_conv = nn.Sequential(
                nn.Conv2d(ich, mch, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, mch, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, och, 3, stride=1, padding=1))
        elif scale == 8:
            self.down_conv = nn.Sequential(
                nn.Conv2d(ich, mch, 3, stride=4, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, mch, 3, stride=2, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.ReLU(inplace=True),
                nn.Conv2d(mch, och, 3, stride=1, padding=1))
        
    def forward(self, x):
        out = self.down_conv(x)
        
        return out
        
class to_rgb(nn.Module):
    def __init__(self, ich):
        super(to_rgb, self).__init__()
        
        self.layer = nn.Conv2d(ich, 3, 1, stride=1, padding=0)
        
    def forward(self, x):
        out = self.layer(x)
        
        return out
        
#===============================================================================================================================
#Original blocks
#===============================================================================================================================
def init_weights(modules):
    pass
   
class MeanShift(nn.Module):
    def __init__(self, mean_rgb, sub):
        super(MeanShift, self).__init__()

        sign = -1 if sub else 1
        r = mean_rgb[0] * sign
        g = mean_rgb[1] * sign
        b = mean_rgb[2] * sign

        self.shifter = nn.Conv2d(3, 3, 1, 1, 0)
        self.shifter.weight.data = torch.eye(3).view(3, 3, 1, 1)
        self.shifter.bias.data   = torch.Tensor([r, g, b])

        # Freeze the mean shift layer
        for params in self.shifter.parameters():
            params.requires_grad = False

    def forward(self, x):
        x = self.shifter(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1, dilation=1):
        super(BasicBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad, dilation),
            nn.ReLU(inplace=True)
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class BasicBlockSig(nn.Module):
    def __init__(self,
                 in_channels, out_channels,
                 ksize=3, stride=1, pad=1):
        super(BasicBlockSig, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, ksize, stride, pad),
            nn.Sigmoid()
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class ResidualBlock(nn.Module):
    def __init__(self, 
                 in_channels, out_channels):
        super(ResidualBlock, self).__init__()

        self.body = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, 1, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, 1, 1),
        )

        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        out = F.relu(out + x)
        return out

class _UpsampleBlock(nn.Module):
    def __init__(self, 
				 n_channels, scale, 
				 group=1):
        super(_UpsampleBlock, self).__init__()

        modules = []
        if scale == 2 or scale == 4 or scale == 8:
            for _ in range(int(math.log(scale, 2))):
                modules += [nn.Conv2d(n_channels, 4*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
                modules += [nn.PixelShuffle(2)]
        elif scale == 3:
            modules += [nn.Conv2d(n_channels, 9*n_channels, 3, 1, 1, groups=group), nn.ReLU(inplace=True)]
            modules += [nn.PixelShuffle(3)]

        self.body = nn.Sequential(*modules)
        init_weights(self.modules)
        
    def forward(self, x):
        out = self.body(x)
        return out

class UpsampleBlock(nn.Module):
    def __init__(self, 
                 n_channels, scale, multi_scale, 
                 group=1):
        super(UpsampleBlock, self).__init__()

        if multi_scale:
            self.up2 = _UpsampleBlock(n_channels, scale=2, group=group)
            self.up3 = _UpsampleBlock(n_channels, scale=3, group=group)
            self.up4 = _UpsampleBlock(n_channels, scale=4, group=group)
        else:
            self.up =  _UpsampleBlock(n_channels, scale=scale, group=group)

        self.multi_scale = multi_scale

    def forward(self, x, scale):
        if self.multi_scale:
            if scale == 2:
                return self.up2(x)
            elif scale == 3:
                return self.up3(x)
            elif scale == 4:
                return self.up4(x)
        else:
            return self.up(x)

class CALayer(nn.Module):
    def __init__(self, channel, reduction=16):
        super(CALayer, self).__init__()

        self.avg_pool = nn.AdaptiveAvgPool2d(1)

        self.c1 = BasicBlock(channel , channel // reduction, 3, 1, 3, 3)
        self.c2 = BasicBlock(channel , channel // reduction, 3, 1, 5, 5)
        self.c3 = BasicBlock(channel , channel // reduction, 3, 1, 7, 7)
        self.c4 = BasicBlockSig((channel // reduction)*3, channel , 3, 1, 1)

    def forward(self, x):
        y = self.avg_pool(x)
        c1 = self.c1(y)
        c2 = self.c2(y)
        c3 = self.c3(y)
        c_out = torch.cat([c1, c2, c3], dim=1)
        y = self.c4(c_out)
        return x * y

class Block(nn.Module):
    def __init__(self, in_channels, out_channels, group=1):
        super(Block, self).__init__()

        self.r1 = ResidualBlock(in_channels, out_channels)
        self.r2 = ResidualBlock(in_channels*2, out_channels*2)
        self.r3 = ResidualBlock(in_channels*4, out_channels*4)
        self.g = BasicBlock(in_channels*8, out_channels, 1, 1, 0)
        self.ca = CALayer(in_channels)

    def forward(self, x):
        c0 =  x

        r1 = self.r1(c0)
        c1 = torch.cat([c0, r1], dim=1)
                
        r2 = self.r2(c1)
        c2 = torch.cat([c1, r2], dim=1)
               
        r3 = self.r3(c2)
        c3 = torch.cat([c2, r3], dim=1)

        g = self.g(c3)
        out = self.ca(g)
        return out
        
#===============================================================================================================================
#Modified model
#===============================================================================================================================
class DRLN_LaUD(nn.Module):
    def __init__(self, scale):
        super(DRLN_LaUD, self).__init__()

        self.scale = scale
        chs=64

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.head = nn.Conv2d(3, chs, 3, 1, 1)

        self.b1 = Block(chs, chs)
        self.b2 = Block(chs, chs)
        self.b3 = Block(chs, chs)
        self.b4 = Block(chs, chs)
        self.b5 = Block(chs, chs)
        self.b6 = Block(chs, chs)

        self.conv1 = nn.Conv2d(2*chs, chs, 3, 1, 1)
        self.b7 = Block(chs, chs)
        self.b8 = Block(chs, chs)
        self.b9 = Block(chs, chs)
        self.b10 = Block(chs, chs)
        self.b11 = Block(chs, chs)
        self.b12 = Block(chs, chs)

        self.conv2 = nn.Conv2d(3*chs, chs, 3, 1, 1)
        self.b13 = Block(chs, chs)
        self.b14 = Block(chs, chs)
        self.b15 = Block(chs, chs)
        self.b16 = Block(chs, chs)
        self.b17 = Block(chs, chs)
        self.b18 = Block(chs, chs)
        self.b19 = Block(chs, chs)
        self.b20 = Block(chs, chs)

        self.c1 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c2 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c3 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c4 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c5 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c6 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c7 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c8 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c9 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c10 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c11 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c12 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c13 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c14 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c15 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c16 = BasicBlock(chs*5, chs, 3, 1, 1)
        self.c17 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c18 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c19 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c20 = BasicBlock(chs*5, chs, 3, 1, 1)

        self.up1 = up_block(chs, chs, chs, scale=self.scale)
        self.up_rgb1 = to_rgb(chs)
        self.det_rgb1 = to_rgb(chs)
        self.down1 = down_block(chs, chs, chs, scale=self.scale)

        self.up2 = up_block(chs, chs, chs, scale=self.scale)
        self.up_rgb2 = to_rgb(chs)
        self.det_rgb2 = to_rgb(chs)
        self.down2 = down_block(chs, chs, chs, scale=self.scale)

        self.up3 = up_block(chs, chs, chs, scale=self.scale)
        self.up_rgb3 = to_rgb(chs)
        self.det_rgb3 = to_rgb(chs)
                
    def forward(self, x):
        sr_lst = []
        det_lst = []
        x = self.sub_mean(x)
        x = self.head(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        a1 = o3 + c0

        b4 = self.b4(a1)
        c4 = torch.cat([o3, b4], dim=1)
        o4 = self.c4(c4)
        b5 = self.b5(a1)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)
        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)
        a2 = o6 + a1

        out1 = a2 + x
        sr1, det1 = self.up1(out1)
        sr_lst.append(self.add_mean(self.up_rgb1(sr1)))
        det_lst.append(self.det_rgb1(det1))
        a2 = self.down1(sr1)

        b7 = self.b7(self.conv1(torch.cat([x, a2], dim=1)))
        c7 = torch.cat([o6, b7], dim=1)
        o7 = self.c7(c7)
        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)
        b9 = self.b9(o8)
        c9 = torch.cat([c8, b9], dim=1)
        o9 = self.c9(c9)
        a3 = o9 + a2

        b10 = self.b10(a3)
        c10 = torch.cat([o9, b10], dim=1)
        o10 = self.c10(c10)
        b11 = self.b11(o10)
        c11 = torch.cat([c10, b11], dim=1)
        o11 = self.c11(c11)
        b12 = self.b12(o11)
        c12 = torch.cat([c11, b12], dim=1)
        o12 = self.c12(c12)
        a4 = o12 + a3

        out2 = a4 + x
        sr2, det2 = self.up2(out2)
        sr_lst.append(self.add_mean(self.up_rgb2(sr2)))
        det_lst.append(self.det_rgb2(det2))
        a4 = self.down2(sr2)

        b13 = self.b13(self.conv2(torch.cat([x, a2, a4], dim=1)))
        c13 = torch.cat([o12, b13], dim=1)
        o13 = self.c13(c13)
        b14 = self.b14(o13)
        c14 = torch.cat([c13, b14], dim=1)
        o14 = self.c14(c14)
        b15 = self.b15(o14)
        c15 = torch.cat([c14, b15], dim=1)
        o15 = self.c15(c15)
        b16 = self.b16(o15)
        c16 = torch.cat([c15, b16], dim=1)
        o16 = self.c16(c16)
        a5 = o16 + a4

        b17 = self.b17(a5)
        c17 = torch.cat([o16, b17], dim=1)
        o17 = self.c17(c17)
        b18 = self.b18(o17)
        c18 = torch.cat([c17, b18], dim=1)
        o18 = self.c18(c18)
        b19 = self.b19(o18)
        c19 = torch.cat([c18, b19], dim=1)
        o19 = self.c19(c19)
        b20 = self.b20(o19)
        c20 = torch.cat([c19, b20], dim=1)
        o20 = self.c20(c20)
        a6 = o20 + a5

        out3 = a6 + x
        sr3, det3 = self.up3(out3)
        sr_lst.append(self.add_mean(self.up_rgb3(sr3)))
        det_lst.append(self.det_rgb3(det3))

        return sr_lst, det_lst

#===============================================================================================================================
#Original model
#===============================================================================================================================
class DRLN(nn.Module):
    def __init__(self, scale):
        super(DRLN, self).__init__()

        self.scale = scale
        chs=64

        self.sub_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=True)
        self.add_mean = MeanShift((0.4488, 0.4371, 0.4040), sub=False)
        
        self.head = nn.Conv2d(3, chs, 3, 1, 1)

        self.b1 = Block(chs, chs)
        self.b2 = Block(chs, chs)
        self.b3 = Block(chs, chs)
        self.b4 = Block(chs, chs)
        self.b5 = Block(chs, chs)
        self.b6 = Block(chs, chs)
        self.b7 = Block(chs, chs)
        self.b8 = Block(chs, chs)
        self.b9 = Block(chs, chs)
        self.b10 = Block(chs, chs)
        self.b11 = Block(chs, chs)
        self.b12 = Block(chs, chs)
        self.b13 = Block(chs, chs)
        self.b14 = Block(chs, chs)
        self.b15 = Block(chs, chs)
        self.b16 = Block(chs, chs)
        self.b17 = Block(chs, chs)
        self.b18 = Block(chs, chs)
        self.b19 = Block(chs, chs)
        self.b20 = Block(chs, chs)

        self.c1 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c2 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c3 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c4 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c5 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c6 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c7 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c8 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c9 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c10 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c11 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c12 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c13 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c14 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c15 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c16 = BasicBlock(chs*5, chs, 3, 1, 1)
        self.c17 = BasicBlock(chs*2, chs, 3, 1, 1)
        self.c18 = BasicBlock(chs*3, chs, 3, 1, 1)
        self.c19 = BasicBlock(chs*4, chs, 3, 1, 1)
        self.c20 = BasicBlock(chs*5, chs, 3, 1, 1)

        self.upsample = UpsampleBlock(chs, self.scale , multi_scale=False)
        self.tail = nn.Conv2d(chs, 3, 3, 1, 1)
                
    def forward(self, x):
        x = self.sub_mean(x)
        x = self.head(x)
        c0 = o0 = x

        b1 = self.b1(o0)
        c1 = torch.cat([c0, b1], dim=1)
        o1 = self.c1(c1)
        
        b2 = self.b2(o1)
        c2 = torch.cat([c1, b2], dim=1)
        o2 = self.c2(c2)
        
        b3 = self.b3(o2)
        c3 = torch.cat([c2, b3], dim=1)
        o3 = self.c3(c3)
        a1 = o3 + c0

        b4 = self.b4(a1)
        c4 = torch.cat([o3, b4], dim=1)
        o4 = self.c4(c4)
 
        b5 = self.b5(a1)
        c5 = torch.cat([c4, b5], dim=1)
        o5 = self.c5(c5)

        b6 = self.b6(o5)
        c6 = torch.cat([c5, b6], dim=1)
        o6 = self.c6(c6)
        a2 = o6 + a1

        b7 = self.b7(a2)
        c7 = torch.cat([o6, b7], dim=1)
        o7 = self.c7(c7)

        b8 = self.b8(o7)
        c8 = torch.cat([c7, b8], dim=1)
        o8 = self.c8(c8)

        b9 = self.b9(o8)
        c9 = torch.cat([c8, b9], dim=1)
        o9 = self.c9(c9)
        a3 = o9 + a2

        b10 = self.b10(a3)
        c10 = torch.cat([o9, b10], dim=1)
        o10 = self.c10(c10)


        b11 = self.b11(o10)
        c11 = torch.cat([c10, b11], dim=1)
        o11 = self.c11(c11)

        b12 = self.b12(o11)
        c12 = torch.cat([c11, b12], dim=1)
        o12 = self.c12(c12)
        a4 = o12 + a3


        b13 = self.b13(a4)
        c13 = torch.cat([o12, b13], dim=1)
        o13 = self.c13(c13)

        b14 = self.b14(o13)
        c14 = torch.cat([c13, b14], dim=1)
        o14 = self.c14(c14)


        b15 = self.b15(o14)
        c15 = torch.cat([c14, b15], dim=1)
        o15 = self.c15(c15)

        b16 = self.b16(o15)
        c16 = torch.cat([c15, b16], dim=1)
        o16 = self.c16(c16)
        a5 = o16 + a4


        b17 = self.b17(a5)
        c17 = torch.cat([o16, b17], dim=1)
        o17 = self.c17(c17)

        b18 = self.b18(o17)
        c18 = torch.cat([c17, b18], dim=1)
        o18 = self.c18(c18)


        b19 = self.b19(o18)
        c19 = torch.cat([c18, b19], dim=1)
        o19 = self.c19(c19)

        b20 = self.b20(o19)
        c20 = torch.cat([c19, b20], dim=1)
        o20 = self.c20(c20)
        a6 = o20 + a5

        b_out = a6 + x
        out = self.upsample(b_out, scale=self.scale )

        out = self.tail(out)
        f_out = self.add_mean(out)

        return f_out
        