import torch
import torch.nn as nn

#===============================================================================================================================
class resblock(nn.Module):
    def __init__(self, ich, och, down=False):
        super(resblock, self).__init__()
        
        if down:
            self.convlayer = nn.Sequential(
                nn.Conv2d(ich, och, 3, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.Conv2d(och, och, 3, stride=1, padding=1))
            self.skip = nn.Conv2d(ich, och, 1, stride=2, padding=0)
        else:
            self.convlayer = nn.Sequential(
                nn.Conv2d(ich, och, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
                nn.Conv2d(och, och, 3, stride=1, padding=1))
            self.skip = nn.Conv2d(ich, och, 1, stride=1, padding=0)
        
    def forward(self, x):
        out = self.convlayer(x) + self.skip(x)
        
        return out
    
class ext_block(nn.Module):
    def __init__(self, ich, och=256):
        super(ext_block, self).__init__()
        
        self.convlayer = nn.Sequential(nn.Conv2d(ich, och, 3, stride=1, padding=1), nn.LeakyReLU(0.02))
        
        self.layer1 = resblock(och, och, down=False)
        self.layer2 = resblock(och, och, down=False)
        self.layer3 = resblock(och, och, down=False)
        self.layer4 = resblock(och, och, down=False)
        
    def forward(self, x):
        
        x = self.convlayer(x)
        
        x = x + self.layer1(x)
        x = x + self.layer2(x)
        x = x + self.layer3(x)
        out = x + self.layer4(x)
        
        return out

class to_rgb(nn.Module):
    def __init__(self, ich):
        super(to_rgb, self).__init__()
        
        self.layer = nn.Conv2d(ich, 3, 1, stride=1, padding=0)
        
    def forward(self, x):
        out = self.layer(x)
        
        return out

#===============================================================================================================================
# Basic model
#===============================================================================================================================  
class up_block(nn.Module):
    def __init__(self, ich, mch, och, scale=2):
        super(up_block, self).__init__()

        if scale == 2:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.Conv2d(mch, och, 3, stride=1, padding=1))
        elif scale == 4:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.ConvTranspose2d(mch, och, kernel_size=4, stride=2, padding=1))
        elif scale == 8:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.ConvTranspose2d(mch, och, kernel_size=6, stride=4, padding=1))
        
        self.up_act = nn.LeakyReLU(0.02)
        self.ex_det = nn.Sequential(
            nn.Conv2d(och, mch, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
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
                nn.Conv2d(ich, mch, 3, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.Conv2d(mch, och, 3, stride=1, padding=1))
        elif scale == 4:
            self.down_conv = nn.Sequential(
                nn.Conv2d(ich, mch, 3, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.Conv2d(mch, och, 3, stride=2, padding=1))
        elif scale == 8:
            self.down_conv = nn.Sequential(
                nn.Conv2d(ich, mch, 3, stride=4, padding=1), nn.LeakyReLU(0.02),
                nn.Conv2d(mch, och, 3, stride=2, padding=1))
        
    def forward(self, x):
        out = self.down_conv(x)
        
        return out
        
#===============================================================================================================================
class LaUD(nn.Module):
    def __init__(self, rudp=3, scale=2, ch=64):
        super(LaUD, self).__init__()
        """
        Args:
            rudp: Number of upscaling steps in RUDP(The maximum supported is 4 in this code).
            scale: Upscaling factor.
            ch: Number of channels. The default is 64.
        """
        
        self.rudp = rudp
        
        self.ext1 = ext_block(ich=3, och=4*ch)
        self.up1 = up_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
        self.to_rgb1_sr = to_rgb(ich=4*ch)
        self.to_rgb1_det = to_rgb(ich=4*ch)
        
        if self.rudp > 1:
            self.down1 = down_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            
            self.ext2 = ext_block(ich=3 + 4*ch, och=4*ch)
            self.up2 = up_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            self.to_rgb2_sr = to_rgb(ich=4*ch)
            self.to_rgb2_det = to_rgb(ich=4*ch)
            
        if self.rudp > 2:
            self.down2 = down_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            
            self.ext3 = ext_block(ich=3 + 4*ch + 4*ch, och=4*ch)
            self.up3 = up_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            self.to_rgb3_sr = to_rgb(ich=4*ch)
            self.to_rgb3_det = to_rgb(ich=4*ch)

        if self.rudp > 3:
            self.down3 = down_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            
            self.ext4 = ext_block(ich=3 + 4*ch + 4*ch + 4*ch, och=4*ch)
            self.up4 = up_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            self.to_rgb4_sr = to_rgb(ich=4*ch)
            self.to_rgb4_det = to_rgb(ich=4*ch)
            
    def forward(self, x):
        sr_lst = []
        det_lst = []
        
        feat = self.ext1(x)
        sr, up_det = self.up1(feat)
        sr_lst.append(self.to_rgb1_sr(sr))
        det_lst.append(self.to_rgb1_det(up_det))

        if self.rudp > 1:
            sr_down = self.down1(sr)
            lr_cat1 = torch.cat((x, sr_down), dim=1)

            feat = self.ext2(lr_cat1)
            sr, up_det = self.up2(feat)
            sr_lst.append(self.to_rgb2_sr(sr))
            det_lst.append(self.to_rgb2_det(up_det))

        if self.rudp > 2:
            sr_down = self.down2(sr)
            lr_cat2 = torch.cat((lr_cat1, sr_down), dim=1)

            feat = self.ext3(lr_cat2)
            sr, up_det = self.up3(feat)
            sr_lst.append(self.to_rgb3_sr(sr))
            det_lst.append(self.to_rgb3_det(up_det))

        if self.rudp > 3:
            sr_down = self.down3(sr)
            lr_cat3 = torch.cat((lr_cat2, sr_down), dim=1)

            feat = self.ext4(lr_cat3)
            sr, up_det = self.up4(feat)
            sr_lst.append(self.to_rgb4_sr(sr))
            det_lst.append(self.to_rgb4_det(up_det))

        return sr_lst, det_lst

#===============================================================================================================================
# Ablation models
#===============================================================================================================================  
class up_block_wo_det(nn.Module):
    def __init__(self, ich, mch, och, scale=2):
        super(up_block_wo_det, self).__init__()

        if scale == 2:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.Conv2d(mch, och, 3, stride=1, padding=1), nn.LeakyReLU(0.02))
        elif scale == 4:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.ConvTranspose2d(mch, och, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02))
        elif scale == 8:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02),
                nn.ConvTranspose2d(mch, och, kernel_size=6, stride=4, padding=1), nn.LeakyReLU(0.02))

        self.conv = nn.Sequential(
            nn.Conv2d(och, mch, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(mch, och, 3, stride=1, padding=1))

    def forward(self, x):
        up = self.up_conv(x)
        sr = self.conv(up)

        return sr
        
class LaUD_wo_det(nn.Module):
    def __init__(self, rudp=3, scale=2, ch=64):
        super(LaUD_wo_det, self).__init__()
        """
        Args are the same as LaUD.
        """

        self.rudp = rudp

        self.ext1 = ext_block(ich=3, och=4*ch)
        self.up1 = up_block_wo_det(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
        self.to_rgb1_sr = to_rgb(ich=4*ch)

        if self.rudp > 1:
            self.down1 = down_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            
            self.ext2 = ext_block(ich=3 + 4*ch, och=4*ch)
            self.up2 = up_block_wo_det(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            self.to_rgb2_sr = to_rgb(ich=4*ch)
            
        if self.rudp > 2:
            self.down2 = down_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            
            self.ext3 = ext_block(ich=3 + 4*ch + 4*ch, och=4*ch)
            self.up3 = up_block_wo_det(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            self.to_rgb3_sr = to_rgb(ich=4*ch)

        if self.rudp > 3:
            self.down3 = down_block(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            
            self.ext4 = ext_block(ich=3 + 4*ch + 4*ch + 4*ch, och=4*ch)
            self.up4 = up_block_wo_det(ich=4*ch, mch=4*ch, och=4*ch, scale=scale)
            self.to_rgb4_sr = to_rgb(ich=4*ch)
            
    def forward(self, x):
        sr_lst = []
        
        feat = self.ext1(x)
        sr = self.up1(feat)
        sr_lst.append(self.to_rgb1_sr(sr))
            
        if self.rudp > 1:
            sr_down = self.down1(sr)
            lr_cat1 = torch.cat((x, sr_down), dim=1)
            
            feat = self.ext2(lr_cat1)
            sr = self.up2(feat)
            sr_lst.append(self.to_rgb2_sr(sr))
            
        if self.rudp > 2:
            sr_down = self.down2(sr)
            lr_cat2 = torch.cat((lr_cat1, sr_down), dim=1)
            
            feat = self.ext3(lr_cat2)
            sr = self.up3(feat)
            sr_lst.append(self.to_rgb3_sr(sr))

        if self.rudp > 3:
            sr_down = self.down3(sr)
            lr_cat3 = torch.cat((lr_cat2, sr_down), dim=1)
            
            feat = self.ext4(lr_cat3)
            sr = self.up4(feat)
            sr_lst.append(self.to_rgb4_sr(sr))

        return sr_lst

#===============================================================================================================================
class PlainNet(nn.Module):
    def __init__(self, ch=64):
        super(PlainNet, self).__init__()
        """
        A plain model without our LP-based detail loss and RUDP.
        We match the number of parameters of this model similar to LaUD in our experiments.
        """

        self.ext = nn.Sequential(
            ext_block(ich=3, och=4*ch),
            ext_block(ich=4*ch, och=4*ch),
            ext_block(ich=4*ch, och=4*ch),
            ext_block(ich=4*ch, och=4*ch),
            ext_block(ich=4*ch, och=4*ch))

        self.up = nn.Sequential(
            nn.ConvTranspose2d(4*ch, 4*ch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(4*ch, 4*ch, 3, stride=1, padding=1))
        
        self.up_act = nn.LeakyReLU(0.02)
        self.up_res = nn.Sequential(
            nn.Conv2d(4*ch, 4*ch, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(4*ch, 4*ch, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(4*ch, 4*ch, 3, stride=1, padding=1))
        self.to_rgb = to_rgb(ich=4*ch)

    def forward(self, x):
        sr_lst = []
            
        feat = self.ext(x)
        up = self.up(feat)
        up_res = self.up_res(self.up_act(up))
        sr_lst.append(self.to_rgb(up+up_res))

        return sr_lst

class PlainNet_wdet(nn.Module):
    def __init__(self, ch=64):
        super(PlainNet_wdet, self).__init__()
        """
        A plain model without RUDP, but with our LP-based detail loss applied.
        We match the number of parameters of this model similar to LaUD in our experiments.
        """
        
        self.ext = nn.Sequential(
            ext_block(ich=3, och=4*ch),
            ext_block(ich=4*ch, och=4*ch),
            ext_block(ich=4*ch, och=4*ch),
            ext_block(ich=4*ch, och=4*ch),
            ext_block(ich=4*ch, och=4*ch))
            
        self.up = nn.Sequential(
            nn.ConvTranspose2d(4*ch, 4*ch, kernel_size=4, stride=2, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(4*ch, 4*ch, 3, stride=1, padding=1))
        
        self.up_act = nn.LeakyReLU(0.02)
        self.up_res = nn.Sequential(
            nn.Conv2d(4*ch, 4*ch, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(4*ch, 4*ch, 3, stride=1, padding=1), nn.LeakyReLU(0.02),
            nn.Conv2d(4*ch, 4*ch, 3, stride=1, padding=1))
        self.up_to_rgb = to_rgb(ich=4*ch)
        self.det_to_rgb = to_rgb(ich=4*ch)

    def forward(self, x):
        sr_lst = []
        det_lst = []
            
        feat = self.ext(x)
        up = self.up(feat)
        up_res = self.up_res(self.up_act(up))
        sr_lst.append(self.up_to_rgb(up+up_res))
        det_lst.append(self.det_to_rgb(up_res))

        return sr_lst, det_lst
