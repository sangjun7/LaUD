import torch
import torch.nn as nn
import torch.nn.functional as F

#===============================================================================================================================
#Our upscale block
#===============================================================================================================================
class up_block(nn.Module):
    def __init__(self, ich, mch, och, scale=2):
        super(up_block, self).__init__()

        if scale == 2:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.PReLU(),
                nn.Conv2d(mch, och, 3, stride=1, padding=1))
        elif scale == 4:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.PReLU(),
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1))
        elif scale == 8:
            self.up_conv = nn.Sequential(
                nn.ConvTranspose2d(ich, mch, kernel_size=4, stride=2, padding=1), nn.PReLU(),
                nn.ConvTranspose2d(ich, mch, kernel_size=6, stride=4, padding=1))
        
        self.up_act = nn.PReLU()
        self.ex_det = nn.Sequential(
            nn.Conv2d(och, mch, 3, stride=1, padding=1), nn.PReLU(),
            nn.Conv2d(mch, mch, 3, stride=1, padding=1), nn.PReLU(),
            nn.Conv2d(mch, och, 3, stride=1, padding=1))
    
    def forward(self, x):
        x_up = self.up_conv(x)
        x_up_det = self.ex_det(self.up_act(x_up))
        
        x_sr = x_up + x_up_det
        
        return x_sr, x_up_det

#===============================================================================================================================
#Original blocks
#===============================================================================================================================
class ConvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(ConvBlock, self).__init__()

        self.conv = torch.nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.conv(x)

        return self.act(out)


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, bias=True):
        super(DeconvBlock, self).__init__()

        self.deconv = torch.nn.ConvTranspose2d(input_size, output_size, kernel_size, stride, padding, bias=bias)

        self.act = torch.nn.PReLU()

    def forward(self, x):
        out = self.deconv(x)

        return self.act(out)


class UpBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(UpBlock, self).__init__()

        self.conv1 = DeconvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        hr = self.conv1(x)
        lr = self.conv2(hr)
        residue = self.local_weight1(x) - lr
        h_residue = self.conv3(residue)
        hr_weight = self.local_weight2(hr)
        return hr_weight + h_residue


class DownBlock(torch.nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding):
        super(DownBlock, self).__init__()

        self.conv1 = ConvBlock(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv2 = DeconvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.conv3 = ConvBlock(output_size, output_size, kernel_size, stride, padding, bias=True)
        self.local_weight1 = ConvBlock(input_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)
        self.local_weight2 = ConvBlock(output_size, output_size, kernel_size=1, stride=1, padding=0, bias=True)

    def forward(self, x):
        lr = self.conv1(x)
        hr = self.conv2(lr)
        residue = self.local_weight1(x) - hr
        l_residue = self.conv3(residue)
        lr_weight = self.local_weight2(lr)
        return lr_weight + l_residue
        
class Space_attention(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Space_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale

        self.K = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)
        if kernel_size == 1:
            self.local_weight = nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)

    def forward(self, x):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        if self.stride > 1:
            Q = self.pool(Q)
        else:
            Q = Q
        V = self.V(x)
        if self.stride > 1:
            V = self.pool(V)
        else:
            V = V
        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)
        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.matmul(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)

        vector = torch.matmul(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        W = self.local_weight(O)
        output = x + W
        return output

class Time_attention(nn.Module):
    def __init__(self, input_size, output_size, kernel_size, stride, padding, scale):
        super(Time_attention, self).__init__()

        self.input_size = input_size
        self.output_size = output_size
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        self.scale = scale

        self.K = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.Q = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.V = nn.Conv2d(input_size, output_size, kernel_size, stride, padding, bias=True)
        self.pool = nn.MaxPool2d(kernel_size=self.scale + 2, stride=self.scale, padding=1)
        if kernel_size == 1:
            self.local_weight = nn.Conv2d(output_size, input_size, kernel_size, stride, padding,
                                                bias=True)
        else:
            self.local_weight = nn.ConvTranspose2d(output_size, input_size, kernel_size, stride, padding,
                                                         bias=True)

    def forward(self, x, y):
        batch_size = x.size(0)
        K = self.K(x)
        Q = self.Q(x)
        if self.stride > 1:
            Q = self.pool(Q)
        else:
            Q = Q
        V = self.V(y)
        if self.stride > 1:
            V = self.pool(V)
        else:
            V = V
        V_reshape = V.view(batch_size, self.output_size, -1)
        V_reshape = V_reshape.permute(0, 2, 1)

        Q_reshape = Q.view(batch_size, self.output_size, -1)

        K_reshape = K.view(batch_size, self.output_size, -1)
        K_reshape = K_reshape.permute(0, 2, 1)

        KQ = torch.matmul(K_reshape, Q_reshape)
        attention = F.softmax(KQ, dim=-1)
        vector = torch.matmul(attention, V_reshape)
        vector_reshape = vector.permute(0, 2, 1).contiguous()
        O = vector_reshape.view(batch_size, self.output_size, x.size(2) // self.stride, x.size(3) // self.stride)
        W = self.local_weight(O)
        output = y + W
        return output
        
#===============================================================================================================================
#Modified model
#===============================================================================================================================
class ABPNv5_LaUD(nn.Module):
    def __init__(self, input_dim, dim):
        super(ABPNv5_LaUD, self).__init__()

        self.feat1 = nn.Sequential(nn.Conv2d(input_dim, 2 * dim, kernel_size=3, stride=1, padding=1), 
                                   nn.PReLU())
        self.SA0 = Space_attention(2 * dim, 2 * dim, kernel_size=1, stride=1, padding=0, scale=1)
        self.feat2 = nn.Sequential(nn.Conv2d(2 * dim, dim, kernel_size=3, stride=1, padding=1), 
                                   nn.PReLU())
        # BP 1
        self.up1 = up_block(dim, dim, dim, scale=4)
        self.down1 = DownBlock(dim, dim, 6, 4, 1)
        self.SA1 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        # BP 2
        self.up2 = up_block(dim, dim, dim, scale=4)
        self.down2 = DownBlock(dim, dim, 6, 4, 1)
        self.SA2 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        # BP 3
        self.weight_up1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.up3 = up_block(dim, dim, dim, scale=4)
        self.weight_down1 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.down3 = DownBlock(dim, dim, 6, 4, 1)
        self.SA3 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        # BP 4
        self.weight_up2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.up4 = up_block(dim, dim, dim, scale=4)
        self.weight_down2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.down4 = DownBlock(dim, dim, 6, 4, 1)
        self.SA4 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        # BP5
        self.weight_up3 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.up5 = up_block(dim, dim, dim, scale=4)
        self.weight_down3 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.down5 = DownBlock(dim, dim, 6, 4, 1)
        self.SA5 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)

        # BP6
        self.weight_up4 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.up6 = up_block(dim, dim, dim, scale=4)
        self.weight_down4 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.down6 = DownBlock(dim, dim, 6, 4, 1)
        self.SA6 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        # BP7
        self.weight_up5 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.up7 = up_block(dim, dim, dim, scale=4)
        self.weight_down5 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.down7 = DownBlock(dim, dim, 6, 4, 1)
        self.SA7 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        # BP8
        self.weight_up6 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.up8 = up_block(dim, dim, dim, scale=4)
        self.weight_down6 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.down8 = DownBlock(dim, dim, 6, 4, 1)
        self.SA8 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        # BP9
        self.weight_up7 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.up9 = up_block(dim, dim, dim, scale=4)
        self.weight_down7 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.down9 = DownBlock(dim, dim, 6, 4, 1)
        self.SA9 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        # BP10
        self.weight_up8 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.up10 = up_block(dim, dim, dim, scale=4)
        self.weight_down8 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.down10 = DownBlock(dim, dim, 6, 4, 1)
        self.SA10 = Time_attention(dim, dim, kernel_size=1, stride=1, padding=0, scale=1)
        
        # reconstruction
        self.SR_conv1 = nn.Sequential(nn.Conv2d(10 * dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.SR_conv2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1), 
                                   nn.PReLU())
        self.LR_conv1 = nn.Sequential(nn.Conv2d(9 * dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.LR_conv2 = UpBlock(dim, dim, 6, 4, 1)
        self.SR_conv3 = nn.Conv2d(dim, input_dim, 3, 1, 1)
        
        # BP final
        self.final_feat1 = nn.Sequential(nn.Conv2d(input_dim, 2 * dim, kernel_size=3, stride=1, padding=1), 
                                   nn.PReLU())
        self.final_SA0 = Space_attention(2 * dim, 2 * dim, 1, 1, 0, 1)
        self.final_feat2 = nn.Conv2d(2 * dim, input_dim, 3, 1, 1)

        self.Det_conv1 = nn.Sequential(nn.Conv2d(10 * dim, dim, kernel_size=1, stride=1, padding=0), 
                                   nn.PReLU())
        self.Det_conv2 = nn.Sequential(nn.Conv2d(dim, dim, kernel_size=3, stride=1, padding=1), 
                                   nn.PReLU())
        self.Det_conv3 = nn.Conv2d(dim, input_dim, 3, 1, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # feature extraction
        bic_x = F.interpolate(x, scale_factor=4, mode='bicubic')
        feat_x = self.feat1(x)
        SA0 = self.SA0(feat_x)
        feat_x = self.feat2(SA0)
        # BP 1
        up1, det1 = self.up1(feat_x)
        down1 = self.down1(up1)
        down1 = self.SA1(feat_x, down1)
        # BP 2
        up2, det2 = self.up2(down1)
        down2 = self.down2(up2)
        down2 = self.SA2(down1, down2)
        # BP 3
        up3, det3 = self.up3(down2)
        up3 = up3 + self.weight_up1(up1)
        down3 = self.down3(up3)
        down3 = self.SA3(self.weight_down1(down1), down3)
        # BP 4
        up4, det4 = self.up4(down3)
        up4 = up4 + self.weight_up2(up2)
        down4 = self.down4(up4)
        down4 = self.SA4(self.weight_down2(down2), down4)
        # BP 5
        up5, det5 = self.up5(down4)
        up5 = up5 + self.weight_up3(up3)
        down5 = self.down5(up5)
        down5 = self.SA5(self.weight_down3(down3), down5)
        # BP 6
        up6, det6 = self.up6(down5)
        up6 = up6 + self.weight_up4(up4)
        down6 = self.down6(up6)
        down6 = self.SA6(self.weight_down4(down4), down6)
        # BP 7
        up7, det7 = self.up7(down6)
        up7 = up7 + self.weight_up5(up5)
        down7 = self.down7(up7)
        down7 = self.SA7(self.weight_down5(down5), down7)
        # BP 8
        up8, det8 = self.up8(down7)
        up8 = up8 + self.weight_up6(up6)
        down8 = self.down8(up8)
        down8 = self.SA8(self.weight_down6(down6), down8)
        # BP 9
        up9, det9 = self.up9(down8)
        up9 = up9 + self.weight_up7(up7)
        down9 = self.down9(up9)
        down9 = self.SA9(self.weight_down7(down7), down9)
        # BP 10
        up10, det10 = self.up10(down9)
        up10 = up10 + self.weight_up8(up8)

        # reconstruction
        HR_feat = torch.cat((up1, up2, up3, up4, up5, up6, up7, up8, up9, up10), 1)
        LR_feat = torch.cat((down1, down2, down3, down4, down5, down6, down7, down8, down9), 1)
        Det_feat = torch.cat((det1, det2, det3, det4, det5, det6, det7, det8, det9, det10), 1)
        
        HR_feat = self.SR_conv1(HR_feat)
        HR_feat = self.SR_conv2(HR_feat)
        LR_feat = self.LR_conv1(LR_feat)
        LR_feat = self.LR_conv2(LR_feat)
        SR_res = self.SR_conv3(HR_feat + LR_feat)

        SR = bic_x + SR_res

        LR_res = x - F.interpolate(SR, scale_factor=0.25, mode='bicubic')
        LR_res = self.final_feat1(LR_res)
        LR_SA = self.final_SA0(LR_res)
        LR_res = self.final_feat2(LR_SA)

        SR_res = F.interpolate(LR_res, scale_factor=4, mode='bicubic')

        SR = SR + SR_res

        Det_feat = self.Det_conv1(Det_feat)
        Det_feat = self.Det_conv2(Det_feat)
        Det = self.Det_conv3(Det_feat)

        return SR, Det

#===============================================================================================================================
#Original model
#===============================================================================================================================
class ABPN_v5(nn.Module):
    def __init__(self, input_dim, dim):
        super(ABPN_v5, self).__init__()
        kernel_size = 6
        pad = 1
        stride = 4

        self.feat1 = ConvBlock(input_dim, 2 * dim, 3, 1, 1)
        self.SA0 = Space_attention(2 * dim, 2 * dim, 1, 1, 0, 1)
        self.feat2 = ConvBlock(2 * dim, dim, 3, 1, 1)
        # BP 1
        self.up1 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.down1 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA1 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP 2
        self.up2 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.down2 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA2 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP 3
        self.weight_up1 = ConvBlock(dim, dim, 1, 1, 0)
        self.up3 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.weight_down1 = ConvBlock(dim, dim, 1, 1, 0)
        self.down3 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA3 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP 4
        self.weight_up2 = ConvBlock(dim, dim, 1, 1, 0)
        self.up4 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.weight_down2 = ConvBlock(dim, dim, 1, 1, 0)
        self.down4 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA4 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP5
        self.weight_up3 = ConvBlock(dim, dim, 1, 1, 0)
        self.up5 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.weight_down3 = ConvBlock(dim, dim, 1, 1, 0)
        self.down5 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA5 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP6
        self.weight_up4 = ConvBlock(dim, dim, 1, 1, 0)
        self.up6 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.weight_down4 = ConvBlock(dim, dim, 1, 1, 0)
        self.down6 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA6 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP7
        self.weight_up5 = ConvBlock(dim, dim, 1, 1, 0)
        self.up7 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.weight_down5 = ConvBlock(dim, dim, 1, 1, 0)
        self.down7 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA7 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP8
        self.weight_up6 = ConvBlock(dim, dim, 1, 1, 0)
        self.up8 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.weight_down6 = ConvBlock(dim, dim, 1, 1, 0)
        self.down8 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA8 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP9
        self.weight_up7 = ConvBlock(dim, dim, 1, 1, 0)
        self.up9 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.weight_down7 = ConvBlock(dim, dim, 1, 1, 0)
        self.down9 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA9 = Time_attention(dim, dim, 1, 1, 0, 1)
        # BP10
        self.weight_up8 = ConvBlock(dim, dim, 1, 1, 0)
        self.up10 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.weight_down8 = ConvBlock(dim, dim, 1, 1, 0)
        self.down10 = DownBlock(dim, dim, kernel_size, stride, pad)
        self.SA10 = Time_attention(dim, dim, 1, 1, 0, 1)
        # reconstruction
        self.SR_conv1 = ConvBlock(10 * dim, dim, 1, 1, 0)
        self.SR_conv2 = ConvBlock(dim, dim, 3, 1, 1)
        self.LR_conv1 = ConvBlock(9 * dim, dim, 1, 1, 0)
        self.LR_conv2 = UpBlock(dim, dim, kernel_size, stride, pad)
        self.SR_conv3 = nn.Conv2d(dim, input_dim, 3, 1, 1)
        # BP final
        self.final_feat1 = ConvBlock(input_dim, 2 * dim, 3, 1, 1)
        self.final_SA0 = Space_attention(2 * dim, 2 * dim, 1, 1, 0, 1)
        self.final_feat2 = nn.Conv2d(2 * dim, input_dim, 3, 1, 1)

        for m in self.modules():
            classname = m.__class__.__name__
            if classname.find('Conv2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()
            elif classname.find('ConvTranspose2d') != -1:
                torch.nn.init.kaiming_normal_(m.weight)
                if m.bias is not None:
                    m.bias.data.zero_()

    def forward(self, x):
        # feature extraction
        bic_x = F.interpolate(x, scale_factor=4, mode='bicubic')
        feat_x = self.feat1(x)
        SA0 = self.SA0(feat_x)
        feat_x = self.feat2(SA0)
        # BP 1
        up1 = self.up1(feat_x)
        down1 = self.down1(up1)
        down1 = self.SA1(feat_x, down1)
        # BP 2
        up2 = self.up2(down1)
        down2 = self.down2(up2)
        down2 = self.SA2(down1, down2)
        # BP 3
        up3 = self.up3(down2) + self.weight_up1(up1)
        down3 = self.down3(up3)
        down3 = self.SA3(self.weight_down1(down1), down3)
        # BP 4
        up4 = self.up4(down3) + self.weight_up2(up2)
        down4 = self.down4(up4)
        down4 = self.SA4(self.weight_down2(down2), down4)
        # BP 5
        up5 = self.up5(down4) + self.weight_up3(up3)
        down5 = self.down5(up5)
        down5 = self.SA5(self.weight_down3(down3), down5)
        # BP 6
        up6 = self.up6(down5) + self.weight_up4(up4)
        down6 = self.down6(up6)
        down6 = self.SA6(self.weight_down4(down4), down6)
        # BP 7
        up7 = self.up7(down6) + self.weight_up5(up5)
        down7 = self.down7(up7)
        down7 = self.SA7(self.weight_down5(down5), down7)
        # BP 8
        up8 = self.up8(down7) + self.weight_up6(up6)
        down8 = self.down8(up8)
        down8 = self.SA8(self.weight_down6(down6), down8)
        # BP 9
        up9 = self.up9(down8) + self.weight_up7(up7)
        down9 = self.down9(up9)
        down9 = self.SA9(self.weight_down7(down7), down9)
        # BP 10
        up10 = self.up10(down9) + self.weight_up8(up8)
        # reconstruction
        HR_feat = torch.cat((up1, up2, up3, up4, up5, up6, up7, up8, up9, up10), 1)
        LR_feat = torch.cat((down1, down2, down3, down4, down5, down6, down7, down8, down9), 1)
        HR_feat = self.SR_conv1(HR_feat)
        HR_feat = self.SR_conv2(HR_feat)
        LR_feat = self.LR_conv1(LR_feat)
        LR_feat = self.LR_conv2(LR_feat)
        SR_res = self.SR_conv3(HR_feat + LR_feat)

        SR = bic_x + SR_res

        LR_res = x - F.interpolate(SR, scale_factor=0.25, mode='bicubic')
        LR_res = self.final_feat1(LR_res)
        LR_SA = self.final_SA0(LR_res)
        LR_res = self.final_feat2(LR_SA)

        SR_res = F.interpolate(LR_res, scale_factor=4, mode='bicubic')

        SR = SR + SR_res

        return SR
        