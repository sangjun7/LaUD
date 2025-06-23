import cv2
import numpy as np
import torch
import torch.nn.functional as F
        
#===============================================================================================================================
def lst_loss(sr_lst, hr_gt, sr_weights, det_lst=None, det_gt=None, det_weights=None, loss_fct_sr=torch.nn.L1Loss(), loss_fct_det=torch.nn.L1Loss()):
    """
    Args :
        ----------------------------------------------------
        sr_lst: List of SR images produced by a model. 
                If k-step RUDP is applied to the model, the elements in this list correspond to the 1st, 2nd, ..., kth SR images.
        
        hr_gt: Ground-truth HR image.
        
        sr_weights: List of weights for the weighted sum of losses between SR and HR images. 
                    The length of the list must be the same as the length of 'sr_lst'.

        loss_fct_sr: Loss function for SR and HR images. (Default: L1 loss)
    
        ----------------------------------------------------
        det_lst: List of detail images produced by a model (if those are produced). 
                If k-step RUDP is applied to the model, the elements in this list correspond to the 1st, 2nd, ..., kth detail images.
        
        det_gt: Ground-truth detail image.
        
        det_weights: List of weights for the weighted sum of losses between the produced detail and ground-truth detail images. 
                    The length of the list must be the same as the length of 'det_lst'.

        loss_fct_det: Loss function for the produced detail and ground-truth detail images. (Default: L1 loss)
    """

    if len(sr_lst) != len(sr_weights):
        raise Exception("ERROR: You must match lengths between sr_lst and sr_weights")

    loss_sr = 0
    for i in range(len(sr_lst)):
        loss_sr += sr_weights[i] * loss_fct_sr(sr_lst[i], hr_gt)

    if det_lst is not None:
        if len(det_lst) != len(det_weights):
            raise Exception("ERROR: You must match lengths between det_lst and det_weights")

        loss_det = 0
        for i in range(len(det_lst)):
            loss_det += det_weights[i] * loss_fct_det(det_lst[i], det_gt)
    
    return loss_sr, loss_det
    
#===============================================================================================================================
def togray(img, ver='YCrCb_BT601', pt=False):
    """
    Args:
        img: Image represented as a PyTorch tensor or NumPy array, to be converted to the Y channel.
        ver: Coefficient version used to convert an RGB image to the Y channel(the gray image). 
             We think 'YCrCb_BT601' is the general method in digital, so we use this in all our experiments.
        pt (bool): Whether the input image is represented as a PyTorch tensor or a NumPy array.
                   If True, the input image is a PyTorch tensor; if False, a NumPy array.
        
    ----------------------------------------------------
    If the input image is a Pytorch tensor, then the parameter 'pt' must be True.
    
    When the 'pt' is True, the type of input image must be torch.float32 in the range [0,1].
    And the image must have the dimensions ordered as (B, C, H, W) with B=1 (For arbitrary B value, we will add later).
    This function changes the dtype to torch.float32 for 8 bits per sample,
    then returns the Y channel from the RGB channel.
    
    Return: A float32 tensor in the range [0,255], with dimensions ordered as (B, C, H, W)
    
    ----------------------------------------------------
    If the input image is a Numpy array, then the parameter 'pt' must be False.
    
    When the 'pt' is False, the type of input image must be np.float32 in the range [0,1].
    And the image must have the dimensions ordered as (H, W, C).
    This function changes the dtype to np.float32 for 8 bits per sample,
    then returns the Y channel from the RGB channel.
    
    Return: A float32 ndarray in the range [0,255].
    
    """
    if not pt:
        img = img.astype(np.float32)           
        if ver == 'YCrCb_BT601':
            coeff = np.array([65.481, 128.553, 24.966]).reshape(1, 1, 3)
            img = img * coeff
            img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2] + 16
            img /= 255.0
        elif ver == 'YCrCb_BT601_single_bitshift':
            coeff = np.array([65.738, 129.057, 25.064]).reshape(1, 1, 3) 
            img = img * coeff
            img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2] + 16
            img /= 255.0
        elif ver == 'YPrPb_BT601':  
            coeff = np.array([0.299, 0.587, 0.114]).reshape(1, 1, 3)
            img = img * coeff
            img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        elif ver == 'YCrCb_BT709':
            coeff = np.array([0.2126, 0.7152, 0.0722]).reshape(1, 1, 3)
            img = img * coeff
            img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        elif ver == 'YCrCb_BT2020':
            coeff = np.array([0.2627, 0.678, 0.0593]).reshape(1, 1, 3)
            img = img * coeff
            img = img[:, :, 0] + img[:, :, 1] + img[:, :, 2]
        img = img * 255.0
        img = img.astype(np.float32)
    else:
        img = img.to(torch.float32)
        if ver == 'YCrCb_BT601':
            coeff = torch.tensor([65.481, 128.553, 24.966]).reshape(1, 3, 1, 1).to(img.device)
            img = img * coeff
            img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :] + 16
            img /= 255.0
        elif ver == 'YCrCb_BT601_single_bitshift': #CAR
            coeff = torch.tensor([65.738, 129.057, 25.064]).reshape(1, 3, 1, 1).to(img.device)
            img = img * coeff
            img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :] + 16
            img /= 255.0
        elif ver == 'YPrPb_BT601':  
            coeff = torch.tensor([0.299, 0.587, 0.114]).reshape(1, 3, 1, 1).to(img.device)
            img = img * coeff
            img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]
        elif ver == 'YCrCb_BT709':
            coeff = torch.tensor([0.2126, 0.7152, 0.0722]).reshape(1, 3, 1, 1).to(img.device)
            img = img * coeff
            img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]
        elif ver == 'YCrCb_BT2020':
            coeff = torch.tensor([0.2627, 0.678, 0.0593]).reshape(1, 3, 1, 1).to(img.device)
            img = img * coeff
            img = img[:, 0, :, :] + img[:, 1, :, :] + img[:, 2, :, :]
        img = img * 255.0
        img = img.unsqueeze(1)
    
    return img

#===============================================================================================================================
def cal_psnr(img, pred, crop_border=0, minmax='-1_1', clamp=True, gray_scale=True, ver='YCrCb_BT601'):
    """
    Args:
        img: Ground-truth HR image with the shape of BCHW (generally, B=1) and the type of torch.float32.
        pred: Generated SR image with the shape of BCHW (generally, B=1) and the type of torch.float32.
        crop_border: The number of pixels to crop from the image border. It is usually the same as the upscaling factor.

        minmax: Min and Max boundaries of input images(PSNR is calculated by setting all images to these boundaries).
                It can choose among '-1_1', '0_1', and '0_255'. In our experiments, we usually use '-1_1' except for applying to attention-based models.
        clamp (bool): Whether to clamp the generated image to the minmax boundaries.
        
        gray_scale (bool): Whether to convert to the gray image. Basically, we assign this 'True' in all our experiments.
        ver: Coefficient version used to convert an RGB image to the Y channel(the gray image).
            We think 'YCrCb_BT601' is the general method in digital, so we use this in all our experiments.
        
    ----------------------------------------------------
    This function takes img(target) and pred(prediction) with a range of 'minmax' as input.
    Inputs have the shape of BCHW (generally, B=1) and the type of torch.float32.
    
    After taking images, we fit the range of images to [0,1](if minmax is '-1_1') and device to CPU.
    Then change the Numpy array from the Pytorch tensor, and change it to dimension of HWC.
    
    If want, clamping values, cropping border, and converting to Y channel are implemented.
    
    ToDo: For just in case, modify code to consider batch size > 1
    """

    if minmax == '0_255':
        img = img.cpu().numpy().transpose(0, 2, 3, 1)    # change device and to numpy and to order of BHWC
        img = img[0, ...]                                # squeeze to HWC by choosing 1 batch of index 0
            
        if clamp:
            pred = torch.clamp(pred,0,255.)   
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
        pred = pred[0, ...]

        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            pred = pred[crop_border:-crop_border, crop_border:-crop_border, ...]
        
        if gray_scale:
            img = togray(img/255.0, ver=ver, pt=False)
            pred = togray(pred/255.0, ver=ver, pt=False)
        else:
            img = img.astype(np.float32)
            pred = pred.astype(np.float32)
    else:
        if minmax == '-1_1':
            img = (img + 1.)/2.                                # change range to 0~1 from -1~1
            pred = (pred + 1.)/2.

        img = img.cpu().numpy().transpose(0, 2, 3, 1)    # change device and to numpy and to order of BHWC
        img = img[0, ...]                                # squeeze to HWC by choosing 1 batch of index 0
            
        if clamp:
            pred = torch.clamp(pred,0,1)   
        pred = pred.cpu().numpy().transpose(0, 2, 3, 1)
        pred = pred[0, ...]
        
        if crop_border != 0:
            img = img[crop_border:-crop_border, crop_border:-crop_border, ...]
            pred = pred[crop_border:-crop_border, crop_border:-crop_border, ...]
        
        if gray_scale:
            img = togray(img, ver=ver, pt=False)
            pred = togray(pred, ver=ver, pt=False)
        else:
            img = img * 255.0
            img = img.astype(np.float32)
            pred = pred * 255.0
            pred = pred.astype(np.float32)
        
    mse = np.mean((img - pred) ** 2)
    if mse == 0:
        psnr = float('inf')
    else:
        psnr = 10. * np.log10(255. * 255. / mse)

    return psnr

def cal_ssim(img, pred, crop_border=0, minmax='-1_1', filter_size=11, filter_sigma=1.5, clamp=True, grayscale=True, ver='YCrCb_BT601'):
    """
    Args:
        img: Ground-truth HR image with the shape of BCHW (generally, B=1) and the type of torch.float32.
        pred: Generated SR image with the shape of BCHW (generally, B=1) and the type of torch.float32.
        crop_border: The number of pixels to crop from the image border. It is usually the same as the upscaling factor.
        minmax: Min and Max boundaries of input images(SSIM is calculated by setting all images to these boundaries).
                It can choose among '-1_1', '0_1', and '0_255'. In our experiments, we usually use '-1_1' except for applying to attention-based models.
        
        filter_size: Filter size of the Gaussian kernel.
        filter_sigma: Standard deviation of the Gaussian kernel.
                
        clamp (bool): Whether to clamp the generated image to the minmax boundaries.
        gray_scale (bool): Whether to convert to the gray image. Basically, we assign this 'True' in all our experiments.
        ver: Coefficient version used to convert an RGB image to the Y channel(the gray image).
            We think 'YCrCb_BT601' is the general method in digital, so we use this in all our experiments.
        
    ----------------------------------------------------
    This function takes img(target) and pred(prediction) with a range of 'minmax' as input.
    Inputs have the shape of BCHW (generally, B=1) and the type of torch.float32.
    
    After taking images, we convert to the Y channel with the range of [0,255], then change to torch.float64.
    
    If want, clamping values and cropping border are implemented.
    """

    if minmax == '0_255':
        if clamp:
            pred = torch.clamp(pred,0,255.)   
        
        if crop_border != 0:
            img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            pred = pred[:, :, crop_border:-crop_border, crop_border:-crop_border]
        
        if grayscale:
            img = togray(img/255.0, ver=ver, pt=True)
            pred = togray(pred/255.0, ver=ver, pt=True)   # get Y channel float32 tensor with range [0,255] and order BCHW (basically, B=1)
    else:
        if minmax == '-1_1':
            img = (img + 1)/2                             # change range to 0~1 from -1~1
            pred = (pred + 1)/2
            
        if clamp:
            pred = torch.clamp(pred,0,1.)   
    
        if crop_border != 0:
            img = img[:, :, crop_border:-crop_border, crop_border:-crop_border]
            pred = pred[:, :, crop_border:-crop_border, crop_border:-crop_border]
    
        if grayscale:
            img = togray(img, ver=ver, pt=True)
            pred = togray(pred, ver=ver, pt=True)         # get Y channel float32 tensor with range [0,255] and order BCHW (basically, B=1)
        else:
            img = img * 255.0
            pred = pred * 255.0
        
    img = img.to(torch.float64)
    pred = pred.to(torch.float64)
        
    b, ch, h, w = img.size()
    k1 = 0.01
    k2 = 0.03
    if grayscale:
        c1 = (k1 * 255.) ** 2
        c2 = (k2 * 255.) ** 2
    else:
        c1 = (k1 * 2553.) ** 2
        c2 = (k2 * 2553.) ** 2
    
    gauss = cv2.getGaussianKernel(filter_size, filter_sigma)
    window = np.outer(gauss, gauss.transpose())
    window = torch.from_numpy(window).reshape(1, 1, filter_size, filter_size)
    window = window.expand(ch, 1, filter_size, filter_size).to(img.dtype).to(img.device)
    
    mu1 = F.conv2d(img, window, stride=1, padding=0, groups=ch)
    mu2 = F.conv2d(pred, window, stride=1, padding=0, groups=ch)
    mu1_sq = mu1 ** 2
    mu2_sq = mu2 ** 2 
    mu12 = mu1 * mu2
    sig1_sq = F.conv2d(img * img, window, stride=1, padding=0, groups=ch) - mu1_sq
    sig2_sq = F.conv2d(pred * pred, window, stride=1, padding=0, groups=ch) - mu2_sq
    sig12 = F.conv2d(img * pred, window, stride=1, padding=0, groups=ch) - mu12
    
    l_comp = (2*mu12 + c1)/(mu1_sq + mu2_sq + c1)
    cs_comp = (2*sig12 + c2)/(sig1_sq + sig2_sq + c2)
    ssim = l_comp * cs_comp
    ssim_mean = ssim.mean([1,2,3])                # Evaluation mean without batch. So it has an order of B (basically, B=1)
    
    return ssim_mean.item()

#===============================================================================================================================
def geo_trans(inp, inv=False, shape='lst'):        
    """
    Args:
        inp: Input for 8 geometric transformations 
        inv (bool): Whether to apply the geometric transformations or the inverse of the geometric transformations.
        shape: Choice for the shape of the output. It can be 'lst' or 'tensor'.

    ----------------------------------------------------
    If inv=False, this returns the ensemble image that is a concatenation of transformed images.
    The shape of the ensemble image is BCHW (generally, B=1).

    If inv=True, this returns the mean of inverse-transformed images(i.e., returned to the original images).
    The shape of the input image is BCHW (generally, B=1).
    
    The order of imgs in the list is original, hori flip, vert flip, hori&vert flip,
                                    rot90, rot90 and hori flip, rot90 and vert flip, rot90 and hori&vert flip.
    """
    if shape == 'lst':                        # output of inv=False and input of inv=True are the list type
        if inv == False:
            timg_lst = []

            timg_lst.append(inp)                             # original (BCHW, B=1)
            timg_lst.append(torch.flip(inp, dims=(3,)))      # horizontal flip
            timg_lst.append(torch.flip(inp, dims=(2,)))      # vertical flip
            timg_lst.append(torch.flip(inp, dims=(2,3)))     # horizontal & vertical flip

            rot_inp = torch.rot90(inp, k=1, dims=(2,3))
            timg_lst.append(rot_inp)                         # 90 rotation
            timg_lst.append(torch.flip(rot_inp, dims=(3,)))  # 90 rotation, horizontal flip
            timg_lst.append(torch.flip(rot_inp, dims=(2,)))  # 90 rotation, vertical flip
            timg_lst.append(torch.flip(rot_inp, dims=(2,3))) # 90 rotation, horizontal & vertical flip

            return timg_lst

        else:
            img_sum = 0
            img_sum += inp[0]                          # original (BCHW, B=1)
            img_sum += torch.flip(inp[1], dims=(3,))   # inverse horizontal flip
            img_sum += torch.flip(inp[2], dims=(2,))   # inverse vertical flip
            img_sum += torch.flip(inp[3], dims=(2,3))  # inverse horizontal & vertical flip
            
            img_sum += torch.rot90(inp[4], k=-1, dims=(2,3))                          # inverse 90 rotation
            img_sum += torch.rot90(torch.flip(inp[5], dims=(3,)), k=-1, dims=(2,3))   # inverse horizontal flip, inverse 90 rotation
            img_sum += torch.rot90(torch.flip(inp[6], dims=(2,)), k=-1, dims=(2,3))   # inverse vertical flip, inverse 90 rotation
            img_sum += torch.rot90(torch.flip(inp[7], dims=(2,3)), k=-1, dims=(2,3))  # inverse horizontal & vertical flip, inverse 90 rotation
            
            ensembled_img = img_sum / 8.

            return ensembled_img
        
    elif shape == 'tensor':                   # all input and output are the tensor type
        if inv == False:
            timg_lst = []

            timg_lst.append(inp)                             # original (BCHW, B=1)
            timg_lst.append(torch.flip(inp, dims=(3,)))      # horizontal flip
            timg_lst.append(torch.flip(inp, dims=(2,)))      # vertical flip
            timg_lst.append(torch.flip(inp, dims=(2,3)))     # horizontal & vertical flip

            rot_inp = torch.rot90(inp, k=1, dims=(2,3))
            timg_lst.append(rot_inp)                         # 90 rotation
            timg_lst.append(torch.flip(rot_inp, dims=(3,)))  # 90 rotation, horizontal flip
            timg_lst.append(torch.flip(rot_inp, dims=(2,)))  # 90 rotation, vertical flip
            timg_lst.append(torch.flip(rot_inp, dims=(2,3))) # 90 rotation, horizontal & vertical flip

            return torch.cat(timg_lst, dim=0)

        else:
            img_sum = 0
            if 'cuda' in inp.device.type:
                inp = inp.cpu()                             # shape : BCHW ( B=8 ). If we take each image along a batch, it has the shape of CHW.

            img_sum += inp[0, ...]                          # original
            img_sum += torch.flip(inp[1, ...], dims=(2,))   # inverse horizontal flip
            img_sum += torch.flip(inp[2, ...], dims=(1,))   # inverse vertical flip
            img_sum += torch.flip(inp[3, ...], dims=(1,2))  # inverse horizontal & vertical flip

            img_sum += torch.rot90(inp[4, ...], k=-1, dims=(1,2))                          # inverse 90 rotation
            img_sum += torch.rot90(torch.flip(inp[5, ...], dims=(2,)), k=-1, dims=(1,2))   # inverse horizontal flip, inverse 90 rotation
            img_sum += torch.rot90(torch.flip(inp[6, ...], dims=(1,)), k=-1, dims=(1,2))   # inverse vertical flip, inverse 90 rotation
            img_sum += torch.rot90(torch.flip(inp[7, ...], dims=(1,2)), k=-1, dims=(1,2))  # inverse horizontal & vertical flip, inverse 90 rotation

            ensembled_img = img_sum / 8.

            return ensembled_img.unsqueeze(0)
    else:
        raise Exception("You need to choose the shape between 'lst' and 'tensor'")

#===============================================================================================================================
def gauss_kernel(channels=3, device=torch.device('cpu')):
    kernel = torch.tensor([[1., 4., 6., 4., 1],
                           [4., 16., 24., 16., 4.],
                           [6., 24., 36., 24., 6.],
                           [4., 16., 24., 16., 4.],
                           [1., 4., 6., 4., 1.]])
    kernel /= 256.
    kernel = kernel.repeat(channels, 1, 1, 1)
    kernel = kernel.to(device)
    return kernel

def gauss(img, kernel):
    h_pad_sz = kernel.size(2) // 2
    w_pad_sz = kernel.size(3) // 2
    img = torch.nn.functional.pad(img, (w_pad_sz, w_pad_sz, h_pad_sz, h_pad_sz), mode='reflect')
    out = torch.nn.functional.conv2d(img, kernel, groups=img.shape[1])
    return out
    
def gauss_up(x):
    cc = torch.cat([x, torch.zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3], device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[2]*2, x.shape[3])
    cc = cc.permute(0,1,3,2)
    cc = torch.cat([cc, torch.zeros(x.shape[0], x.shape[1], x.shape[3], x.shape[2]*2, device=x.device)], dim=3)
    cc = cc.view(x.shape[0], x.shape[1], x.shape[3]*2, x.shape[2]*2)
    x_up = cc.permute(0,1,3,2)
    return gauss(x_up, 4*gauss_kernel(channels=x.shape[1], device=x.device))
    
def get_LP_detail(torch_img, out_other=False):
    """
    Args:
        torch_img: Tensor image, with dimensions of (B, C, H, W), to be decomposed into approximation and detail images using the Laplacian pyramid algorithm.
        out_other (bool): If True, additionally returns an approximation image along with its re-upscaled version.
    """
    gauss_down = gauss(torch_img, gauss_kernel(channels=torch_img.shape[1], device=torch_img.device))[:, :, ::2, ::2]
    re_up = gauss_up(gauss_down)
    det = torch_img - re_up

    if out_other:
        return gauss_down, re_up, det
    else:
        return det
    