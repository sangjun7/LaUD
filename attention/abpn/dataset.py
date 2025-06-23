import re
from os import listdir
from PIL import Image, ImageOps
import random
import torch
import torchvision.transforms as transforms

#===============================================================================================================================
def load_image(img_path, is_resize=False, resize_h=144, resize_w=None, is_rcrop=False, crop_h=128, crop_w=None, scale=2,
               is_rrot=False, rand_hori_flip=True, rand_vert_flip=True, grayscale=False):
    """
    Args:
        img_path: Path of an image.

        is_resize (bool): Whether to resize the input image.
        resize_h: Resized height size.
        resize_w: Resized width size.
        is_rcrop (bool): Whether to randomly crop the input image.
        crop_h: Randomly cropped height size.
        crop_w: Randomly cropped width size.

        scale: Upscaling factor.

        is_rrot (bool): Whether to randomly rotate 90 degrees.
        rand_hori_flip (bool): Whether to randomly horizontal flip.
        rand_vert_flip (bool): Whether to randomly vertical flip.
        grayscale (bool): Whether to convert to the gray image.
    """

    #open img
    img = Image.open(img_path)
    if grayscale and img.mode != 'L' :
        img = img.convert('L')
    if grayscale is False and img.mode != 'RGB' :
        img = img.convert('RGB')
    
    #augmentation
    if rand_hori_flip and random.randint(0,1) == 0:
        img = ImageOps.mirror(img)
    if rand_vert_flip and random.randint(0,1) == 0:
        img = ImageOps.flip(img)
    if is_resize :
        if resize_w is None :
            resize_w = resize_h
        img = img.resize((resize_w, resize_h), Image.BICUBIC)
    if is_rcrop :
        [w, h] = img.size
        if crop_w is None :
            crop_w = crop_h
        cx1 = random.randint(0, w-crop_w)
        cx2 = cx1 + crop_w
        cy1 = random.randint(0, h-crop_h)
        cy2 = cy1 + crop_h
        
        img = img.crop((cx1, cy1, cx2, cy2))

    if is_rrot and random.randint(0,1) == 0:
        img = img.rotate(90)

    if not is_resize and not is_rcrop:
        w, h = img.size
        if w % scale != 0:
            img = img.crop((0, 0, w-(w % scale), h))
        w, h = img.size
        if h % scale != 0:
            img = img.crop((0, 0, w, h-(h % scale)))

    #making LR image
    img_lr = img.resize((int(img.size[0]/scale), int(img.size[1]/scale)), Image.BICUBIC)
        
    return img, img_lr

def rgb_shuffle(tensor_img, order=None):
    """
    Args:
        tensor_img: Pytorch tensor input image.
        order: Target RGB channel order. 
               The default is None. If None, the order is random.
    """
    
    if order is not None:
        rand_lst = order
    else:
        rand_lst = random.sample(range(3), k=len(range(3)))
        
    tensor_img_shuffle = torch.cat([tensor_img[rand_lst[0],:,:].unsqueeze(0), tensor_img[rand_lst[1],:,:].unsqueeze(0), tensor_img[rand_lst[2],:,:].unsqueeze(0)], dim=0)
    
    return tensor_img_shuffle
    
#===============================================================================================================================
class train_dataset(torch.utils.data.Dataset):
    def __init__(self, mode='train', root_path=None,
                 is_resize=True, resize_h=256, resize_w=None, is_rcrop=True, crop_h=128, crop_w=None, scale=2,
                 is_rrot=False, rand_hori_flip=True, rand_vert_flip=True, rgb_shuffle=True, grayscale=False, norm=True):
        super(train_dataset, self).__init__()
        """
        Args:
            mode: Label indicating the dataset split ('train', 'test', or 'valid' for ImageNet; 'train' or 'test' for DF2K).
            root_path: Root directory containing ImageNet images.

            rgb_shuffle (bool): Whether to randomly shuffle RGB channels.
            norm (bool): Whether to normalize images. 
                         The default is True for training our CNN models.
                         For attention-based models, this follows the original model settings.

            Others: Same as the arguments in the function 'load_image'.
        """
        
        if mode not in ['train', 'test', 'valid']:
            raise Exception("The mode must be one of 'train', 'test', or 'valid' for ImageNet; 'train' or 'test' for DF2K.")

        self.img_path = '/'.join([root_path, mode])
        self.img_lst = listdir(self.img_path)
        self.scale = scale
        
        self.is_resize = is_resize
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.is_rcrop = is_rcrop
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.is_rrot = is_rrot
        self.rand_hori_flip = rand_hori_flip
        self.rand_vert_flip = rand_vert_flip
        self.rgb_shuffle = rgb_shuffle
        self.grayscale = grayscale
        
        if norm:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
                               
    def __getitem__(self, index):
                               
        path = '/'.join([self.img_path, self.img_lst[index]])
        img, img_lr = load_image(path, is_resize=self.is_resize, resize_h=self.resize_h, resize_w=self.resize_w, 
                         is_rcrop=self.is_rcrop, crop_h=self.crop_h, crop_w=self.crop_w, scale=self.scale,
                         is_rrot=self.is_rrot, rand_hori_flip=self.rand_hori_flip, rand_vert_flip=self.rand_vert_flip, grayscale=self.grayscale)
        
        img = self.transform(img)
        img_lr = self.transform(img_lr)
        
        if self.rgb_shuffle and random.randint(0,1) == 0:
            order = random.sample(range(3), k=len(range(3)))
            img = rgb_shuffle(img, order)
            img_lr = rgb_shuffle(img_lr, order)
        
        return img, img_lr
        
    def __len__(self):
        
        return len(self.img_lst)

#===============================================================================================================================
class test_dataset(torch.utils.data.Dataset):
    def __init__(self, root_path=None, type='Set5',
                 is_resize=False, resize_h=144, resize_w=None, is_rcrop=False, crop_h=128, crop_w=None, scale=2, 
                 is_rrot=False, rand_hori_flip=False, rand_vert_flip=False, rgb_shuffle=False, grayscale=False, norm=True):
        super(test_dataset, self).__init__()
        
        self.img_path = '/'.join([root_path, type, 'image_SRF_'+str(2)])
        tot_img_lst = listdir(self.img_path)
        HR_lst = []
        LR_lst = []
        for img in tot_img_lst:
            if 'HR' in img:
                HR_lst.append(img)
            else:
                LR_lst.append(img)
        self.HR_lst = sorted(HR_lst, key = lambda s: int(re.findall(r'\d\d\d+', s)[-1]))
        self.LR_lst = sorted(LR_lst, key = lambda s: int(re.findall(r'\d\d\d+', s)[-1]))
        self.scale = scale
        
        self.is_resize = is_resize
        self.resize_h = resize_h
        self.resize_w = resize_w
        self.is_rcrop = is_rcrop
        self.crop_h = crop_h
        self.crop_w = crop_w
        self.is_rrot = is_rrot
        self.rand_hori_flip = rand_hori_flip
        self.rand_vert_flip = rand_vert_flip
        self.rgb_shuffle = rgb_shuffle
        self.grayscale = grayscale

        if norm:
            self.transform = transforms.Compose([transforms.ToTensor(),
                                                 transforms.Normalize((0.5, 0.5, 0.5),(0.5, 0.5, 0.5))])
        else:
            self.transform = transforms.Compose([transforms.ToTensor()])
                               
    def __getitem__(self, index):
                               
        hr_path = '/'.join([self.img_path, self.HR_lst[index]])
        img_hr, img_lr = load_image(hr_path, is_resize=self.is_resize, resize_h=self.resize_h, resize_w=self.resize_w, 
                                    is_rcrop=self.is_rcrop, crop_h=self.crop_h, crop_w=self.crop_w, scale=self.scale, 
                                    is_rrot=self.is_rrot, rand_hori_flip=self.rand_hori_flip, rand_vert_flip=self.rand_vert_flip, grayscale=self.grayscale)

        img_hr = self.transform(img_hr)
        img_lr = self.transform(img_lr)
        
        return img_hr, img_lr
        
    def __len__(self):
        
        return len(self.HR_lst)
        
