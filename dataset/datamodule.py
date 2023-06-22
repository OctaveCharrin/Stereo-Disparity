import torch.utils.data as data
import numpy as np
import cv2
from PIL import Image

class KittiDataset(data.Dataset):

    def __init__(self,
                 stereo_path,
                 gtdisp_path,
                 input_transforms=None,
                 co_transforms=None,
                 size=(512,256)
                ):
        self.size = size
        with open(stereo_path,'r') as f:
            l = f.readlines()
            stereo_filelist = [l[i].rstrip().split(" ") for i in range(len(l))]
        self.stereo_filelist = stereo_filelist

        with open(gtdisp_path,'r') as f:
            gt_filelist = [s.rstrip() for s in f.readlines()]
        self.gt_filelist = gt_filelist
        
        assert(len(gt_filelist)==len(stereo_filelist))

        # data augmentation
        self.input_transforms = input_transforms
        self.co_transforms = co_transforms

    def __getitem__(self, index):
        left_path = self.stereo_filelist[index][0]
        right_path = self.stereo_filelist[index][1]
        gt_path = self.gt_filelist[index]

        left_im, right_im = rgb_image_loader(left_path, self.size), rgb_image_loader(right_path, self.size)
        target = gt_image_loader(gt_path, self.size)

        # apply transforms for inputs
        if self.input_transforms is not None:
            left_im = self.input_transforms(left_im)
            right_im = self.input_transforms(right_im)
        
        # apply co transforms
        if self.co_transforms is not None:
            left_im = self.co_transforms(left_im)
            right_im = self.co_transforms(right_im)
            target = self.co_transforms(target)

        imgs = (left_im, right_im)
        return imgs, target

    def __len__(self):
        """Length."""
        return len(self.stereo_filelist)
    
def rgb_image_loader(filename, size):
    # image = cv2.imread(filename, cv2.IMREAD_COLOR).astype(np.float16) # BGR image loaded as a numpy array
    # image = cv2.resize(image, dsize=size, interpolation=cv2.INTER_LINEAR)
    # image/=255.
    # return image
    im = np.array(Image.open(filename)).astype(np.float32)

    # Resize
    im = cv2.resize(im, size, interpolation=cv2.INTER_LINEAR).astype(np.float32)

    # Normalize between 0 - +1
    im /= 255.

    return im

def gt_image_loader(filename,size):
    image = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.
    h,w = image.shape

    # Resize and adapt the disparity accordingly
    image = (512./w) * cv2.resize(image, size, interpolation=cv2.INTER_NEAREST).astype(np.float16)
    # Set the disparity to 0 for pixels that are super close
    image[image < 1.] = 0.

    # TODO : see if disparity normalisation is needed
    # Add one dimension as channel
    image = image.reshape(image.shape+(1,))
    return image