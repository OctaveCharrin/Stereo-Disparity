import numpy as np
import random
from torchvision.transforms import functional as TF
from PIL import Image


class Compose(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, input):
        for t in self.transforms:
            input = t(input)
        return input


class RandomHorizontalFlip(object):
    """
    Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __HorizontalFlip__(self, input):
        return np.copy(np.fliplr(input))

    def __call__(self, input):
        if random.random() < 0.5:
            input = self.__HorizontalFlip__(input)
        return input


class RandomColorJitter(object):
    """
    Randomly horizontally flips the given PIL.Image with a probability of 0.5
    """
    def __init__(self, brightness, contrast, saturation, hue, gamma):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
        self.gamma = gamma

        self.brightness_factor = 0.
        self.contrast_factor = 0.
        self.saturation_factor = 0.
        self.hue_factor = 0.
        self.gamma_factor = 0.

    def __ColorJitter__(self, input):

        if input.shape[2] < 3:
            return input

        img = Image.fromarray(np.uint8(input * 255.),'RGB')

        if not isinstance(img, Image.Image):
            raise TypeError('img should be PIL Image. Got {}'.format(type(img)))

        l = [1,2,3,4,5]

        random.shuffle(l)

        for e in l:

            # if e == 1 and self.brightness is not None and random.random() < 0.5:
            if e == 1 and self.brightness is not None:
                img = TF.adjust_brightness(img, self.brightness_factor)
            
            # if e == 2 and self.contrast is not None and random.random() < 0.5:
            if e == 2 and self.contrast is not None:
                img = TF.adjust_contrast(img, self.contrast_factor)

            # if e == 3 and self.saturation is not None and random.random() < 0.5:
            if e == 3 and self.saturation is not None:
                img = TF.adjust_saturation(img, self.saturation_factor)
            
            # if e == 4 and self.hue is not None and random.random() < 0.5:
            if e == 4 and self.hue is not None:
                img = TF.adjust_hue(img, self.hue_factor)
            
            # if e == 4 and self.hue is not None and random.random() < 0.5:
            if e == 5 and self.gamma is not None:
                img = TF.adjust_gamma(img, self.gamma_factor)                
        
        return np.array(img).astype(np.float32) / 255.

    def __call__(self, input):

        self.brightness_factor = random.uniform(1. - self.brightness, 1. + self.brightness)
        self.contrast_factor = random.uniform(1. - self.contrast, 1. + self.contrast)
        self.saturation_factor = random.uniform(1. - self.saturation, 1. + self.saturation)
        self.hue_factor = random.uniform(-self.hue, self.hue)
        self.gamma_factor = random.uniform(1. - self.gamma, 1. + self.gamma)

        return self.__ColorJitter__(input)