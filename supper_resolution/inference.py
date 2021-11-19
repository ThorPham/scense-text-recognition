
from .model import tsrn
import torch
import sys
import time
import os
from time import gmtime, strftime
from datetime import datetime
from tqdm import tqdm
import math
import copy
from IPython import embed
from torchvision import transforms
import torchvision
from torch.autograd import Variable
import torch.nn as nn
import torch.nn.functional as F
# from thop import profile
from PIL import Image
import numpy as np
import cv2
import torchvision
# from tensor2numpy import tensor2image
from typing import Union, Optional, List, Tuple, Text, BinaryIO
import io
import pathlib
import torch
import math
irange = range


def make_grid(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
) -> torch.Tensor:
    """Make a grid of images.

    Args:
        tensor (Tensor or list): 4D mini-batch Tensor of shape (B x C x H x W)
            or a list of images all of the same size.
        nrow (int, optional): Number of images displayed in each row of the grid.
            The final grid size is ``(B / nrow, nrow)``. Default: ``8``.
        padding (int, optional): amount of padding. Default: ``2``.
        normalize (bool, optional): If True, shift the image to the range (0, 1),
            by the min and max values specified by :attr:`range`. Default: ``False``.
        range (tuple, optional): tuple (min, max) where min and max are numbers,
            then these numbers are used to normalize the image. By default, min and max
            are computed from the tensor.
        scale_each (bool, optional): If ``True``, scale each image in the batch of
            images separately rather than the (min, max) over all images. Default: ``False``.
        pad_value (float, optional): Value for the padded pixels. Default: ``0``.

    Example:
        See this notebook `here <https://gist.github.com/anonymous/bf16430f7750c023141c562f3e9f2a91>`_

    """
    if not (torch.is_tensor(tensor) or
            (isinstance(tensor, list) and all(torch.is_tensor(t) for t in tensor))):
        raise TypeError('tensor or list of tensors expected, got {}'.format(type(tensor)))

    # if list of tensors, convert to a 4D mini-batch Tensor
    if isinstance(tensor, list):
        tensor = torch.stack(tensor, dim=0)

    if tensor.dim() == 2:  # single image H x W
        tensor = tensor.unsqueeze(0)
    if tensor.dim() == 3:  # single image
        if tensor.size(0) == 1:  # if single-channel, convert to 3-channel
            tensor = torch.cat((tensor, tensor, tensor), 0)
        tensor = tensor.unsqueeze(0)

    if tensor.dim() == 4 and tensor.size(1) == 1:  # single-channel images
        tensor = torch.cat((tensor, tensor, tensor), 1)

    if normalize is True:
        tensor = tensor.clone()  # avoid modifying tensor in-place
        if range is not None:
            assert isinstance(range, tuple), \
                "range has to be a tuple (min, max) if specified. min and max are numbers"

        def norm_ip(img, min, max):
            img.clamp_(min=min, max=max)
            img.add_(-min).div_(max - min + 1e-5)

        def norm_range(t, range):
            if range is not None:
                norm_ip(t, range[0], range[1])
            else:
                norm_ip(t, float(t.min()), float(t.max()))

        if scale_each is True:
            for t in tensor:  # loop over mini-batch dimension
                norm_range(t, range)
        else:
            norm_range(tensor, range)

    if tensor.size(0) == 1:
        return tensor.squeeze(0)

    # make the mini-batch of images into a grid
    nmaps = tensor.size(0)
    xmaps = min(nrow, nmaps)
    ymaps = int(math.ceil(float(nmaps) / xmaps))
    height, width = int(tensor.size(2) + padding), int(tensor.size(3) + padding)
    num_channels = tensor.size(1)
    grid = tensor.new_full((num_channels, height * ymaps + padding, width * xmaps + padding), pad_value)
    k = 0
    for y in irange(ymaps):
        for x in irange(xmaps):
            if k >= nmaps:
                break
            # Tensor.copy_() is a valid method but seems to be missing from the stubs
            # https://pytorch.org/docs/stable/tensors.html#torch.Tensor.copy_
            grid.narrow(1, y * height + padding, height - padding).narrow(  # type: ignore[attr-defined]
                2, x * width + padding, width - padding
            ).copy_(tensor[k])
            k = k + 1
    return grid


def tensor2image(
    tensor: Union[torch.Tensor, List[torch.Tensor]],
    nrow: int = 8,
    padding: int = 2,
    normalize: bool = False,
    range: Optional[Tuple[int, int]] = None,
    scale_each: bool = False,
    pad_value: int = 0,
    format: Optional[str] = None,
) -> None:
   """Save a given Tensor into an image file.

   Args:
      tensor (Tensor or list): Image to be saved. If given a mini-batch tensor,
         saves the tensor as a grid of images by calling ``make_grid``.
      fp (string or file object): A filename or a file object
      format(Optional):  If omitted, the format to use is determined from the filename extension.
         If a file object was used instead of a filename, this parameter should always be used.
      **kwargs: Other arguments are documented in ``make_grid``.
   """
   from PIL import Image
   grid = make_grid(tensor, nrow=nrow, padding=padding, pad_value=pad_value,
                  normalize=normalize, range=range, scale_each=scale_each)
   # Add 0.5 after unnormalizing to [0, 255] to round to nearest integer
   ndarr = grid.mul(255).add_(0.5).clamp_(0, 255).permute(1, 2, 0).to('cpu', torch.uint8).numpy()
   return ndarr



device = 'cuda'
def transform_(image):
   # img = Image.open(image)
   # img = img.resize((100, 32), Image.BICUBIC)
   img = Image.fromarray(image)
   img_tensor = transforms.ToTensor()(img)

   mask = img.convert('L')
   thres = np.array(mask).mean()
   mask = mask.point(lambda x: 0 if x > thres else 255)
   mask = transforms.ToTensor()(mask)
   img_tensor = torch.cat((img_tensor, mask), 0)
   img_tensor = img_tensor.unsqueeze(0)
   return img_tensor
def generator_init(weight):
     
      model = tsrn.TSRN(scale_factor=2, width=128, height=32,
                                       STN=True, mask=True, srb_nums=5, hidden_units=32)
      model.load_state_dict(torch.load(weight)['state_dict_G'])
      return  model.to(device).eval()
   
def predict(image,model):
   # model = generator_init(weight)
   images_lr = transform_(image)
   images_lr = images_lr.to(device)

   images_sr = model(images_lr)
   tensor_out = images_sr[0][:3,:,:]
   # torchvision.utils.save_image(tensor_out, os.path.join('./out', 'out' +'.jpg'), padding=0)
   return tensor2image(tensor_out)
if __name__ == '__main__':
   output = predict(image_path='my_data_test/word_43.jpg',weight='model_best_0.pth')