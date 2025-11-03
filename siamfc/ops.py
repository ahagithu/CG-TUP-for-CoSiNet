from __future__ import absolute_import, division

import torch.nn as nn
import cv2
import numpy as np


def init_weights(model, gain=1):
    for m in model.modules():
        if isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.BatchNorm2d):
            nn.init.constant_(m.weight, 1)
            nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight, gain)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)


def read_image(img_file, cvt_code=cv2.COLOR_BGR2RGB):
    img = cv2.imread(img_file, cv2.IMREAD_COLOR)
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)
    return img


def show_image(img, boxes=None, box_fmt='ltwh', colors=None,
               thickness=3, fig_n=1, delay=1, visualize=True,
               cvt_code=cv2.COLOR_RGB2BGR):
    if cvt_code is not None:
        img = cv2.cvtColor(img, cvt_code)

    # resize img if necessary
    max_size = 960
    if max(img.shape[:2]) > max_size:
        scale = max_size / max(img.shape[:2])
        out_size = (
            int(img.shape[1] * scale),
            int(img.shape[0] * scale))
        img = cv2.resize(img, out_size)
        if boxes is not None:
            boxes = np.array(boxes, dtype=np.float32) * scale

    if boxes is not None:
        assert box_fmt in ['ltwh', 'ltrb']
        boxes = np.array(boxes, dtype=np.int32)
        if boxes.ndim == 1:
            boxes = np.expand_dims(boxes, axis=0)
        if box_fmt == 'ltrb':
            boxes[:, 2:] -= boxes[:, :2]

        # clip bounding boxes
        bound = np.array(img.shape[1::-1])[None, :]
        boxes[:, :2] = np.clip(boxes[:, :2], 0, bound)
        boxes[:, 2:] = np.clip(boxes[:, 2:], 0, bound - boxes[:, :2])

        if colors is None:
            colors = [
                (0, 0, 255),
                (0, 255, 0),
                (255, 0, 0),
                (0, 255, 255),
                (255, 0, 255),
                (255, 255, 0),
                (0, 0, 128),
                (0, 128, 0),
                (128, 0, 0),
                (0, 128, 128),
                (128, 0, 128),
                (128, 128, 0)]
        colors = np.array(colors, dtype=np.int32)
        if colors.ndim == 1:
            colors = np.expand_dims(colors, axis=0)

        for i, box in enumerate(boxes):
            color = colors[i % len(colors)]
            pt1 = (box[0], box[1])
            pt2 = (box[0] + box[2], box[1] + box[3])
            img = cv2.rectangle(img, pt1, pt2, color.tolist(), thickness)

    if visualize:
        winname = 'window_{}'.format(fig_n)
        cv2.imshow(winname, img)
        cv2.waitKey(delay)

    return img

#crop一块以object为中心的，变长为size大小的patch，然后将其resize成out_size的大小；
#传入size和center计算出角落坐标形式的正方形patch，即（ymin，xmin，ymax，xmax）；
#因为这样扩大的坐标有可能会超出原来的图片，所以就要计算左上角和右下角相对原图片超出多少，好进行pad，
#然后根据他们超出当中的最大值npad来在原图像周围pad，因为原图像增大了，所以Corner相对坐标也变了了。

def crop_and_resize(img, center, size, out_size,
                    border_type=cv2.BORDER_CONSTANT,
                    border_value=(0, 0, 0),
                    interp=cv2.INTER_LINEAR):
    # Input validation
    if img is None or img.size == 0:
        # Return a blank patch if input image is invalid
        return np.ones((out_size, out_size, 3), dtype=np.uint8) * np.array(border_value, dtype=np.uint8)

    # Convert size to integer
    size = round(size)

    # Validate size
    if size <= 0:
        size = 1  # Minimum valid size

    # Calculate corners
    corners = np.concatenate((
        np.round(center - (size - 1) / 2),
        np.round(center - (size - 1) / 2) + size))
    corners = np.round(corners).astype(int)

    # Get original image dimensions
    img_h, img_w = img.shape[:2]

    # Check if crop region is completely outside image bounds
    if (corners[0] >= img_h or corners[2] <= 0 or
            corners[1] >= img_w or corners[3] <= 0):
        # Return a default patch if completely outside
        return np.ones((out_size, out_size, 3), dtype=np.uint8) * np.array(border_value, dtype=np.uint8)

    # Calculate padding needed
    pads = np.concatenate((
        -corners[:2], corners[2:] - [img_h, img_w]))
    npad = max(0, int(pads.max()))

    # Apply padding if necessary
    if npad > 0:
        img = cv2.copyMakeBorder(
            img, npad, npad, npad, npad,
            border_type, value=border_value)

    # Update corners with padding
    corners = (corners + npad).astype(int)

    # Ensure corners are within bounds after padding
    corners[0] = max(0, corners[0])
    corners[1] = max(0, corners[1])
    corners[2] = min(img.shape[0], corners[2])
    corners[3] = min(img.shape[1], corners[3])

    # Extract patch with additional safety check
    try:
        patch = img[corners[0]:corners[2], corners[1]:corners[3]]

        # Check if patch is empty
        if patch.size == 0:
            return np.ones((out_size, out_size, 3), dtype=np.uint8) * np.array(border_value, dtype=np.uint8)

        # Resize to out_size
        patch = cv2.resize(patch, (out_size, out_size), interpolation=interp)

    except Exception as e:
        # Fallback in case of any error during crop/resize
        print(f"Warning: crop_and_resize failed: {e}")
        return np.ones((out_size, out_size, 3), dtype=np.uint8) * np.array(border_value, dtype=np.uint8)

    return patch

#    ------------------------
#    -                      -
#    -                      -
#    -  original image      -
#    -                      -
#    -                      -
#    -                      -
#    -                      -
#    -                      -
#    ------------------------

#### 假如search area在original image边界里时就不用padding
#    ------------------------
#    -                      -
#    -    ++++++++++        -
#    -    +        +        -
#    -    + search +        -
#    -    + aera   +        -
#    -    ++++++++++        -
#    -                      -
#    -                      -
#    ------------------------

#### 假如searcg area超出original image边界，做padding，且以超出边界中最大那个长度来padding
#### 则新conner变为原connor+padding num，向左上、右下展开

####                      padding to
####                    ---------------->                 
####                    (左2右2上2下2 总4)

#  +++++++++++++++++                                    *+++++++++++++++**************
#  +               +                                    *+ -           +             *
#  +               +                                    *+ -           +             *
#  + --------------+---------                           *+ ------------+-----------  *
#  + -             +        -                           *+ -           +          -  *
#  + -             +        -                           *+ -           +          -  *
#  +++++++++++++++++        -                           *+++++++++++++++          -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    -                      -                           *  -                      -  *
#    ------------------------                           *  ------------------------  *                          -
#                                                       *                            *
#                                                       *                            *
#                                                       ******************************  
#
