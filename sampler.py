import h5py
import numpy as np
import random
import cv2
import pickle
import scipy
import tensorflow as tf
import matplotlib
import matplotlib.cm

class Sampler:
    def __init__(self, h5_path, pkl_path, hw=256, upscaling=16):
        # load args
        self.hw = hw
        self.us = upscaling

        # Read dictionnary
        with open(pkl_path, 'rb') as handle:
            self.dataset = pickle.load(handle)
        self.num_samples = len(self.dataset.keys())
        self.h5 = h5py.File(h5_path,'r')

    def getDataset(self):
        generator = self._generator
        return tf.data.Dataset.from_generator(generator,
                              args=[],
                              output_types=(tf.float32, tf.float32),
                              output_shapes = (tf.TensorShape([self.hw, self.hw, 2]),tf.TensorShape([self.hw, self.hw, 1])))

    def _generator(self):
        # Generator (to act as dataset)
        keys = list(self.dataset.keys())
        random.shuffle(keys)
        for key in keys:
            dem, ort = self.dataset[key]
            img,lbl = self._getImg(dem,ort)
            yield (img, lbl)

    def _getImg(self, key_dem, key_ort):
        hw = 500+int(random.random()*498)
        res = 1000 - hw
        plx = int(random.random()*res)
        prx = res - plx
        ply = int(random.random()*res)
        pry = res - ply
        raw_ort = self.h5[key_ort][plx:-prx,ply:-pry]
        raw_dem = self.h5[key_dem][plx:-prx,ply:-pry]
        raw_dem = (raw_dem*1.0 - raw_dem.min())/(raw_dem.max() - raw_dem.min())
        raw_ort = cv2.resize(raw_ort,(self.hw,self.hw),cv2.INTER_CUBIC)
        raw_dem = cv2.resize(raw_dem,(self.hw,self.hw),cv2.INTER_CUBIC)
        raw_dem = raw_dem + random.random()*np.repeat(np.expand_dims(np.arange(self.hw,dtype=np.float32),-1),self.hw,-1)/(self.hw/2.0)
        raw_dem = raw_dem + random.random()*np.repeat(np.expand_dims(np.arange(self.hw,dtype=np.float32),0),self.hw,0)/(self.hw/2.0)
        raw_dem = (raw_dem*1.0 - raw_dem.min())/(raw_dem.max() - raw_dem.min())
        raw_dem = np.expand_dims(raw_dem - 0.5,-1)
        smt_ort = cv2.resize(cv2.resize(raw_dem,(self.hw//self.us,self.hw//self.us),cv2.INTER_AREA),(self.hw,self.hw),cv2.INTER_CUBIC)
        raw_ort = np.expand_dims(raw_ort/255.0 - 0.5,-1)
        img = np.concatenate([raw_ort,np.expand_dims(smt_ort,-1)],-1)
        assert not np.any(np.isnan(raw_dem))
        assert not np.any(np.isnan(raw_ort))
        #print(img.shape, raw_dem.shape)
        return img, raw_dem

@tf.function
def randomRotate(x, y):
    k = tf.random.uniform((),minval=0,maxval=4,dtype=tf.int32)
    x = tf.image.rot90(x, k=k)
    y = tf.image.rot90(y, k=k)
    return (x,y)

@tf.function
def randomBrightnessContrast(x,y, max_brightness_delta=0.2, max_contrast_factor=0.3):
    alpha = tf.random.uniform(()) * max_brightness_delta - max_brightness_delta/2
    beta = tf.random.uniform(()) * max_contrast_factor - max_contrast_factor/2
    img,dem = tf.split(x,2,-1)
    img = img*(1 + alpha) + beta
    return tf.concat([img,dem],-1), y

@tf.function
def randomFlip(x,y):
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_left_right(x)
        y = tf.image.flip_left_right(y)
    if tf.random.uniform(()) > 0.5:
        x = tf.image.flip_up_down(x)
        y = tf.image.flip_up_down(y)
    return x, y

@tf.function
def augmentImage(x,y):
    x,y = randomRotate(x,y)
    x,y = randomFlip(x,y)
    x,y = randomBrightnessContrast(x,y)
    return x,y

def colorize(value, vmin=None, vmax=None, cmap=None):
        """
        A utility function for TensorFlow that maps a grayscale image to a matplotlib
        colormap for use with TensorBoard image summaries.
        Arguments:
          - value: 2D Tensor of shape [height, width] or 3D Tensor of shape
            [height, width, 1].
          - vmin: the minimum value of the range used for normalization.
            (Default: value minimum)
          - vmax: the maximum value of the range used for normalization.
            (Default: value maximum)
          - cmap: a valid cmap named for use with matplotlib's `get_cmap`.
            (Default: 'gray')
        Example usage:
        ```
        output = tf.random_uniform(shape=[256, 256, 1])
        output_color = colorize(output, vmin=0.0, vmax=1.0, cmap='plasma')
        tf.summary.image('output', output_color)
        ```

        Returns a 3D tensor of shape [height, width, 3].
        """

        # normalize
        vmin = tf.reduce_min(value) if vmin is None else vmin
        vmax = tf.reduce_max(value) if vmax is None else vmax
        value = (value - vmin) / (vmax - vmin) # vmin..vmax

        # squeeze last dim if it exists
        value = tf.squeeze(value)

        # quantize
        indices = tf.cast(tf.round(value * 255),tf.int32)

        # gather
        cm = matplotlib.cm.get_cmap(cmap if cmap is not None else 'gray')
        colors = cm(np.arange(256))[:, :3]
        colors = tf.constant(colors, dtype=tf.float32)
        value = tf.gather(colors, indices)

        return value
