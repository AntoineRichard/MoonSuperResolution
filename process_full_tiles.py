from osgeo import gdal
import numpy as np 
import argparse
import cv2 
import os 

#from matplotlib import pyplot as plt

from spade.models.model import GauGAN

BATCH_SIZE = 16
IMAGE_SIZE = 512
STRIDE=16
gaugan = GauGAN(512, BATCH_SIZE, latent_dim=256)
gaugan.compile()
path = '/home/users/arichard/MoonProject/exp_spade/models/20220724-121426/epoch_6/'
gaugan.load(path+'generator',path+'discriminator',path+'encoder')

class DEMSuperResolution:
    def __init__(self, model=lambda x, training=False: x, folder_path=None, map_name=None, save_path=None, image_size=256, stride=64, tile_size=1024, batch_size=64):
        self.map_name=map_name
        self.save_path=save_path
        self.folder_path = folder_path
        self.left_image_name = "run-DRG.tif" 
        self.dem_name = "run-DEM.tif" 
        self.no_value = -32768.0
        self.stride = stride
        self.image_size = image_size
        self.batch_size = batch_size
        self.upsample_factor = 1
        self.tile_size = tile_size
        self.model = model

    def loadImages(self):
        img_path = os.path.join(self.folder_path, self.left_image_name) 
        dem_path = os.path.join(self.folder_path, self.dem_name) 

        img = gdal.Open(img_path,-1) 
        self.img = np.array(img.GetRasterBand(1).ReadAsArray(), dtype=np.float32)  
        dem = gdal.Open(dem_path,-1) 
        self.geo_transform = dem.GetGeoTransform()
        self.geo_projection = dem.GetProjection()
        self.dem = np.array(dem.GetRasterBand(1).ReadAsArray(), dtype=np.float32) 

        self.dem_shape = self.dem.shape
        self.img_shape = self.img.shape
        img_max = np.max(self.img)
        img_min = np.min(self.img[self.img > self.no_value])
        self.img = (self.img - img_min) / (img_max - img_min)

    def padInputs(self):
        # Pads the inputs such that the processing can be done with minimum memory footprint.
        # First we need to create something that can be cut into N by M tiles.
        # Then we need to be able to extend by (image_stride - stride) around these tiles to get the whole of the data.
        new_x_size = ((self.dem_shape[1] // 1024) + 1)*1024 + (self.image_size - self.stride)*2
        new_y_size = ((self.dem_shape[0] // 1024) + 1)*1024 + (self.image_size - self.stride)*2
        # We then need to copy the original images into the new padded images. 
        self.pad_x = new_x_size - self.dem_shape[1] - (self.image_size - self.stride)
        self.pad_y = new_y_size - self.dem_shape[0] - (self.image_size - self.stride)
        # We pad by the no_value value.
        self.dem_padded = np.ones((new_y_size, new_x_size), dtype=np.float32) * self.no_value
        self.img_padded = np.ones((new_y_size, new_x_size), dtype=np.float32) * self.no_value
        self.dem_padded[self.image_size - self.stride:-self.pad_y, self.image_size-self.stride:-self.pad_x] = self.dem
        self.img_padded[self.image_size - self.stride:-self.pad_y, self.image_size-self.stride:-self.pad_x] = self.img
        self.dem_padded_shape = self.dem_padded.shape
        self.img_padded_shape = self.img_padded.shape
        # Set the original dem and img to None to save some memory
        self.dem = None
        self.img = None

    def getPatch(self, px, py):
        is_valid = True
        img_patch = self.img_padded[py:py+self.image_size, px:px+self.image_size]
        dem_patch = self.dem_padded[py:py+self.image_size, px:px+self.image_size]
        # Checks if there are incorrect values inside the dem or image.
        if (img_patch==self.no_value).any():
            is_valid = False
        elif (dem_patch==self.no_value).any():
            is_valid = False
        return is_valid, img_patch, dem_patch

    def normalize(self, img_patch, dem_patch):
        img_patch_norm = img_patch - 0.5
        dem_patch_min_max = (dem_patch.min(), dem_patch.max())
        dem_patch_norm = ((dem_patch - dem_patch.min()) / (dem_patch.max() - dem_patch.min())) - 0.5
        inputs = np.concatenate([np.expand_dims(img_patch_norm, -1), np.expand_dims(dem_patch_norm, -1)], -1)
        return inputs, dem_patch_min_max

    def generateTileList(self):
        # Generates a list of tiles to be processed.
        # Can be used to distribute the load.
        tile_list = []
        for yy in range(0, self.dem_shape[0], self.tile_size):
            for xx in range(0, self.dem_shape[1], self.tile_size):
                tile_list.append((xx, yy))
        return tile_list

    def processBatch(self, batch, batch_index, patches):
        # Run the model in inference mode
        pred_dems = self.model(np.array(batch), training = False)
        # Transform to numyp and get rid of the last dim added by tensorflow.
        pred_dems = np.array(pred_dems)[:,:,:,-1] + 0.5
        for pred_dem, pred_idx in zip(pred_dems, batch_index):
            # Check if this is a padding element
            if pred_idx != (-1, -1):
                patches[pred_idx] = pred_dem

    def makeGaussianKernel(self):
        def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
            return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
        x = np.linspace(-self.image_size/2, self.image_size/2, self.image_size)
        y = np.linspace(-self.image_size/2, self.image_size/2, self.image_size)
        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
        kern = gaus2d(x, y,sx=self.image_size/4,sy=self.image_size/4)
        kern = (kern - kern.min())/(kern.max() - kern.min())
        return kern

    def rebuildTile(self, generated_dems, generated_minmax, K):
        # Generate empty maps
        votes = np.zeros((self.tile_size + self.image_size*2 - self.stride*2, self.tile_size + self.image_size*2 - self.stride*2), dtype=np.float32)
        tile = np.zeros((self.tile_size + self.image_size*2 - self.stride*2, self.tile_size + self.image_size*2 - self.stride*2), dtype=np.float32)
        tile2 = np.zeros((self.tile_size + self.image_size*2 - self.stride*2, self.tile_size + self.image_size*2 - self.stride*2), dtype=np.float32)
        tile3 = np.zeros((self.tile_size + self.image_size*2 - self.stride*2, self.tile_size + self.image_size*2 - self.stride*2), dtype=np.float32)
        # Creates gaussian kernel for smooth interpolation
        w = self.makeGaussianKernel()
        # Add the elements
        for key in generated_dems.keys():
            dem_patch = generated_dems[key]*(generated_minmax[key][1] - generated_minmax[key][0]) + generated_minmax[key][0]
            votes[key[1]:key[1] + self.image_size, key[0]:key[0] + self.image_size] += w
            tile[key[1]:key[1] + self.image_size, key[0]:key[0] + self.image_size] += dem_patch*w
            tile2[key[1]:key[1] + self.image_size, key[0]:key[0] + self.image_size] += (dem_patch - K)*w
            tile3[key[1]:key[1] + self.image_size, key[0]:key[0] + self.image_size] += np.square(dem_patch - K)*w
        # remove padding
        votes = votes[self.image_size-self.stride:-self.image_size+self.stride, self.image_size-self.stride:-self.image_size+self.stride]
        tile = tile[self.image_size-self.stride:-self.image_size+self.stride, self.image_size-self.stride:-self.image_size+self.stride]
        tile2 = tile2[self.image_size-self.stride:-self.image_size+self.stride, self.image_size-self.stride:-self.image_size+self.stride]
        tile3 = tile3[self.image_size-self.stride:-self.image_size+self.stride, self.image_size-self.stride:-self.image_size+self.stride]
        # Get the pixels that have been reconstructed
        good = (votes > 0) * 1.0
        # Compute the mean and standard deviation
        mean = (tile / votes)
        std = np.sqrt((tile3/votes) - np.square(tile2/votes))
        mean[good == 0] = self.no_value
        std[good == 0] = self.no_value
        return mean, std, good.astype(np.uint8)

    def saveTile(self, mean, std, good, name):
        os.makedirs(os.path.join(self.save_path, "tile_"+name), exist_ok=True)
        cv2.imwrite(os.path.join(self.save_path, "tile_"+name,"tile_"+name+"_mean.tif"), mean)
        cv2.imwrite(os.path.join(self.save_path, "tile_"+name,"tile_"+name+"_std.tif"), std)
        cv2.imwrite(os.path.join(self.save_path, "tile_"+name,"tile_"+name+"_correct.tif"), good)

    def processTile(self, px, py):
        generated_dems = {}
        generated_minmax = {}
        batch = []
        batch_index = []
        # Get the approximate average of the tile. Used to increase the stability of the standard deviation computation.
        tmp = self.dem_padded[py:py + self.tile_size + self.image_size - self.stride,px:px + self.tile_size + self.image_size - self.stride]
        K = tmp[tmp > self.no_value].mean()
        print(K)
        tmp = None
        # The image is padded by (image_size - stride)
        # To build a complete tile we need to go from -(image_size - stride) to (tile_size - stride)
        # So since the image is padded by (image_size - stride) we start with no offset and add go to tile_size + image_size - stride.
        # The second (- stride) is applied automatically by the range.
        for yy in range(py, py + self.tile_size + self.image_size - self.stride, self.stride):
            for xx in range(px, px + self.tile_size + self.image_size - self.stride, self.stride):
                valid, img_patch, dem_patch = self.getPatch(xx, yy)
                if not valid:
                    continue
                patch, dem_patch_min_max = self.normalize(img_patch, dem_patch)
                generated_minmax[(xx - px, yy - py)] = dem_patch_min_max
                batch.append(patch)
                batch_index.append((xx - px, yy - py))
                if len(batch) == self.batch_size:
                    self.processBatch(batch, batch_index, generated_dems)
                    # Resets the lists
                    batch = []
                    batch_index = []
        if len(batch) > 0:
            to_pad = self.batch_size - len(batch)
            for _ in range(to_pad):
                batch.append(np.zeros([self.image_size, self.image_size, 2]))
                batch_index.append((-1,-1))
            self.processBatch(batch, batch_index, generated_dems)
        mean, std, good = self.rebuildTile(generated_dems, generated_minmax, K)
        self.saveTile(mean, std, good, str(px)+"_"+str(py))

    def saveGTiff(self, data:np.ndarray, data_type:int, name:str):
        driver = gdal.GetDriverByName("GTiff")
        if data_type == np.float32:
            data_type = gdal.GDT_Float32
        elif data_type == np.float16:
            data_type = gdal.GDT_Float16
        elif data_type == np.uint8:
            data_type = gdal.GDT_UInt16
            data = data.astype(np.uint16)
        elif data_type == np.uint16:
            data_type = gdal.GDT_UInt16
        else:
            raise ValueError("Unknown data-type.")

        if len(data.shape) < 2:
            raise ValueError("Data is of incorrect shape. The array must be 2-dimensional at least.")
        elif len(data.shape) == 2:
            cols  = data.shape[1]
            rows = data.shape[0]
            channels = 1
            data = np.expand_dims(data,-1)
        elif len(data.shape) == 3:
            cols  = data.shape[1]
            rows = data.shape[0]
            channels = data.shape[2]
        else:
            raise ValueError("Data is of incorrect shape")

        tiff_writer = driver.Create(os.path.join(self.save_path, self.map_name+"_"+name+".tiff"), cols, rows, channels, data_type,options=['COMPRESS=LZW', 'PREDICTOR=2'])
        tiff_writer.SetGeoTransform(self.geo_transform)
        tiff_writer.SetProjection(self.geo_projection)
        for chn in range(1,1+channels,1):
            tiff_writer.GetRasterBand(chn).WriteArray(data[:,:,chn-1])
            tiff_writer.GetRasterBand(chn).SetNoDataValue(self.no_value)
        tiff_writer.FlushCache()

    def rebuildMap(self):
        mean = np.zeros(self.dem_padded_shape, dtype=np.float32)
        for yy in range(0, self.dem_shape[0], self.tile_size):
            for xx in range(0, self.dem_shape[1], self.tile_size):
                mean[yy:yy+self.tile_size,xx:xx+self.tile_size] = cv2.imread(os.path.join(self.save_path,"tile_"+str(xx)+"_"+str(yy),"tile_"+str(xx)+"_"+str(yy)+"_mean.tif"),-1)
        mean = mean[:-self.pad_y - self.image_size + self.stride,:-self.pad_x - self.image_size + self.stride]
        tmp_mean = mean[1024:2048, 1024:2048]
        self.saveGTiff(mean, mean.dtype,"mean")
        mean = None

        std =  np.zeros(self.dem_padded_shape, dtype=np.float32)
        for yy in range(0, self.dem_shape[0], self.tile_size):
            for xx in range(0, self.dem_shape[1], self.tile_size):
                std[yy:yy+self.tile_size,xx:xx+self.tile_size]  = cv2.imread(os.path.join(self.save_path,"tile_"+str(xx)+"_"+str(yy),"tile_"+str(xx)+"_"+str(yy)+"_std.tif"),-1)
        std = std[:-self.pad_y - self.image_size + self.stride,:-self.pad_x - self.image_size + self.stride]
        self.saveGTiff(std, std.dtype,"std")
        std = None

        good = np.zeros(self.dem_padded_shape, dtype=np.uint8)
        for yy in range(0, self.dem_shape[0], self.tile_size):
            for xx in range(0, self.dem_shape[1], self.tile_size):
                good[yy:yy+self.tile_size,xx:xx+self.tile_size] = cv2.imread(os.path.join(self.save_path,"tile_"+str(xx)+"_"+str(yy),"tile_"+str(xx)+"_"+str(yy)+"_correct.tif"),-1)
        good = good[:-self.pad_y - self.image_size + self.stride,:-self.pad_x - self.image_size + self.stride]
        self.saveGTiff(good, good.dtype,"good")
        good = None
        
    def processMap(self):
        self.loadImages()
        self.padInputs()
        tile_list = self.generateTileList()
        print("Cutting the image in",self.dem_shape[1]//self.tile_size + 1,"by",self.dem_shape[0]//self.tile_size + 1,"tiles.")
        for tile in tile_list:
            print("Processing tile",tile[0],tile[1])
            self.processTile(*tile)
        self.dem_padded = None
        self.img_padded = None
        self.rebuildMap()

if __name__ == '__main__':
    parser = argparse.ArgumentParser("Convert OBJ/STL assets to USD")
    parser.add_argument(
        "--folder_path", type=str, default=None, help="List of folders to convert (space seperated)."
    )
    parser.add_argument(
        "--save_path", type=str, default=None, help="List of folders to convert (space seperated)."
    )
    parser.add_argument(
        "--map_name", type=str, default=None, help="If specified, directly replaces the already existiing blocks."
    )
    parser.add_argument(
        "--run_name", type=str, default=None, help="If specified, directly replaces the already existiing blocks."
    )
    args, unknown_args = parser.parse_known_args()

    save_path = os.path.join(args.save_path,'SR_'+args.map_name)
    folder_path = os.path.join(args.folder_path,args.map_name,args.run_name+'_map')
    DSR = DEMSuperResolution(model=gaugan, map_name=args.map_name, save_path=save_path, folder_path=folder_path, image_size=IMAGE_SIZE, batch_size=BATCH_SIZE, stride=STRIDE)
    DSR.processMap()
