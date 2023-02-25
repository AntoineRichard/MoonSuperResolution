from typing import Optional, Tuple, Union, List
from osgeo import gdal
import dataclasses
import numpy as np 
import argparse
import cv2 
import os 

from spade.models.model import GauGAN


def load_GAN_model(path: str, image_size: int, batch_size: int) -> GauGAN:
    """ Creates a SPADE model and loads the model weights provided by the user.

    Args:
        path (str): The path to the weight of the network.
        image_size (int): The size of the images the network should ingest.
        batch_size (int): The batch size to be used by the network.

    Returns:
        TensorFlow-Model: The gan model.

    Raises:
        assert Ensures the weight path given by the user exists.
    """
    assert os.path.exists(path), "The path to the neural-network weight is invalid. Please ensure you gave a valid path."
    gaugan = GauGAN(image_size, batch_size, latent_dim=256)
    gaugan.compile()
    gaugan.load(path+'generator',path+'discriminator',path+'encoder')
    return gaugan

@dataclasses.dataclass
class DSRConfig:
    image_size: int = 256
    stride: int = 32
    batch_size: int = 16
    tile_size: int = 1024
    no_value: float = -32768.0
    upsample_factor: float = 1.0
    map_name: str = None
    save_path: str = None
    source_folder_path: str = None
    ortho_image_name: str = "run-DRG.tif"
    dem_name: str = "run-DEM.tif"
    model_path: str = None

def parse_args() -> DSRConfig:
    """ Parses the arguments provided by the user.

    Returns:
        DSRConfig : The configuration of DEM Super Resolution pipeline.
    """
    parser = argparse.ArgumentParser("DEM Super Resolution config parser.")
    parser.add_argument(
        "--source_folder_path", type=str, required=True, default=None, help="The path to the folder containing both the ortho image and the DEM."
    )
    parser.add_argument(
        "--map_name", type=str, required=True, default=None, help="The name of the map to be processes."
    )
    parser.add_argument(
        "--save_path", type=str, required=True, default=None, help="The path to the folder where the reconstructed map will be stored. It will be named SR_MAP-NAME"
    )
    parser.add_argument(
        "--ortho_image_name", type=str, default="run-DRG.tif", help="The name of the ortho image."
    )
    parser.add_argument(
        "--dem_name", type=str, default="run-DEM.tif", help="The name of the DEM image."
    )
    parser.add_argument(
        "--model_path", type=str, default=None, help="The path to the model. Do not specify to run indentity processing."
    )
    parser.add_argument(
        "--image_size", type=int, default=256, help="The size of the images the model can process."
    )
    parser.add_argument(
        "--stride", type=int, default=32, help="The amount of displacement between two images.\
                                                A small stride should yield better results and more accurate std measurements.\
                                                A large stride will be faster to process, but may be less resolute.\
                                                A good stride value is 1/8th of the image size."
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="The size of the images the model can process."
    )
    parser.add_argument(
        "--tile_size", type=int, default=1024, help="The size of the tiles used to limit the memory footprint of the program. It can also be useful to easily post-process the DEMs."
    )
    parser.add_argument(
        "--no_value", type=int, default=-32768.0, help="The value used in the DEM generation to specify points where there are no values."
    )
    parser.add_argument(
        "--upsample_factor", type=float, default=1.0, help="How much the DEM should be upsampled before being processed. Not used for now."
    ) 
    args, unknown_args = parser.parse_known_args()
    config = DSRConfig(source_folder_path=args.source_folder_path,
                       map_name=args.map_name,
                       save_path=args.save_path,
                       ortho_image_name=args.ortho_image_name,
                       dem_name=args.dem_name,
                       model_path=args.model_path,
                       image_size=args.image_size,
                       stride=args.stride,
                       batch_size=args.batch_size,
                       tile_size=args.tile_size,
                       no_value=args.no_value,
                       upsample_factor=args.upsample_factor)
    return config

class DEMSuperResolution:
    """ Allows to perform DEM super resolution on large images.
    This object takes as input a configuration and a callable neural-network model.
    Once initializated this processing tools loads the original dem, and reference ortho-image,
    and processes them in a tiled fashion to prevent excessive RAM usage.
    Each time a tile is processed, it is saved to disk. At the end of the processing the tiles are
    loaded one by one and assembled inside a single geo-tiff.

    Args:
        config (DSRConfig): The configuration to be used by the super-resolution process.
        model (function or class, Optional): A function or callable class that is used to perform the super-resolution.
                                             By default this is set to an identity funciton.
                                             This allows to check that the code is behaving as expected.
    """
    def __init__(self, config: DSRConfig, model=lambda x, training=False: x) -> None:
        self.map_name = config.map_name
        self.save_path = config.save_path
        self.folder_path = config.source_folder_path
        self.left_image_name = config.ortho_image_name 
        self.dem_name = config.dem_name
        self.no_value = config.no_value
        self.stride = config.stride
        self.image_size = config.image_size
        self.batch_size = config.batch_size
        self.upsample_factor = 1
        self.tile_size = config.tile_size
        self.model = model
        return

    def loadImages(self) -> None:
        """ Loads the dem and ortho-image, and extracts some basic information about them, such as the geo-data and shapes.

        Raises:
            Exception: if the provided paths do not exist.
        """
        # Check paths
        img_path = os.path.join(self.folder_path, self.left_image_name) 
        dem_path = os.path.join(self.folder_path, self.dem_name) 
        if not os.path.exists(img_path):
            raise ValueError("The path given for the ortho-image does not exist. Provided path is: "+img_path)
        if not os.path.exists(dem_path):
            raise ValueError("The path given for the dem does not exist. Provided path is: "+dem_path)
        # Load images
        img = gdal.Open(img_path,-1) 
        self.img = np.array(img.GetRasterBand(1).ReadAsArray(), dtype=np.float32)  
        dem = gdal.Open(dem_path,-1) 
        self.dem = np.array(dem.GetRasterBand(1).ReadAsArray(), dtype=np.float32) 
        # Collect geo-data
        self.geo_transform = dem.GetGeoTransform()
        self.geo_projection = dem.GetProjection()
        # Collect shapes
        self.dem_shape = self.dem.shape
        self.img_shape = self.img.shape
        # Normalizes the ortho-image (the DEMs are instanced normalize later)
        self.img_max = np.max(self.img)
        self.img_min = np.min(self.img[self.img > self.no_value])
        return

    def padInputs(self) -> None:
        """ Pads the initial images to ease the tiling operation.
        """
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
        return

    def getPatch(self, px: int, py: int) -> Tuple[bool, np.ndarray, np.ndarray]:
        """ Gets a image and dem patch using the provided coordinates (top-left corner).
        The returned images will be of size [self.image_size, self.image_size].
        In addition to returning images, it also returns if the image is valid or not.
        A image is valid if it has no pixels equal to the no-value value (self.no_value).

        Args:
            px (int): The x position of the top left corner.
            py (int): The y position of the top left corner.

        Returns:
            Tuple[bool, np.ndarray, np.ndarray]: True if the data is valid, i.e. there is no incorrect values inside it.
                                                The ortho-image patch. The dem patch.

        """
        is_valid = True
        # Gets the patches
        img_patch = self.img_padded[py:py+self.image_size, px:px+self.image_size]
        dem_patch = self.dem_padded[py:py+self.image_size, px:px+self.image_size]
        # Checks if there are incorrect values inside the dem or image.
        if (img_patch<=self.no_value).any():
            is_valid = False
        elif (dem_patch<=self.no_value).any():
            is_valid = False
        return is_valid, img_patch, dem_patch

    def normalize(self, img_patch: np.ndarray, dem_patch: np.ndarray) -> Tuple[np.ndarray, Tuple[float,float]]:
        """ Normalizes the dem patch and stores its min and max values to de-normalize it afterwards.
        It also makes the image patch zero-centered.

        Args:
            img_patch (np.ndarray): The image patch of size [self.image_size,self.image_size]
            dem_patch (np.ndarray): The dem patch of size [self.image_size, self.image_size]

        Returns:
            Tuple[np.ndarray, Tuple[float, float]]]: The normalized, zero-centered, image and dem patches.
                                     And the (min,max) of the DEM before normalization.
        """
        img_patch_norm = ((img_patch - self.img_min) / (self.img_max - self.img_min)) - 0.5
        dem_patch_min_max = (dem_patch.min(), dem_patch.max())
        dem_patch_norm = ((dem_patch - dem_patch.min()) / (dem_patch.max() - dem_patch.min())) - 0.5
        inputs = np.concatenate([np.expand_dims(img_patch_norm, -1), np.expand_dims(dem_patch_norm, -1)], -1)
        return inputs, dem_patch_min_max

    def generateTileList(self) -> List[str]:
        """ Creates a list of tiles to be processed.

        Returns:
            List[str]: The list of tiles to be processed.
        """
        # Generates a list of tiles to be processed.
        # Can be used to distribute the load.
        tile_list = []
        for yy in range(0, self.dem_shape[0], self.tile_size):
            for xx in range(0, self.dem_shape[1], self.tile_size):
                tile_list.append((xx, yy))
        return tile_list

    def processBatch(self, batch: List[np.ndarray], batch_index: List[Tuple[int,int]], patches: dict) -> None:
        """ Takes a batch and sends it through the neural-network.
        Once this is done, each element of the patch is indexed by its position
        inside the dem, and the zero-centering is being removed.

        Args:
            batch (List[np.ndarray]): The batch to be processed, a normalized inputs.
            batch_index (List[Tuple[int,int]]): The index in the image of each element in the batch.
            patches (dict): A dictionary of generated dem indexed by their position in the image.
        """
        # Run the model in inference mode
        pred_dems = self.model(np.array(batch), training = False)
        # Transform to numyp and get rid of the last dim added by tensorflow.
        pred_dems = np.array(pred_dems)[:,:,:,-1] + 0.5
        for pred_dem, pred_idx in zip(pred_dems, batch_index):
            # Check if this is a padding element
            if pred_idx != (-1, -1):
                patches[pred_idx] = pred_dem
        return

    def makeGaussianKernel(self) -> np.ndarray:
        """ Makes a gaussian kernel of size (image_size, image_size).
        This kernel is used when assembling multiple generated images together.

        Returns:
            np.ndarray: The generated gaussian kernel
        """
        def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
            return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
        x = np.linspace(-self.image_size/2, self.image_size/2, self.image_size)
        y = np.linspace(-self.image_size/2, self.image_size/2, self.image_size)
        x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
        kern = gaus2d(x, y,sx=self.image_size/5,sy=self.image_size/5)
        kern = (kern - kern.min())/(kern.max() - kern.min())
        return kern

    def rebuildTile(self, generated_dems: dict, generated_minmax: dict) -> List[np.ndarray]:
        """ Rebuilds a tile using the generated dems. When the tile is being generated,
        there is a certain amount of overlapp between each generated images, to be concise there
        is (self.image_size - self.stride) overlapping pixels. We recommend to set the stride to 1/8th
        of the image size. This means that each tile can be in seen as an assembly of (image_size/8) patches.
        For each patches there should be 64 (8x8) overlapping generations. This overlapp allows us to not only
        average the generations and get a smoother output, but also estimate the network's uncertainty. These
        overlapping generations act as monte-carlo estimator.

        Note: When averaging the generated images, each pixels inside the image is not given the same weight.
        The closer the pixels are to the center of the image the higher their weight. This prevents weird border
        effects to occur and thus improve the blending of the generated images.

        Args:
            generated_dems (dict): The dictionary containing all the generated images indexed by their
                                   position in the image.
            generated_minmax (dict): The dictionary containing the min and max of the generated images
                                     indexed by their position in the image. 
        Returns:
            List[np.ndarray]: The mean, the standard deviation, and the valid pixels
                                                      of the images generated by the GANs.
        """
        # Generate empty maps
        w_sum = np.zeros((self.tile_size + self.image_size*2 - self.stride*2, self.tile_size + self.image_size*2 - self.stride*2), dtype=np.float32)
        w_sum2 = np.zeros((self.tile_size + self.image_size*2 - self.stride*2, self.tile_size + self.image_size*2 - self.stride*2), dtype=np.float32)
        mean = np.zeros((self.tile_size + self.image_size*2 - self.stride*2, self.tile_size + self.image_size*2 - self.stride*2), dtype=np.float32)
        S = np.zeros((self.tile_size + self.image_size*2 - self.stride*2, self.tile_size + self.image_size*2 - self.stride*2), dtype=np.float32)
        # Gaussian kernel for smooth blending
        w = self.makeGaussianKernel() + 1e-7
        purge = self.image_size // 16
        w = w[purge:-purge,purge:-purge]
        # Add the elements using the weighted incremental algorithm
        for key in generated_dems.keys():
            dem_patch = generated_dems[key]*(generated_minmax[key][1] - generated_minmax[key][0]) + generated_minmax[key][0]
            dem_patch = dem_patch[purge:-purge,purge:-purge]
            w_sum[key[1]+purge:key[1] + self.image_size-purge, key[0]+purge:key[0] + self.image_size-purge] += w
            w_sum2[key[1]+purge:key[1] + self.image_size-purge, key[0]+purge:key[0] + self.image_size-purge] += w**2
            mean_old = mean[key[1]+purge:key[1] + self.image_size-purge, key[0]+purge:key[0] + self.image_size-purge]
            mean[key[1]+purge:key[1] + self.image_size-purge, key[0]+purge:key[0] + self.image_size-purge] = mean_old + (w / w_sum[key[1]+purge:key[1] + self.image_size-purge, key[0]+purge:key[0] + self.image_size-purge]) * (dem_patch - mean_old)
            S[key[1]+purge:key[1] + self.image_size-purge, key[0]+purge:key[0] + self.image_size-purge] += w * (dem_patch - mean_old) * (dem_patch - mean[key[1]+purge:key[1] + self.image_size-purge, key[0]+purge:key[0] + self.image_size-purge])
        # remove padding
        w_sum = w_sum[self.image_size-self.stride:-self.image_size+self.stride, self.image_size-self.stride:-self.image_size+self.stride]
        w_sum2 = w_sum2[self.image_size-self.stride:-self.image_size+self.stride, self.image_size-self.stride:-self.image_size+self.stride]
        mean = mean[self.image_size-self.stride:-self.image_size+self.stride, self.image_size-self.stride:-self.image_size+self.stride]
        S = S[self.image_size-self.stride:-self.image_size+self.stride, self.image_size-self.stride:-self.image_size+self.stride]
        # Get the pixels that have been reconstructed
        good = (w_sum > 0) * 1.0
        # Compute the mean and standard deviation
        std = np.sqrt(S / w_sum)
        mean[good == 0] = self.no_value
        std[good == 0] = self.no_value
        return mean, std, good.astype(np.uint8)

    def saveTile(self, mean: np.ndarray, std: np.ndarray, good:np.ndarray, name:str) -> None:
        """ Save the generated tiles.

        Args:
            mean (np.ndarray): The mean of the tile to be saved.
            std (np.ndarray): The standard deviation of the tile to be saved.
            good (np.ndarray): Whether a pixel is reconstructed or not.
            name (str): The name of the tile to be saved.
        """
        os.makedirs(os.path.join(self.save_path, "tile_"+name), exist_ok=True)
        cv2.imwrite(os.path.join(self.save_path, "tile_"+name,"tile_"+name+"_mean.tif"), mean)
        cv2.imwrite(os.path.join(self.save_path, "tile_"+name,"tile_"+name+"_std.tif"), std)
        cv2.imwrite(os.path.join(self.save_path, "tile_"+name,"tile_"+name+"_correct.tif"), good)
        return

    def processTile(self, px: int, py: int) -> None:
        """ Takes a tile an applies the super-resolution pipeline on it. Collects patches of the padded tile
        normalizes them, and stores them into a batch. If a patch contains incorrect values (self.no_value),
        then this patch is not included in the reconstruction. Each element of the batch has its top-left
        corner coordinates stored so the tile can be reconstructed later on.
        When all the tiles have been processed, the tile is reconstructed using the weighted incremental 
        algorithm for variance estimation. Each patch is weighted by a gaussian whose mean is set
        to the patch center. The borders of the patch are also removed to prevent border effects from 
        degrading the reconstruction.

        Args:
            px (int): The x coordinate of the tile to be processed.
            py (int): The y coordinate of the tile to be processed.
        """
        generated_dems = {}
        generated_minmax = {}
        batch = []
        batch_index = []
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
        # Check if there are some tiles left to be processed.
        if len(batch) > 0:
            # If so, add padding images.
            to_pad = self.batch_size - len(batch)
            for _ in range(to_pad):
                batch.append(np.zeros([self.image_size, self.image_size, 2]))
                batch_index.append((-1,-1))
            self.processBatch(batch, batch_index, generated_dems)
        # Reconstruct the tile.
        mean, std, good = self.rebuildTile(generated_dems, generated_minmax)
        # Save to disk to minimize RAM usage.
        self.saveTile(mean, std, good, str(px)+"_"+str(py))
        return

    def saveGTiff(self, data:np.ndarray, data_type:int, name:str) -> None:
        """ Saves the assembled tiles as a geo-tiff.

        Args:
            data (np.ndarray): The data to be saved.
            data-type (int): The type of the data.
            name (str): The name to give it.

        Raises:
            Exception if the data-type is not supported.
            Exception if the data is not of the correct shape.
        """
        # Gets the correct data-type to save the data as a geo-tiff.
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
            raise ValueError("Unsupported data-type.")
        # Ensures the data is properly shaped
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
        # Creates the driver to save the data.
        tiff_writer = driver.Create(os.path.join(self.save_path, self.map_name+"_"+name+".tiff"), cols, rows, channels, data_type,options=['COMPRESS=LZW', 'PREDICTOR=2'])
        # Adds the geo-data.
        tiff_writer.SetGeoTransform(self.geo_transform)
        tiff_writer.SetProjection(self.geo_projection)
        # Loops for as many bands as needed.
        for chn in range(1,1+channels,1):
            tiff_writer.GetRasterBand(chn).WriteArray(data[:,:,chn-1])
            tiff_writer.GetRasterBand(chn).SetNoDataValue(self.no_value)
        # Saves the data.
        tiff_writer.FlushCache()
        return

    def rebuildMap(self) -> None:
        """ Assembles the tiles together and saves them.

        Note: The tiles are loaded from disk, and not read from RAM.

        TODO: Make a function for the assembly process.
        """
        # Reconstructs the mean map.
        mean = np.zeros(self.dem_padded_shape, dtype=np.float32)
        for yy in range(0, self.dem_shape[0], self.tile_size):
            for xx in range(0, self.dem_shape[1], self.tile_size):
                mean[yy:yy+self.tile_size,xx:xx+self.tile_size] = cv2.imread(os.path.join(self.save_path,"tile_"+str(xx)+"_"+str(yy),"tile_"+str(xx)+"_"+str(yy)+"_mean.tif"),-1)
        mean = mean[:-self.pad_y - self.image_size + self.stride,:-self.pad_x - self.image_size + self.stride]
        self.saveGTiff(mean, mean.dtype,"mean")
        mean = None

        # Reconstructs the standard deviation map.
        std =  np.zeros(self.dem_padded_shape, dtype=np.float32)
        for yy in range(0, self.dem_shape[0], self.tile_size):
            for xx in range(0, self.dem_shape[1], self.tile_size):
                std[yy:yy+self.tile_size,xx:xx+self.tile_size]  = cv2.imread(os.path.join(self.save_path,"tile_"+str(xx)+"_"+str(yy),"tile_"+str(xx)+"_"+str(yy)+"_std.tif"),-1)
        std = std[:-self.pad_y - self.image_size + self.stride,:-self.pad_x - self.image_size + self.stride]
        self.saveGTiff(std, std.dtype,"std")
        std = None

        # Reconstructs the good pixel map.
        good = np.zeros(self.dem_padded_shape, dtype=np.uint8)
        for yy in range(0, self.dem_shape[0], self.tile_size):
            for xx in range(0, self.dem_shape[1], self.tile_size):
                good[yy:yy+self.tile_size,xx:xx+self.tile_size] = cv2.imread(os.path.join(self.save_path,"tile_"+str(xx)+"_"+str(yy),"tile_"+str(xx)+"_"+str(yy)+"_correct.tif"),-1)
        good = good[:-self.pad_y - self.image_size + self.stride,:-self.pad_x - self.image_size + self.stride]
        self.saveGTiff(good, good.dtype,"good")
        good = None
        return
        
    def processMap(self) -> None:
        """ Takes a DEM and associates ortho-image and perform super-resolution on them.
        """
        # Loads and prepares the data.
        self.loadImages()
        self.padInputs()
        # Gets the list of tiles to process.
        tile_list = self.generateTileList()
        # Processes each tile one by one.
        print("Cutting the image in",self.dem_shape[1]//self.tile_size + 1,"by",self.dem_shape[0]//self.tile_size + 1,"tiles.")
        for tile in tile_list:
            print("Processing tile",tile[0],tile[1])
            self.processTile(*tile)
        # Removes the original data.
        self.dem_padded = None
        self.img_padded = None
        # Assembles the tiles.
        self.rebuildMap()
        return

if __name__ == '__main__':
    DSR_cfg = parse_args()
    gaugan = load_GAN_model(DSR_cfg.model_path, DSR_cfg.image_size, DSR_cfg.batch_size)
    DSR = DEMSuperResolution(DSR_cfg, model=gaugan)
    DSR.processMap()
