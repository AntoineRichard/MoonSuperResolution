import numpy as np
import cv2
from spade.models.model import GauGAN
import os
#MOS__ = cv2.imread('jose_color_part.png',-1)[:,:,0]
#DEM__ = cv2.imread('jose_dem_part_patched_10.tif',-1)
    
def getPatch(DEM,MOS,px,py,w=256,h=256):
    DEMp = DEM[px:px+w,py:py+h]
    MOSp = MOS[px:px+w,py:py+h]
    return DEMp, MOSp

def makeGaussianKernel(h=256,w=256):
    def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
    x = np.linspace(-int(w/2), int(w/2),int(w))
    y = np.linspace(-int(h/2), int(h/2), int(h))
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    kern = gaus2d(x, y,sx=int(w/4),sy=int(h/4))
    kern = (kern - kern.min())/(kern.max() - kern.min())
    return kern + 1e-7
    
BATCH_SIZE = 16
gaugan = GauGAN(512, BATCH_SIZE, latent_dim=256)
gaugan.compile()
path = '/home/gpu_user/antoine/MoonProject/exp_spade/models/20220724-121426/epoch_12/'
gaugan.load(path+'generator',path+'discriminator',path+'encoder')

class Buffer:
    def __init__(self, total_cols, total_rows, stride, block_size):
        self.kern = makeGaussianKernel(h=block_size, w=block_size)

        self.total_rows = total_rows
        self.total_cols = total_cols
        self.stride = stride
        self.block_size = block_size

        self.mean_dem = np.zeros((total_cols, total_rows))
        self.std_dem = np.zeros((total_cols, total_rows))
        #self.img = np.zeros((total_h, total_w))
        #self.img_c = np.zeros((total_h, total_w))
        #self.dlr = np.zeros((total_h, total_w))
        #self.dlr_c = np.zeros((total_h, total_w))
        self.demBuffer = {}
        self.nrmBuffer = {}
        #self.dlrBuffer = {}
        #self.imgBuffer = {}

    def getBufferSize(self, buff):
        c = 0
        for key in buff.keys():
            c += len(buff[key].keys())
        return c

    def addToDemBuffer(self, position, data):
        #print("Adding",position," to DEM Buffer")
        # Double stage buffer, row, col.
        if not (position[0] in self.demBuffer.keys()):
            self.demBuffer[position[0]] = {}
        # Add to buffer
        self.demBuffer[position[0]][position[1]] = data
        # Build new block in global map 
        if position[0] < self.stride:
            return
        if position[1] < self.stride:
            new_position = [0,0]
            new_position[0] = position[0] - self.stride
            stop_pos = self.total_rows - self.block_size
            for i in range(self.block_size // self.stride + 1):
                new_position[1] = stop_pos + self.stride*i
                self.addDEMToGlobalMap(new_position)
            self.cleanDEMBuffer(new_position)
            return
        self.addDEMToGlobalMap(position)
        self.cleanDEMBuffer(position)

    def addToNormBuffer(self, position, norm):
        # Double stage buffer, row, col.
        if not (position[0] in self.nrmBuffer.keys()) :
            self.nrmBuffer[position[0]] = {}
        # Add to buffer
        self.nrmBuffer[position[0]][position[1]] = norm

    def addDEMToGlobalMap(self, position):
        # Fetch data from buffer
        min_bound_row = position[0] - self.block_size
        max_bound_row = position[0]
        min_bound_col = position[1] - self.block_size
        max_bound_col = position[1]

        block_list = []
        gaussian_list = []
        std_list = []
        active = False
        # find rows

        available_rows = [key for key in self.demBuffer.keys() if ((min_bound_row<=key) and (key<max_bound_row))]
        for available_row in available_rows:
            # find cols
            available_cols = [key for key in self.demBuffer[available_row].keys() if ((min_bound_col<=key) and (key<max_bound_col))]
            for available_col in available_cols:
                active = True
                #print("Doing Something for",position,"with tile",[available_row, available_col])
                # get the index of the correct block inside the data
                sr = position[0] - available_row - self.stride
                sc = position[1] - available_col - self.stride
                #print("start row",sr,",start col",sc)
                # fetch the block, normalize, and apply gaussian
                nmin = self.nrmBuffer[available_row][available_col][0]
                nmax = self.nrmBuffer[available_row][available_col][1]
                block = self.demBuffer[available_row][available_col][sr:sr+self.stride, sc:sc+self.stride]
                block = block * (nmax - nmin) + nmin
                block_list.append(block * self.kern[sr:sr+self.stride, sc:sc+self.stride])
                gaussian_list.append(self.kern[sr:sr+self.stride, sc:sc+self.stride])
                #std_list.append(block)
        if active:
            new = np.sum(block_list, axis=0)/np.sum(gaussian_list,axis=0)
            self.mean_dem[position[0]-self.stride:position[0],position[1]-self.stride:position[1]] = new


    def cleanDEMBuffer(self, position):
        min_bound_row = position[0] - self.block_size
        min_bound_col = position[1] - self.block_size
        available_rows = [key for key in self.demBuffer.keys() if (key < min_bound_row)]
        for available_row in available_rows:
            available_cols = [key for key in self.demBuffer[available_row].keys() if (key < min_bound_col)]
            for available_col in available_cols:
                #print("Removing",[available_row, available_col])
                self.demBuffer[available_row].pop(available_col)

    def emptyBuffer(self):
        num_block_rows = self.total_rows // self.stride
        num_block_cols = self.total_cols // self.stride
        for i in range(num_block_rows):
            for j in range(num_block_cols):
                if self.mean_dem[i*self.stride,j*self.stride] != 0:
                    continue
                position = [(i+1)*self.stride, (j+1)*self.stride]
                self.addDEMToGlobalMap(position)
                continue
                if position[1] < self.stride:
                    new_position = [0,0]
                    new_position[0] = position[0] - self.stride
                    stop_pos = self.total_rows - self.block_size
                    for i in range(self.block_size // self.stride + 1):
                        new_position[1] = stop_pos + self.stride*i
                        self.addDEMToGlobalMap(new_position)
                    self.cleanDEMBuffer(new_position)
                self.addDEMToGlobalMap(position)
                self.cleanDEMBuffer(position)

def process(folder, gan_map, output):
    MOS__ = cv2.resize(np.load(os.path.join(folder,'color.npy'))[:,:,0], (0,0), fx=3, fy=3, interpolation=cv2.INTER_CUBIC)
    DEM__ = cv2.resize(np.load(os.path.join(gan_map,'generated_tile.npy')), (0,0), fx=9, fy=9, interpolation=cv2.INTER_CUBIC)
    #DEM__ = DEM__[:1500,:1500]
    #DEM__[0,0] = DEM__[-1,-1]
    MOS__ = MOS__[:DEM__.shape[0],:DEM__.shape[1]]
    
    H,W = DEM__.shape
    
    #DEM_low = {}
    #DEM_pred= {}
    #norm = {}
    #MOS_high = {}
    stride = 32
    block_size = 512
    num_tiles = DEM__.shape[0] // stride - block_size // stride + 1
    new_size = num_tiles*stride + block_size - stride

    Buff = Buffer(new_size, new_size, stride, block_size)
    print(num_tiles)
    count = 0
    batch = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            DEM, MOSp0 = getPatch(DEM__,MOS__,i*stride,j*stride, w=block_size, h=block_size)
            #MOS_high[str(i)+'-'+str(j)] = MOSp0 
            #DEM_low[str(i)+'-'+str(j)] = DEM.copy()
            #norm[str(i)+'-'+str(j)] = [DEM.min(), DEM.max()]
            norm = [DEM.min(), DEM.max()]
            Buff.addToNormBuffer([i*stride,j*stride], norm)
            DEM = (DEM - DEM.min()) / (DEM.max() - DEM.min())
            MOS = MOSp0 / 255.0
            #MOS = cv2.resize(MOS, (512,512), interpolation=cv2.INTER_CUBIC)
            #DEM = cv2.resize(DEM, (512,512), interpolation=cv2.INTER_CUBIC)
            dat = np.concatenate([np.expand_dims(MOS,-1)-0.5,np.expand_dims(DEM,-1)-0.5],-1)
            batch.append(dat)
            count += 1
            if count%BATCH_SIZE == 0:
                pred = gaugan(np.array(batch), training=False)
                dem = np.array(batch)[:,:,:,1]
                for iii, jjj in enumerate(range(count - BATCH_SIZE, count, 1)):
                    ii = jjj//num_tiles
                    jj = jjj%num_tiles
                    src = np.array(pred[iii,:,:,0] + 0.5)
                    Buff.addToDemBuffer([ii*stride, jj*stride], src)
                    #DEM_pred[str(ii)+'-'+str(jj)] = src#cv2.resize(src, (256,256), interpolation=cv2.INTER_CUBIC)
                batch = []
        to_pad = BATCH_SIZE - (count%BATCH_SIZE)
    for j in range(to_pad):
        dat = np.zeros_like(dat)
        batch.append(dat)
        count += 1
    pred = gaugan(np.array(batch))
    dem = np.array(batch)[:,:,:,1]
    for iii, jjj in enumerate(range(count-BATCH_SIZE, count-to_pad, 1)):
        ii = jjj//num_tiles
        jj = jjj%num_tiles
        #DEM_pred[str(ii)+'-'+str(jj)] = cv2.resize(np.array(pred[iii,:,:,0] + 0.5), (256,256), interpolation=cv2.INTER_CUBIC)
        src = pred[iii,:,:,0] + 0.5
        Buff.addToDemBuffer([ii*stride, jj*stride], src)
        #DEM_pred[str(ii)+'-'+str(jj)] = src
    Buff.emptyBuffer() 
    os.makedirs(os.path.join(output,folder), exist_ok=True)
    #w = makeGaussianKernel(h=512,w=512)
    #print(w) 
    # Assembling tiles
    #master = np.zeros((num_tiles*32+512-32,num_tiles*32+512-32))
    #counts = np.zeros((num_tiles*32+512-32,num_tiles*32+512-32))
    #for i in range(num_tiles):
    #    for j in range(num_tiles):
    #        counts[i*32:i*32+512,j*32:j*32+512] = counts[i*32:i*32+512,j*32:j*32+512] + w
    #        tmp1 = (DEM_pred[str(i)+'-'+str(j)]*(norm[str(i)+'-'+str(j)][1] - norm[str(i)+'-'+str(j)][0]) + norm[str(i)+'-'+str(j)][0])
    #        master[i*32:i*32+512,j*32:j*32+512] = master[i*32:i*32+512,j*32:j*32+512] + w*tmp1
    #master = master/counts
    X = np.save(os.path.join(output,folder,'generated_tile.npy'), Buff.mean_dem)
    
    #master = np.zeros((num_tiles*32+512-32,num_tiles*32+512-32))
    #counts = np.zeros((num_tiles*32+512-32,num_tiles*32+512-32))
    #for i in range(num_tiles):
    #    for j in range(num_tiles):
    #        counts[i*32:i*32+512,j*32:j*32+512] = counts[i*32:i*32+512,j*32:j*32+512] + 1
    #        master[i*32:i*32+512,j*32:j*32+512] = master[i*32:i*32+512,j*32:j*32+512] + DEM_low[str(i)+'-'+str(j)]
    #master = master/counts
    #X = np.save(os.path.join(output,folder,'lowres_tile.npy'),master)
    
    #master = np.zeros((num_tiles*32+512-32,num_tiles*32+512-32))
    #counts = np.zeros((num_tiles*32+512-32,num_tiles*32+512-32))
    #for i in range(num_tiles):
    #    for j in range(num_tiles):
    #        counts[i*32:i*32+512,j*32:j*32+512] = counts[i*32:i*32+512,j*32:j*32+512] + 1
    #        master[i*32:i*32+512,j*32:j*32+512] = master[i*32:i*32+512,j*32:j*32+512] + MOS_high[str(i)+'-'+str(j)]
    #master = master/counts
    #X = np.save(os.path.join(output,folder,'source_tile.npy'),master)

output = "MAP_1_GANonGAN_512"
input_f = "map_1"
input_2 = "MAP_1_GAN_512/map_1"
flist = os.listdir(input_f)
folders = [os.path.join(input_f, folder) for folder in flist]
folders_gan = [os.path.join(input_2, folder) for folder in flist]
for folder, folder2 in zip(folders, folders_gan):
    process(folder, folder2,  output)
