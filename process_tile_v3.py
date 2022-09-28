import numpy as np
import cv2
from spade.models.model import GauGAN
import os
#MOS__ = cv2.imread('jose_color_part.png',-1)[:,:,0]
#DEM__ = cv2.imread('jose_dem_part_patched_10.tif',-1)
    
def getPatch(DEM,MOS,px,py,w=256,h=256):
    print(px,py)
    DEMp = DEM[px:px+w,py:py+h]
    MOSp = MOS[px:px+w,py:py+h]
    return DEMp, MOSp

def makeGaussianKernel():
    def gaus2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
        return 1. / (2. * np.pi * sx * sy) * np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))
    x = np.linspace(-128, 128,256)
    y = np.linspace(-128, 128,256)
    x, y = np.meshgrid(x, y) # get 2D variables instead of 1D
    kern = gaus2d(x, y,sx=64,sy=64)
    kern = (kern - kern.min())/(kern.max() - kern.min())
    return kern
    
BATCH_SIZE = 16
gaugan = GauGAN(512, BATCH_SIZE, latent_dim=256)
gaugan.compile()
path = '/home/gpu_user/antoine/MoonProject/exp_spade/models/20220724-121426/epoch_12/'
gaugan.load(path+'generator',path+'discriminator',path+'encoder')


def process(folder, output):
    MOS__ = cv2.resize(np.load(os.path.join(folder,'color.npy'))[:,:,0],(1000,1000),cv2.INTER_AREA)
    DEM__ = cv2.resize(np.load(os.path.join(folder,'dem.npy')),(1000,1000),cv2.INTER_AREA)
    H,W = DEM__.shape
    
    DEM_low = {}
    DEM_pred= {}
    norm = {}
    MOS_high = {}
    
    num_tiles = MOS__.shape[0] // 16 - 256//16
    print(num_tiles)
    count = 0
    batch = []
    for i in range(num_tiles):
        for j in range(num_tiles):
            DEM, MOSp0 = getPatch(DEM__,MOS__,i*16,j*16)
            MOS_high[str(i)+'-'+str(j)] = MOSp0 
            DEM_low[str(i)+'-'+str(j)] = DEM.copy()
            norm[str(i)+'-'+str(j)] = [DEM.min(), DEM.max()]
            DEM = (DEM - DEM.min()) / (DEM.max() - DEM.min())
            MOS = MOSp0 / 255.0
            MOS = cv2.resize(MOS, (512,512), interpolation=cv2.INTER_CUBIC)
            DEM = cv2.resize(DEM, (512,512), interpolation=cv2.INTER_CUBIC)
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
                    DEM_pred[str(ii)+'-'+str(jj)] = cv2.resize(src, (256,256), interpolation=cv2.INTER_CUBIC)
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
        DEM_pred[str(ii)+'-'+str(jj)] = cv2.resize(np.array(pred[iii,:,:,0] + 0.5), (256,256), interpolation=cv2.INTER_CUBIC)
    
    
    np.save(os.path.join(output,folder,'generated_tiles.npy'),DEM_pred)
    np.save(os.path.join(output,folder,'lowres_tiles.npy'),DEM_low)
    np.save(os.path.join(output,folder,'image_tiles.npy'),MOS_high)
    os.makedirs(os.path.join(output,folder), exist_ok=True)
    w = makeGaussianKernel()
    
    # Assembling tiles
    master = np.zeros((num_tiles*16+256-16,num_tiles*16+256-16))
    counts = np.zeros((num_tiles*16+256-16,num_tiles*16+256-16))
    for i in range(num_tiles):
        for j in range(num_tiles):
            counts[i*16:i*16+256,j*16:j*16+256] = counts[i*16:i*16+256,j*16:j*16+256] + w
            tmp1 = (DEM_pred[str(i)+'-'+str(j)]*(norm[str(i)+'-'+str(j)][1] - norm[str(i)+'-'+str(j)][0]) + norm[str(i)+'-'+str(j)][0])
            print(tmp1.shape)
            master[i*16:i*16+256,j*16:j*16+256] = master[i*16:i*16+256,j*16:j*16+256] + w*tmp1
    master = master/counts
    X = np.save(os.path.join(output,folder,'generated_tile.npy'),master)
    
    master = np.zeros((num_tiles*16+256-16,num_tiles*16+256-16))
    counts = np.zeros((num_tiles*16+256-16,num_tiles*16+256-16))
    for i in range(num_tiles):
        for j in range(num_tiles):
            counts[i*16:i*16+256,j*16:j*16+256] = counts[i*16:i*16+256,j*16:j*16+256] + 1
            master[i*16:i*16+256,j*16:j*16+256] = master[i*16:i*16+256,j*16:j*16+256] + DEM_low[str(i)+'-'+str(j)]
    master = master/counts
    X = np.save(os.path.join(output,folder,'lowres_tile.npy'),master)
    
    master = np.zeros((num_tiles*16+256-16,num_tiles*16+256-16))
    counts = np.zeros((num_tiles*16+256-16,num_tiles*16+256-16))
    for i in range(num_tiles):
        for j in range(num_tiles):
            counts[i*16:i*16+256,j*16:j*16+256] = counts[i*16:i*16+256,j*16:j*16+256] + 1
            master[i*16:i*16+256,j*16:j*16+256] = master[i*16:i*16+256,j*16:j*16+256] + MOS_high[str(i)+'-'+str(j)]
    master = master/counts
    X = np.save(os.path.join(output,folder,'source_tile.npy'),master)

output = "MAP_1_GAN_512"
input_f = "map_1"
folders = [os.path.join(input_f, folder) for folder in os.listdir(input_f)]
for folder in folders:
    process(folder, output)

