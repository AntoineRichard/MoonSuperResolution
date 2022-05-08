import numpy as np
import cv2
from spade.models.model import GauGAN

MOS__ = np.load('data/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0n_60n_240_360.npy')
DEM__ = np.fromfile('data/sldem2015_256_0n_60n_240_360_float.img',dtype=np.float32)
DEM__ = DEM__.reshape([15360,-1])
H,W = DEM__.shape
MOS__ = cv2.resize(MOS__,(W,H),cv2.INTER_AREA)

def getPatch(DEM,MOS,px,py,w=256,h=256):
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


DEM_high = {}
DEM_low = {}
DEM_pred= {}
norm = {}
MOS_high = {}

gaugan = GauGAN(256, 64, latent_dim=256)
gaugan.compile()
#path = '/home/gpu_user/antoine/MoonProject/trainings/spade_consistency_loss/models/20220508-083520/epoch_30/'
path = '/home/gpu_user/antoine/MoonProject/trainings/spade_consistency_loss/models/20220508-144559/epoch_12/'
gaugan.load(path+'generator',path+'discriminator',path+'encoder')
# Generating high resolution tiles
for i in range(64):
    batch = []
    for j in range(64):
        DEMp0, MOSp0 = getPatch(DEM__,MOS__,i*16,j*16)
        DEM_high[str(i)+'-'+str(j)] = DEMp0
        MOS_high[str(i)+'-'+str(j)] = MOSp0 
        DEMrs = cv2.resize(DEMp0, (25,25), cv2.INTER_AREA)
        DEM = cv2.resize(DEMrs, (256,256), cv2.INTER_CUBIC)
        DEM_low[str(i)+'-'+str(j)] = DEM
        norm[str(i)+'-'+str(j)] = [DEM.min(), DEM.max()]
        DEM = (DEM - DEM.min()) / (DEM.max() - DEM.min())
        MOS = MOSp0 / 255.0
        dat = np.concatenate([np.expand_dims(MOS,-1)-0.5,np.expand_dims(DEM,-1)-0.5],-1)
        batch.append(dat)
    pred = gaugan(np.array(batch))
    dem = np.array(batch)[:,:,:,1]
    for j in range(64):
        DEM_pred[str(i)+'-'+str(j)] = pred[j,:,:,0] + 0.5

X = np.save('test_source_tile-sample-gen.npy',DEM_pred['0-0'])
X = np.save('test_source_tile-sample-lowres.npy',DEM_low['0-0'])
X = np.save('test_source_tile-sample-true.npy',DEM_high['0-0'])
X = np.save('test_source_tile-sample-WAC.npy',MOS_high['0-0'])

w = makeGaussianKernel()

# Assembling tiles
master = np.zeros((64*16+256-16,64*16+256-16))
counts = np.zeros((64*16+256-16,64*16+256-16))
for i in range(64):
    for j in range(64):
        counts[i*16:i*16+256,j*16:j*16+256] = counts[i*16:i*16+256,j*16:j*16+256] + w
        tmp1 = (DEM_pred[str(i)+'-'+str(j)]*(norm[str(i)+'-'+str(j)][1] - norm[str(i)+'-'+str(j)][0]) + norm[str(i)+'-'+str(j)][0])
        master[i*16:i*16+256,j*16:j*16+256] = master[i*16:i*16+256,j*16:j*16+256] + w*tmp1
        #tmp2 = DEM_low[str(i)+'-'+str(j)]
        #tmp3 = DEM_high[str(i)+'-'+str(j)]
        #print(np.mean(np.square(tmp2 -tmp1)))
        #print(np.mean(np.square(tmp3 -tmp1)))
master = master/counts
X = np.save('test_generated_tile.npy',master)
master = np.zeros((64*16+256-16,64*16+256-16))
counts = np.zeros((64*16+256-16,64*16+256-16))
for i in range(64):
    for j in range(64):
        counts[i*16:i*16+256,j*16:j*16+256] = counts[i*16:i*16+256,j*16:j*16+256] + 1
        master[i*16:i*16+256,j*16:j*16+256] = master[i*16:i*16+256,j*16:j*16+256] + DEM_low[str(i)+'-'+str(j)]
master = master/counts
X = np.save('test_lowres_tile.npy',master)
master = np.zeros((64*16+256-16,64*16+256-16))
counts = np.zeros((64*16+256-16,64*16+256-16))
for i in range(64):
    for j in range(64):
        counts[i*16:i*16+256-16,j*16:j*16+256-16] = counts[i*16:i*16+256,j*16:j*16+256] + 1
        master[i*16:i*16+256-16,j*16:j*16+256-16] = master[i*16:i*16+256,j*16:j*16+256] + DEM_high[str(i)+'-'+str(j)]
master = master/counts
X = np.save('test_true_tile.npy',master)
master = np.zeros((64*16+256-16,64*16+256-16))
counts = np.zeros((64*16+256-16,64*16+256-16))
for i in range(64):
    for j in range(64):
        counts[i*16:i*16+256,j*16:j*16+256] = counts[i*16:i*16+256,j*16:j*16+256] + 1
        master[i*16:i*16+256,j*16:j*16+256] = master[i*16:i*16+256,j*16:j*16+256] + MOS_high[str(i)+'-'+str(j)]
master = master/counts
X = np.save('test_source_tile.npy',master)
