import numpy as np
import cv2
from matplotlib import pyplot as plt

MOS = np.load('Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0n_60n_240_360.npy')
print(MOS.shape)
DEM = np.fromfile('sldem2015_256_0n_60n_240_360_float.img',dtype=np.float32)
DEM = DEM.reshape([15360,-1])
print(DEM.shape)
print(np.array(MOS.shape)/np.array(DEM.shape))
H,W = DEM.shape
MOS = cv2.resize(MOS,(W,H),cv2.INTER_AREA)

def getPatch(DEM,MOS,px,py,w=256,h=256):
    DEMp = DEM[px:px+w,py:py+h]
    MOSp = MOS[px:px+w,py:py+h]
    DEMpn = (DEMp - DEMp.min()) / (DEMp.max() - DEMp.min())
    MOSpn = (MOSp - MOSp.min()) / (MOSp.max() - MOSp.min())
    DEMpnu = (DEMpn*255).astype(np.uint8)
    MOSpnu = (MOSpn*255).astype(np.uint8)
    DEMpnucm = np.flip(cv2.applyColorMap(DEMpnu,cv2.COLORMAP_JET),-1)
    MOSpnucm = np.repeat(np.expand_dims(MOSpnu,-1),3,-1)
    return DEMpnucm, MOSpnucm

DEMp0, MOSp0 = getPatch(DEM,MOS,0,0)
DEMp1, MOSp1 = getPatch(DEM,MOS,15000,0)
DEMp2, MOSp2 = getPatch(DEM,MOS,0,30000)
DEMp3, MOSp3 = getPatch(DEM,MOS,15000,30000)
fig, axs = plt.subplots(2, 4, sharex=True)
axs[0][0].imshow(DEMp0)
axs[0][0].xaxis.set_ticks(np.arange(0, 256, 32))
axs[0][0].yaxis.set_ticks(np.arange(0, 256, 32))
axs[0][0].grid(color='w', linestyle='-', linewidth=1)
axs[0][1].imshow(MOSp0)
axs[0][1].xaxis.set_ticks(np.arange(0, 256, 32))
axs[0][1].yaxis.set_ticks(np.arange(0, 256, 32))
axs[0][1].grid(color='w', linestyle='-', linewidth=1)
axs[1][0].imshow(DEMp1)
axs[1][0].xaxis.set_ticks(np.arange(0, 256, 32))
axs[1][0].yaxis.set_ticks(np.arange(0, 256, 32))
axs[1][0].grid(color='w', linestyle='-', linewidth=1)
axs[1][1].imshow(MOSp1)
axs[1][1].xaxis.set_ticks(np.arange(0, 256, 32))
axs[1][1].yaxis.set_ticks(np.arange(0, 256, 32))
axs[1][1].grid(color='w', linestyle='-', linewidth=1)
axs[0][2].imshow(DEMp2)
axs[0][2].xaxis.set_ticks(np.arange(0, 256, 32))
axs[0][2].yaxis.set_ticks(np.arange(0, 256, 32))
axs[0][2].grid(color='w', linestyle='-', linewidth=1)
axs[0][3].imshow(MOSp2)
axs[0][3].xaxis.set_ticks(np.arange(0, 256, 32))
axs[0][3].yaxis.set_ticks(np.arange(0, 256, 32))
axs[0][3].grid(color='w', linestyle='-', linewidth=1)
axs[1][2].imshow(DEMp3)
axs[1][2].xaxis.set_ticks(np.arange(0, 256, 32))
axs[1][2].yaxis.set_ticks(np.arange(0, 256, 32))
axs[1][2].grid(color='w', linestyle='-', linewidth=1)
axs[1][3].imshow(MOSp3)
axs[1][3].xaxis.set_ticks(np.arange(0, 256, 32))
axs[1][3].yaxis.set_ticks(np.arange(0, 256, 32))
axs[1][3].grid(color='w', linestyle='-', linewidth=1)
plt.show()

    
