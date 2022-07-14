import numpy as np
import h5py
import cv2
import pickle
import os
import argparse

tile_size = 1000
tile_offset = 500

coord=['N0-60_W000-120','N0-60_W120-240','N0-60_W240-360','S0-60_W000-120','S0-60_W120-240','S0-60_W240-360']
DEM_tiles={'N0-60_W000-120':'sldem2015_256_0n_60n_000_120_float.img',
           'N0-60_W120-240':'sldem2015_256_0n_60n_120_240_float.img',
           'N0-60_W240-360':'sldem2015_256_0n_60n_240_360_float.img',
           'S0-60_W000-120':'sldem2015_256_60s_0s_000_120_float.img',
           'S0-60_W120-240':'sldem2015_256_60s_0s_120_240_float.img',
           'S0-60_W240-360':'sldem2015_256_60s_0s_240_360_float.img'}
ORT_tiles={'N0-60_W000-120':'Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0n_60n_0_120.npy',
           'N0-60_W120-240':'Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0n_60n_120_240.npy',
           'N0-60_W240-360':'Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0n_60n_240_360.npy',
           'S0-60_W000-120':'Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0s_60s_0_120.npy',
           'S0-60_W120-240':'Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0s_60s_120_240.npy',
           'S0-60_W240-360':'Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0s_60s_240_360.npy'}


def loadOrtho(path):
    return np.load(path)

def loadDEM(path):
    return np.fromfile(path,dtype=np.float32).reshape([15360,-1])

def loadPair(path, key, DEMDict, ORTDict):
    ort = loadOrtho(os.path.join(path,ORTDict[key]))
    dem = loadDEM(os.path.join(path,DEMDict[key]))
    h,w = dem.shape
    ort = cv2.resize(ort,(w,h),cv2.INTER_AREA)
    return ort, dem
    
def tilePair(ort, dem, key, h5, dct, tile_size, tile_offset):
    h,w = ort.shape
    htiles = int(h/tile_offset)
    wtiles = int(w/tile_offset)
    wrem = (w - wtiles*tile_offset)
    for i in range(htiles):
        for j in range(wtiles):
            dem_tile = dem[tile_offset*i:tile_offset*i+tile_size,tile_offset*j:tile_offset*j+tile_size]
            dem_tile = (dem_tile - dem_tile.min())/(dem_tile.max() - dem_tile.min())*(2**16)
            dem_tile = dem_tile.astype(np.uint16)
            ort_tile = ort[tile_offset*i:tile_offset*i+tile_size,tile_offset*j:tile_offset*j+tile_size]
            if dem_tile.shape[1] != tile_size:
                break
            if dem_tile.shape[0] != tile_size:
                break
            dem_lbl = key+'-dem-'+str(i*tile_offset)+'-'+str(j*tile_offset)
            ort_lbl = key+'-ort-'+str(i*tile_offset)+'-'+str(j*tile_offset)
            h5[dem_lbl] = dem_tile
            h5[ort_lbl] = ort_tile
            dct[key+'-'+str(i)+'-'+str(j)] = [dem_lbl, ort_lbl]
    print(len(dct.keys()))
    return h5, dct

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str)
    parser.add_argument('--output_path',type=str,default=".")
    return parser.parse_args()

args = parse()

h5 = h5py.File(os.path.join(args.output_path,'MoonORTO2DEM.hdf5'),'w')
dct = {}
for key in coord:
    ort, dem = loadPair(args.data_path, key, DEM_tiles, ORT_tiles)
    h5, dct = tilePair(ort,dem,key,h5,dct,tile_size,tile_offset)
h5.close()
num_samples = len(dct.keys())
keys = list(dct.keys())
idx = np.random.choice(range(num_samples-100),size=50,replace=False)
idx = np.concatenate([idx + i for i in range(20)])  

train_dct = {}
val_dct = {}
for i in range(num_samples):
    if i in idx:
        val_dct[keys[i]] = dct[keys[i]]
    else:
        train_dct[keys[i]] = dct[keys[i]]

with open(os.path.join(args.output_path,'MoonORTO2DEM_train.pkl'),'wb') as f:
    pickle.dump(train_dct, f)

with open(os.path.join(args.output_path,'MoonORTO2DEM_val.pkl'),'wb') as f:
    pickle.dump(val_dct, f)

