import gdal
import numpy as np
def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path',type=str)
    return parser.parse_args()

args = parse()

ds = gdal.Open(os.path.join(args.data_path,"Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif"))
X = np.array(ds.GetRasterBand(1).ReadAsArray())
print(X.shape)
print(X.dtype)
H,W = X.shape
X = np.concatenate([X[:,int(W/2):],X[:,:int(W/2)]],axis=1)
X = X[int(H/6):-int(H/6)]
H,W = X.shape
np.save(os.path.join(args.data_path,"Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0n_60n_0_120.npy"),X[:int(H/2),:int(W/3)])
np.save(os.path.join(args.data_path,"Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0n_60n_120_240.npy"),X[:int(H/2),int(W/3):int(W/3)*2])
np.save(os.path.join(args.data_path,"Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0n_60n_240_360.npy"),X[:int(H/2),int(W/3)*2:])
np.save(os.path.join(args.data_path,"Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0s_60s_0_120.npy"),X[int(H/2):,:int(W/3)])
np.save(os.path.join(args.data_path,"Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0s_60s_120_240.npy"),X[int(H/2):,int(W/3):int(W/3)*2])
np.save(os.path.join(args.data_path,"Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013_0s_60s_240_360.npy"),X[int(H/2):,int(W/3)*2:])
