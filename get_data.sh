#!/bin/sh
mkdir data
cd data
wget https://planetarymaps.usgs.gov/mosaic/Lunar_LRO_LROC-WAC_Mosaic_global_100m_June2013.tif
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/sldem2015_256_0n_60n_000_120_float.img
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/sldem2015_256_0n_60n_120_240_float.img
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/sldem2015_256_0n_60n_240_360_float.img
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/sldem2015_256_60s_0s_000_120_float.img
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/sldem2015_256_60s_0s_120_240_float.img
wget https://pds-geosciences.wustl.edu/lro/lro-l-lola-3-rdr-v1/lrolol_1xxx/data/sldem2015/tiles/float_img/sldem2015_256_60s_0s_240_360_float.img
cd ..
