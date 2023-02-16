# This repository aims at teaching GANs to increase the resolution of Lunar DEMs (Digital Elevation Maps) leveraging satellite imagery (WAC from LRO).
| Assembled tiles | Zoom on assembled tiles |
|:-------:|:------:|
![alt text](images/SuperResolutionTile.png) | ![alt text](images/SuperResolutionZOOMonTile.png)

In this repository, we train GANs to generate higher resolution DEMs using WAC images.
These models can then be applied onto NAC ortho-images and their associated DEMs to further improve their resolution.
In the end they generate pixel level details on these images, enabling sub-meter details to be reproduced inside the DEM.
In addition of providing these high resolution DEMs, it also provides an estimate of the GANs' uncertainty.

With this repository you should be able to process large DEMs (15000x70000 pixels DEMs) and get the super-resolution result as 
a geo-tiff.

## Requirements

### Requirements to run our SuperResolution Pipeline

To run the code in this repository you first need to install GDAL. GDAL is required to generate the training dataset.
To install GDAL on Ubuntu run:\
`sudo apt-get install gdal-bin`\
`sudo apt-get install libgdal-dev`\
`version=$(gdal-config --version)`\
`python3 -m pip install gdal==$version`\

Or follow: https://mothergeo-py.readthedocs.io/en/latest/development/how-to/gdal-ubuntu-pkg.html

To train/run the network, you need to install tensorflow, our code is tested against tensorflow 2.4/5. Instruction on how to install tensorflow can be found here:\
https://www.tensorflow.org/install

Alternatively, you can check the conda environment we are using to run our code on the UL HPC.\
To replicate our environment you can run the following commands:\
```bash
conda config --set channel_priority strict
env_name=YOUR_ENVIRONMENT_NAME
conda create --name ${env_name} --file conda-env.txt
conda activate ${env_name}
${HOME}/.conda/envs/${env_name}/bin/pip install -r pip-env.txt
```

Note that it requires installing some dependencies through pip.

### Requirements to run NASA Ames Stereo Pipeline (ASP)

To install ASP you can follow the tutorials here:\
https://stereopipeline.readthedocs.io/en/latest/installation.html#

We would recomend using conda and for that, the easiest is to first follow the procedure available here:\
https://github.com/USGS-Astrogeology/ISIS3#isis-installation-with-conda

However, we would recomend runing `conda install -c usgs-astrogeology isis=7.1.0` instead of `conda install -c usgs-astrogeology isis=7.0.0`.

Do not forget to proceed with the setup of the environment variables: (step 7).

Please make sure you also set-up the ISIS data required to process LRO images:\
https://github.com/USGS-Astrogeology/ISIS3#the-isis-data-area

If you have a lot of storage you can download this data directly on your drive by running:\
```bash
downloadIsisData base $ISISDATA
downloadIsisData lro $ISISDATA
```
If you don't, you can use ISIS SPICE Web Service to fetch only the data you need.

Once ISIS is installed, you can install ASP by following the procedure here:\
https://stereopipeline.readthedocs.io/en/latest/installation.html#precompiled-binaries-linux-and-macos

Don't forget to export ASP's binaries!

For completeness we also share the conda environment used with ISIS/ASP:\
`isis-asp-conda-env.txt`.

## Generating High-Resolution DEMs

To generate high-resolution image, we first need to generate a low resolution DEM and get its associated ortho-image.
To do that we need to select a pair of stereo images. A good place to select such images is on quick-map:\
https://quickmap.lroc.asu.edu\
Quick map is a powerful tool enabling to search for NAC stereo pairs. To ease this process we'd recommend going in the `layers`
tab, and then inside `LROC WAC Basemaps`. Inside this activate only `WAC+NAC+NAC_ROI_MOSAIC`.
Then, navigate to the `Draw/Search Tool`. This tool allows to draw regions in the map allowing you to select regions of interest.
For instance, you could use the `dot` tool, and place a dot on the moon. This will open a tab on the right part of the screen.
This tab will have a foldable selection menu. In that menu pick `NAC`, and you should see the list of NAC images going over the point
you just clicked. You can then `toogle footprints` to see how each of these NACs overlap.
You then want to select two NACs that have strong overlapp and good illumination conditions.
In our pipeline, we use two pairs of NAC to make the generate maps as large as possible.
When we say pair, we mean that we used both the left and right images of the LRO.
Here is a list of pairs we've used: `M104318871_M104311715`, `M1121224102_M1121209902`, `M1137182100_M1137189213`, `M1184050309_M1184064515`, `M1251856688_M1251863722`,
`M172458879_M172452093`, `M190844867_M190852014`, `M1099481313_M1099488460`, `M1127640108_M1173577649`, `M1168293957_M1168308171`, `M1219322049_M1219329084`,
`M1369700963_M1369693934`, `M181058717_M181073012`, `M191181075_M191166775`, `M111571816_M111578606`, `M1136903839_M1136889617`, `M1175701560_M1175694455`,
`M181073012_M181058717`.

### Running ASP

To run ASP you can take inspitation from:\
`run_asp.sh`

In practice, we fist download the EDR images of both pairs, which leaves us with 4 images.
Then we run `lronac2mosaic`, and use it to stich the left and right images together.
After this process we have two `.cub` file.
These files are then used to perform a first coarse stereo reconstruction using `parallel_stereo`.
A DEM is then generated from the coarse stereo reconstruction.
Then, each cub is map-projected onto this DEM.
Each map-projected cub is then used inside a high resolution `parallel_stereo`.
Finally, a DEM is generated along with a matching ortho-image.

### Running the GAN on the ASP generated maps



## Training the GANs
### Downloading and formating the data
First, to download the data, run the following script. These files will take about 18Gb:\
`./get_data.sh` 

The data then needs to be reformated. These files will take an additional 3.6Gb:\
`python3 tile_WAC_MOS.py --data_path data`

The files then need to be formated into an h5 format. This will take an additional 30Gb:\
`python3 make_h5.py --data_path data --output .`

### Train spade/gaugan
```bash
python3 train_spade_256.py --path_h5 MoonORTO2DEM.hdf5 --path_trn MoonORTO2DEM_train.pkl --path_val MoonORTO2DEM_val.pkl --output_path exp_spade
```
or

```bash
python3 train_spade_512.py --path_h5 MoonORTO2DEM.hdf5 --path_trn MoonORTO2DEM_train.pkl --path_val MoonORTO2DEM_val.pkl --output_path exp_spade
```

### Tensorboard visualization:
To visualize the progress of the training you can use the tensorboard frontend using a command similar to this one:\
`tensorboard --logdir exp_spade --bind_all`
The visualization allows to monitor the losses as well as the images generated by the network. Please note that the images are display as `jet` colormaps but the actual predicted images are normalized float32 images. An example of image visualisation can be seen below. On the first row are the targets (GT), on the second row are the input DEMs (GT down sampled by a factor 10), on the third row are the satelite images, and on the last row are the predicted images.
![alt text](images/example.png)

## Aknowledgements:
Our code builds upon the following sources:\
Spade: https://github.com/soumik12345/tf2_gans (Adapted to our problem, fixed encoder not learning, and added a consistency loss).\
Pix2pix: https://www.tensorflow.org/tutorials/generative/pix2pix (Reworked the network, adapted to our problem).

We would also like to thank the folks at NASA Ames, ASU, and the LRO team for their amazing work.
Without them none of this would not be possible, kudos.
