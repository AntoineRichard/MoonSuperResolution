import os
import argparse
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.keras import backend as K

#from spade import GauGAN
from spade.models.model import GauGAN_no_KL
from sampler import Sampler, augmentImage, colorize

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_h5',type=str)
    parser.add_argument('--path_trn',type=str)
    parser.add_argument('--path_val',type=str)
    parser.add_argument('--output_path',type=str,default=".")
    return parser.parse_args()

args = parse()
BATCH_SIZE = 2
EPOCHS = 300

TRN_SIS = Sampler(args.path_h5, args.path_trn, hw=512)
VAL_SIS = Sampler(args.path_h5, args.path_val, hw=512)
max_steps = TRN_SIS.num_samples//BATCH_SIZE
max_steps_prct = max_steps//10
PRINT_STEP = max_steps//10

train_ds = TRN_SIS.getDataset()
val_ds = VAL_SIS.getDataset()

train_ds = train_ds.prefetch(1000)
train_ds = train_ds.map(lambda x, y: augmentImage(x, y), num_parallel_calls=10)
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(BATCH_SIZE)

val_ds = val_ds.prefetch(1000)
val_ds = val_ds.batch(BATCH_SIZE)

gaugan = GauGAN_no_KL(512, BATCH_SIZE, latent_dim=256)
gaugan.compile()

current_time = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
try:
    os.mkdir(os.path.join(args.ouput_path,'models'))
except:
    pass
try:
    os.mkdir(os.path.join(args.output_path,'models',current_time))
except:
    pass

train_log_dir = os.path.join(args.output_path,'tensorboard/' + current_time + '/train')
test_log_dir = os.path.join(args.output_path,'tensorboard/' + current_time + '/test')
train_writer = tf.summary.create_file_writer(train_log_dir)
val_writer = tf.summary.create_file_writer(test_log_dir)

for epoch in range(EPOCHS):
  for step, (x_train, y_train) in enumerate(train_ds):
    if x_train.shape[0] != BATCH_SIZE:
        continue
    metrics, y_ = gaugan.train_step(x_train, y_train)
    if step%PRINT_STEP == 0:
      template = 'Train epoch {} {}%, '+', '.join([metric+': {}'for metric in metrics.keys()])+'.'
      print (template.format(epoch+1,int(100*step/max_steps),*metrics.values()))
      with train_writer.as_default():
        hm_in   = tf.map_fn(lambda img: colorize(img, cmap='jet'), tf.expand_dims(x_train[:,:,:,1],-1))
        hm_pred = tf.map_fn(lambda img: colorize(img, cmap='jet'), y_)
        hm_true = tf.map_fn(lambda img: colorize(img, cmap='jet'), y_train)
        tf.summary.image('GT', hm_true, step=epoch*max_steps+step, max_outputs=3, description=None)
        tf.summary.image('pred', hm_pred, step=epoch*max_steps+step, max_outputs=3, description=None)
        tf.summary.image('input_hmap', hm_in, step=epoch*max_steps+step, max_outputs=3, description=None)
        tf.summary.image('input_image', tf.expand_dims(x_train[:,:,:,0],-1)+0.5, step=epoch*max_steps+step, max_outputs=3, description=None)
        for key in metrics.keys():
            tf.summary.scalar(key, metrics[key], step = epoch*max_steps+step)
        tf.summary.flush()

  for step, (x_train, y_train) in enumerate(val_ds):
    if x_train.shape[0] != BATCH_SIZE:
        continue
    x_val = x_train
    y_val = y_train
    metrics, y_ = gaugan.val_step(x_val, y_val)

  template = 'Valid epoch {}, '+', '.join([metric+': {}'for metric in metrics.keys()])+'.'
  print (template.format(epoch+1,*metrics.values()))
  with val_writer.as_default():
    hm_in   = tf.map_fn(lambda img: colorize(img, cmap='jet'), tf.expand_dims(x_val[:,:,:,1],-1))
    hm_pred = tf.map_fn(lambda img: colorize(img, cmap='jet'), y_)
    hm_true = tf.map_fn(lambda img: colorize(img, cmap='jet'), y_val)
    tf.summary.image('GT', hm_true, step=epoch*max_steps+step, max_outputs=9, description=None)
    tf.summary.image('pred', hm_pred, step=epoch*max_steps+step, max_outputs=9, description=None)
    tf.summary.image('input_hmap', hm_in, step=epoch*max_steps+step, max_outputs=9, description=None)
    tf.summary.image('input_image', tf.expand_dims(x_val[:,:,:,0],-1)+0.5, step=epoch*max_steps+step, max_outputs=9, description=None)
    for key in metrics.keys():
        tf.summary.scalar(key, metrics[key], step = epoch*max_steps+step)
    tf.summary.flush()
  checkpoint_path = os.path.join(args.output_path,'models',current_time,'epoch_'+str(epoch))
  gaugan.save(checkpoint_path)
