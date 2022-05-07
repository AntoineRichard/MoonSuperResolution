import os
import argparse
import numpy as np
import tensorflow as tf
import datetime
from tensorflow.python.keras import backend as K

from pix2pix import Pix2Pix
from sampler import Sampler, augmentImage, colorize

def parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--path_h5',type=str)
    parser.add_argument('--path_trn',type=str)
    parser.add_argument('--path_val',type=str)
    parser.add_argument('--output_path',type=str,default=".")
    return parser.parse_args()

args = parse()

TRN_SIS = Sampler(args.path_h5, args.path_trn)
VAL_SIS = Sampler(args.path_h5, args.path_val)

batch_size = 64
max_steps = TRN_SIS.num_samples//batch_size
max_steps_prct = max_steps//10
EPOCHS = 300

train_ds = TRN_SIS.getDataset()
val_ds = VAL_SIS.getDataset()

train_ds = train_ds.prefetch(1000)
train_ds = train_ds.map(lambda x, y: augmentImage(x, y), num_parallel_calls=10)
train_ds = train_ds.shuffle(1000)
train_ds = train_ds.batch(64)

val_ds = val_ds.prefetch(1000)
val_ds = val_ds.batch(64)

EPOCHS = 50
LAMBDA = 100
MAX_TR_OUTPUTS = 3
MAX_VL_OUTPUTS = 9

BATCH_SIZE = 64
max_steps = TRN_SIS.num_samples//BATCH_SIZE
PRINT_STEP = max_steps//10
EPOCHS = 300

pix2pix = Pix2Pix()

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
    y_, gloss, dloss = pix2pix.train_step(x_train, y_train, epoch)
    if step%PRINT_STEP == 0:
      template = 'Epoch {} {}%, G-Loss: {}, D-Loss: {}'
      print (template.format(epoch+1,int(100*step/max_steps),gloss, dloss))
      with train_writer.as_default():
        hm_in   = tf.map_fn(lambda img: colorize(img, cmap='jet'), tf.expand_dims(x_train[:,:,:,1],-1))
        hm_pred = tf.map_fn(lambda img: colorize(img, cmap='jet'), y_)
        hm_true = tf.map_fn(lambda img: colorize(img, cmap='jet'), y_train)
        tf.summary.image('GT', hm_true, step=epoch*max_steps+step, max_outputs=3, description=None)
        tf.summary.image('pred', hm_pred, step=epoch*max_steps+step, max_outputs=3, description=None)
        tf.summary.image('input_hmap', hm_in, step=epoch*max_steps+step, max_outputs=3, description=None)
        tf.summary.image('input_image', tf.expand_dims(x_train[:,:,:,0],-1)+0.5, step=epoch*max_steps+step, max_outputs=3, description=None)
        tf.summary.scalar('generator loss', gloss, step = epoch*max_steps+step)
        tf.summary.scalar('discriminator loss', dloss, step = epoch*max_steps+step)
        tf.summary.flush()

  for step, (x_train, y_train) in enumerate(val_ds):
    y_, vgloss, vdloss = pix2pix.val_step(x_train, y_train)

  template = 'Epoch {}, G-Loss: {}, D-Loss: {}, val G-Loss: {}, val D-Loss: {}'
  print (template.format(epoch+1, gloss, dloss, vgloss, vdloss))
  with val_writer.as_default():
    hm_in   = tf.map_fn(lambda img: colorize(img, cmap='jet'), tf.expand_dims(x_train[:,:,:,1],-1))
    hm_pred = tf.map_fn(lambda img: colorize(img, cmap='jet'), y_)
    hm_true = tf.map_fn(lambda img: colorize(img, cmap='jet'), y_train)
    tf.summary.image('GT', hm_true, step=epoch*max_steps+step, max_outputs=3, description=None)
    tf.summary.image('pred', hm_pred, step=epoch*max_steps+step, max_outputs=3, description=None)
    tf.summary.image('input_hmap', hm_in, step=epoch*max_steps+step, max_outputs=3, description=None)
    tf.summary.image('input_image', tf.expand_dims(x_train[:,:,:,0],-1)+0.5, step=epoch*max_steps+step, max_outputs=3, description=None)
    tf.summary.scalar('generator loss', vgloss, step = epoch*max_steps+step)
    tf.summary.scalar('discriminator loss', vdloss, step = epoch*max_steps+step)
    tf.summary.flush()
  checkpoint_g_path = os.path.join(args.output_path,'models',current_time,'Generator/epoch_'+str(epoch))
  checkpoint_d_path = os.path.join(args.output_path,'models',current_time,'Discriminator/epoch_'+str(epoch))
  pix2pix.generator.save(checkpoint_g_path)
  pix2pix.discriminator.save(checkpoint_d_path)
