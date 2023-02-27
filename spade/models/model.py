"""References:
- [Hands-On Image Generation with TensorFlow](https://www.packtpub.com/product/hands-on-image-generation-with-tensorflow/9781838826789)
- [Implementing SPADE using fastai](https://towardsdatascience.com/implementing-spade-using-fastai-6ad86b94030a)
- 
"""


import os
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers, models

from .sampling import GaussianSampler
from .networks import build_encoder, build_generator, build_discriminator
from ..losses import (
    generator_loss,
    kl_divergence_loss,
    DiscriminatorLoss,
    FeatureMatchingLoss,
    VGGFeatureMatchingLoss,
    ConsistencyLoss,
    MSE
)

class GauGAN_no_KL(Model):
    def __init__(
        self,
        image_size: int,
        batch_size: int,
        latent_dim:int,
        feature_loss_coeff=10,
        vgg_feature_loss_coeff=0.1,
        consistency_loss_coeff=2,
        upscaling_factor=16,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.target_shape = (image_size, image_size, 1)
        self.source_shape = (image_size, image_size, 2)
        self.feature_loss_coeff = feature_loss_coeff
        self.vgg_feature_loss_coeff = vgg_feature_loss_coeff
        self.consistency_loss_coeff = consistency_loss_coeff
        self.upscaling_factor = upscaling_factor

        self.discriminator = build_discriminator(
            self.source_shape,
            self.target_shape,
            downsample_factor=64,
            alpha=0.2,
            dropout=0.5,
        )
        self.generator = build_generator(
            self.source_shape, latent_dim=self.latent_dim, alpha=0.2
        )
        self.encoder = build_encoder(
            self.source_shape,
            encoder_downsample_factor=64,
            latent_dim=self.latent_dim,
            alpha=0.2,
            dropout=0.5,
        )
        self.sampler = GaussianSampler(batch_size, self.latent_dim)
        self.patch_size, self.combined_model = self.build_combined_generator()

        self.disc_loss_val = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_val = tf.keras.metrics.Mean(name="gen_loss")
        self.feat_loss_val = tf.keras.metrics.Mean(name="feat_loss")
        self.vgg_loss_val = tf.keras.metrics.Mean(name="vgg_loss")
        self.cons_loss_val = tf.keras.metrics.Mean(name="cons_loss")

        self.disc_loss_trn = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_trn = tf.keras.metrics.Mean(name="gen_loss")
        self.feat_loss_trn = tf.keras.metrics.Mean(name="feat_loss")
        self.vgg_loss_trn = tf.keras.metrics.Mean(name="vgg_loss")
        self.cons_loss_trn = tf.keras.metrics.Mean(name="cons_loss")

    @property
    def val_metrics(self):
        return [
            self.disc_loss_val,
            self.gen_loss_val,
            self.feat_loss_val,
            self.vgg_loss_val,
            self.cons_loss_val,
        ]
    
    @property
    def trn_metrics(self):
        return [
            self.disc_loss_trn,
            self.gen_loss_trn,
            self.feat_loss_trn,
            self.vgg_loss_trn,
            self.cons_loss_trn,
        ]

    def build_combined_generator(self):
        # This method builds a model that takes as inputs the following:
        # latent vector, one-hot encoded segmentation label map, and
        # a segmentation map. It then (i) generates an image with the generator,
        # (ii) passes the generated images and segmentation map to the discriminator.
        # Finally, the model produces the following outputs: (a) discriminator outputs,
        # (b) generated image.
        # We will be using this model to simplify the implementation.
        self.discriminator.trainable = False
        source_input = Input(shape=self.source_shape, name="source")
        latent_input = Input(shape=(self.latent_dim), name="latent")
        generated_image = self.generator([latent_input, source_input])
        discriminator_output = self.discriminator([source_input, generated_image])
        patch_size = discriminator_output[-1].shape[1]
        combined_model = Model(
            [latent_input, source_input],
            [discriminator_output, generated_image],
        )
        return patch_size, combined_model

    def compile(self, gen_lr: float = 1e-4, disc_lr: float = 5e-5, **kwargs):
        super().compile(**kwargs)
        self.generator_optimizer = optimizers.Adam(gen_lr, beta_1=0.0, beta_2=0.999)
        self.discriminator_optimizer = optimizers.Adam(
            disc_lr, beta_1=0.0, beta_2=0.999
        )
        self.consistency_loss = ConsistencyLoss(upscaling=self.upscaling_factor)
        self.discriminator_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.vgg_loss = VGGFeatureMatchingLoss()

    def train_discriminator(self, source, target):
        mean, variance = self.encoder(source)
        latent_vector = self.sampler([mean, variance])
        fake_images = self.generator([latent_vector, source])
        with tf.GradientTape() as gradient_tape:
            pred_fake = self.discriminator([source, fake_images])[-1]
            pred_real = self.discriminator([source, target])[-1]
            loss_fake = self.discriminator_loss(pred_fake, False)
            loss_real = self.discriminator_loss(pred_real, True)
            total_loss = 0.5 * (loss_fake + loss_real)

        self.discriminator.trainable = True
        gradients = gradient_tape.gradient(
            total_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        return total_loss

    def train_generator(
        self, source, target
    ):
        # Generator learns through the signal provided by the discriminator. During
        # backpropagation, we only update the generator parameters.
        self.discriminator.trainable = False
        with tf.GradientTape() as tape:
            mean, variance = self.encoder(source)
            latent_vector = self.sampler([mean, variance])
            real_d_output = self.discriminator([source, target])
            fake_d_output, fake_image = self.combined_model(
                [latent_vector, source]
            )
            pred = fake_d_output[-1]

            # Compute generator losses.
            g_loss = generator_loss(pred)
            vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(tf.repeat(target,3,-1), tf.repeat(fake_image,3,-1))
            feature_loss = self.feature_loss_coeff * self.feature_matching_loss(
                real_d_output, fake_d_output
            )
            consistency_loss = self.consistency_loss_coeff * self.consistency_loss(fake_image, target)
            total_loss = g_loss + vgg_loss + feature_loss + consistency_loss

        all_trainable_variables = (
            self.combined_model.trainable_variables + self.encoder.trainable_variables
        )

        gradients = tape.gradient(total_loss, all_trainable_variables,)
        self.generator_optimizer.apply_gradients(
            zip(gradients, all_trainable_variables,)
        )
        return total_loss, feature_loss, vgg_loss, consistency_loss, fake_image

    def train_step(self, source, target):
        discriminator_loss = self.train_discriminator(
            source, target
        )
        (generator_loss, feature_loss, vgg_loss, cons_loss, fake_image) = self.train_generator(
            source, target
        )

        # Report progress.
        self.disc_loss_trn.update_state(discriminator_loss)
        self.gen_loss_trn.update_state(generator_loss)
        self.feat_loss_trn.update_state(feature_loss)
        self.vgg_loss_trn.update_state(vgg_loss)
        self.cons_loss_trn.update_state(cons_loss)
        results = {m.name: m.result() for m in self.trn_metrics}
        return results, fake_image

    def val_step(self, source, target):
        # Obtain the learned moments of the real image distribution.
        mean, variance = self.encoder(source)

        # Sample a latent from the distribution defined by the learned moments.
        latent_vector = self.sampler([mean, variance])

        # Generate the fake images,
        fake_images = self.generator([latent_vector, source])

        # Calculate the losses.
        pred_fake = self.discriminator([source, fake_images])[-1]
        pred_real = self.discriminator([source, target])[-1]
        loss_fake = self.discriminator_loss(pred_fake, False)
        loss_real = self.discriminator_loss(pred_real, True)
        total_discriminator_loss = 0.5 * (loss_fake + loss_real)
        real_d_output = self.discriminator([source, target])
        fake_d_output, fake_image = self.combined_model(
            [latent_vector, source]
        )
        pred = fake_d_output[-1]
        g_loss = generator_loss(pred)
        vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(tf.repeat(target,3,-1), tf.repeat(fake_images,3,-1))
        feature_loss = self.feature_loss_coeff * self.feature_matching_loss(
            real_d_output, fake_d_output
        )
        consistency_loss = self.consistency_loss_coeff * self.consistency_loss(fake_image, target)
        total_generator_loss = g_loss + vgg_loss + feature_loss + consistency_loss

        # Report progress.
        self.disc_loss_val.update_state(total_discriminator_loss)
        self.gen_loss_val.update_state(total_generator_loss)
        self.feat_loss_val.update_state(feature_loss)
        self.vgg_loss_val.update_state(vgg_loss)
        self.cons_loss_val.update_state(consistency_loss)
        results = {m.name: m.result() for m in self.val_metrics}
        return results, fake_images

    def call(self, source):
        mean, variance = self.encoder(source)
        latent_vector = self.sampler([mean, variance])
        return self.generator([latent_vector, source])

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        self.generator.save(
            os.path.join(filepath, "generator"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.discriminator.save(
            os.path.join(filepath, "discriminator"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.encoder.save(
            os.path.join(filepath, "encoder"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

    def load(self, generator_filepath: str, discriminator_filepath: str, encoder_filepath: str):
        self.generator = models.load_model(generator_filepath)
        self.discriminator = models.load_model(discriminator_filepath)
        self.encoder = models.load_model(encoder_filepath)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.generator.save_weights(
            os.path.join(filepath, "generator-checkpoints"),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.discriminator.save_weights(
            os.path.join(filepath, "discriminator-checkpoints"),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.generator.load_weights(
            os.path.join(filepath, "generator-checkpoints"),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
        self.discriminator.load_weights(
            os.path.join(filepath, "discriminator-checkpoints"),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )

class GauGAN(Model):
    def __init__(
        self,
        image_size: int,
        batch_size: int,
        latent_dim:int,
        feature_loss_coeff=10,
        vgg_feature_loss_coeff=0.1,
        kl_divergence_loss_coeff=0.1,
        consistency_loss_coeff=2,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.target_shape = (image_size, image_size, 1)
        self.source_shape = (image_size, image_size, 2)
        self.feature_loss_coeff = feature_loss_coeff
        self.vgg_feature_loss_coeff = vgg_feature_loss_coeff
        self.kl_divergence_loss_coeff = kl_divergence_loss_coeff
        self.consistency_loss_coeff = consistency_loss_coeff

        self.discriminator = build_discriminator(
            self.source_shape,
            self.target_shape,
            downsample_factor=64,
            alpha=0.2,
            dropout=0.5,
        )
        self.generator = build_generator(
            self.source_shape, latent_dim=self.latent_dim, alpha=0.2
        )
        self.encoder = build_encoder(
            self.source_shape,
            encoder_downsample_factor=64,
            latent_dim=self.latent_dim,
            alpha=0.2,
            dropout=0.5,
        )
        self.sampler = GaussianSampler(batch_size, self.latent_dim)
        self.patch_size, self.combined_model = self.build_combined_generator()

        self.disc_loss_val = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_val = tf.keras.metrics.Mean(name="gen_loss")
        self.feat_loss_val = tf.keras.metrics.Mean(name="feat_loss")
        self.vgg_loss_val = tf.keras.metrics.Mean(name="vgg_loss")
        self.kl_loss_val = tf.keras.metrics.Mean(name="kl_loss")
        self.cons_loss_val = tf.keras.metrics.Mean(name="cons_loss")

        self.disc_loss_trn = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_trn = tf.keras.metrics.Mean(name="gen_loss")
        self.feat_loss_trn = tf.keras.metrics.Mean(name="feat_loss")
        self.vgg_loss_trn = tf.keras.metrics.Mean(name="vgg_loss")
        self.kl_loss_trn = tf.keras.metrics.Mean(name="kl_loss")
        self.cons_loss_trn = tf.keras.metrics.Mean(name="cons_loss")

    @property
    def val_metrics(self):
        return [
            self.disc_loss_val,
            self.gen_loss_val,
            self.feat_loss_val,
            self.vgg_loss_val,
            self.kl_loss_val,
            self.cons_loss_val,
        ]
    
    @property
    def trn_metrics(self):
        return [
            self.disc_loss_trn,
            self.gen_loss_trn,
            self.feat_loss_trn,
            self.vgg_loss_trn,
            self.kl_loss_trn,
            self.cons_loss_trn,
        ]

    def build_combined_generator(self):
        # This method builds a model that takes as inputs the following:
        # latent vector, one-hot encoded segmentation label map, and
        # a segmentation map. It then (i) generates an image with the generator,
        # (ii) passes the generated images and segmentation map to the discriminator.
        # Finally, the model produces the following outputs: (a) discriminator outputs,
        # (b) generated image.
        # We will be using this model to simplify the implementation.
        self.discriminator.trainable = False
        source_input = Input(shape=self.source_shape, name="source")
        latent_input = Input(shape=(self.latent_dim), name="latent")
        generated_image = self.generator([latent_input, source_input])
        discriminator_output = self.discriminator([source_input, generated_image])
        patch_size = discriminator_output[-1].shape[1]
        combined_model = Model(
            [latent_input, source_input],
            [discriminator_output, generated_image],
        )
        return patch_size, combined_model

    def compile(self, gen_lr: float = 1e-4, disc_lr: float = 5e-5, **kwargs):
        super().compile(**kwargs)
        self.generator_optimizer = optimizers.Adam(gen_lr, beta_1=0.0, beta_2=0.999)
        self.discriminator_optimizer = optimizers.Adam(
            disc_lr, beta_1=0.0, beta_2=0.999
        )
        self.consistency_loss = ConsistencyLoss()
        self.discriminator_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.vgg_loss = VGGFeatureMatchingLoss()

    def train_discriminator(self, source, target):
        mean, variance = self.encoder(source)
        latent_vector = self.sampler([mean, variance])
        fake_images = self.generator([latent_vector, source])
        with tf.GradientTape() as gradient_tape:
            pred_fake = self.discriminator([source, fake_images])[-1]
            pred_real = self.discriminator([source, target])[-1]
            loss_fake = self.discriminator_loss(pred_fake, False)
            loss_real = self.discriminator_loss(pred_real, True)
            total_loss = 0.5 * (loss_fake + loss_real)

        self.discriminator.trainable = True
        gradients = gradient_tape.gradient(
            total_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        return total_loss

    def train_generator(
        self, source, target
    ):
        # Generator learns through the signal provided by the discriminator. During
        # backpropagation, we only update the generator parameters.
        self.discriminator.trainable = False
        with tf.GradientTape() as tape:
            mean, variance = self.encoder(source)
            latent_vector = self.sampler([mean, variance])
            real_d_output = self.discriminator([source, target])
            fake_d_output, fake_image = self.combined_model(
                [latent_vector, source]
            )
            pred = fake_d_output[-1]

            # Compute generator losses.
            g_loss = generator_loss(pred)
            kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
            vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(tf.repeat(target,3,-1), tf.repeat(fake_image,3,-1))
            feature_loss = self.feature_loss_coeff * self.feature_matching_loss(
                real_d_output, fake_d_output
            )
            consistency_loss = self.consistency_loss_coeff * self.consistency_loss(fake_image, target)
            total_loss = g_loss + kl_loss + vgg_loss + feature_loss + consistency_loss

        all_trainable_variables = (
            self.combined_model.trainable_variables + self.encoder.trainable_variables
        )

        gradients = tape.gradient(total_loss, all_trainable_variables,)
        self.generator_optimizer.apply_gradients(
            zip(gradients, all_trainable_variables,)
        )
        return total_loss, feature_loss, vgg_loss, kl_loss, consistency_loss, fake_image

    def train_step(self, source, target):
        discriminator_loss = self.train_discriminator(
            source, target
        )
        (generator_loss, feature_loss, vgg_loss, kl_loss, cons_loss, fake_image) = self.train_generator(
            source, target
        )

        # Report progress.
        self.disc_loss_trn.update_state(discriminator_loss)
        self.gen_loss_trn.update_state(generator_loss)
        self.feat_loss_trn.update_state(feature_loss)
        self.vgg_loss_trn.update_state(vgg_loss)
        self.kl_loss_trn.update_state(kl_loss)
        self.cons_loss_trn.update_state(cons_loss)
        results = {m.name: m.result() for m in self.trn_metrics}
        return results, fake_image

    def val_step(self, source, target):
        # Obtain the learned moments of the real image distribution.
        mean, variance = self.encoder(source)

        # Sample a latent from the distribution defined by the learned moments.
        latent_vector = self.sampler([mean, variance])

        # Generate the fake images,
        fake_images = self.generator([latent_vector, source])

        # Calculate the losses.
        pred_fake = self.discriminator([source, fake_images])[-1]
        pred_real = self.discriminator([source, target])[-1]
        loss_fake = self.discriminator_loss(pred_fake, False)
        loss_real = self.discriminator_loss(pred_real, True)
        total_discriminator_loss = 0.5 * (loss_fake + loss_real)
        real_d_output = self.discriminator([source, target])
        fake_d_output, fake_image = self.combined_model(
            [latent_vector, source]
        )
        pred = fake_d_output[-1]
        g_loss = generator_loss(pred)
        kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
        vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(tf.repeat(target,3,-1), tf.repeat(fake_images,3,-1))
        feature_loss = self.feature_loss_coeff * self.feature_matching_loss(
            real_d_output, fake_d_output
        )
        consistency_loss = self.consistency_loss_coeff * self.consistency_loss(fake_image, target)
        total_generator_loss = g_loss + kl_loss + vgg_loss + feature_loss + consistency_loss

        # Report progress.
        self.disc_loss_val.update_state(total_discriminator_loss)
        self.gen_loss_val.update_state(total_generator_loss)
        self.feat_loss_val.update_state(feature_loss)
        self.vgg_loss_val.update_state(vgg_loss)
        self.kl_loss_val.update_state(kl_loss)
        self.cons_loss_val.update_state(consistency_loss)
        results = {m.name: m.result() for m in self.val_metrics}
        return results, fake_images

    def call(self, source):
        mean, variance = self.encoder(source)
        latent_vector = self.sampler([mean, variance])
        return self.generator([latent_vector, source])

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        self.generator.save(
            os.path.join(filepath, "generator"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.discriminator.save(
            os.path.join(filepath, "discriminator"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.encoder.save(
            os.path.join(filepath, "encoder"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

    def load(self, generator_filepath: str, discriminator_filepath: str, encoder_filepath: str):
        self.generator = models.load_model(generator_filepath)
        self.discriminator = models.load_model(discriminator_filepath)
        self.encoder = models.load_model(encoder_filepath)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.generator.save_weights(
            os.path.join(filepath, "generator-checkpoints"),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.discriminator.save_weights(
            os.path.join(filepath, "discriminator-checkpoints"),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.generator.load_weights(
            os.path.join(filepath, "generator-checkpoints"),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
        self.discriminator.load_weights(
            os.path.join(filepath, "discriminator-checkpoints"),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )

class CNNSpade(Model):
    def __init__(
        self,
        image_size: int,
        batch_size: int,
        latent_dim:int,
        vgg_feature_loss_coeff=0.1,
        kl_divergence_loss_coeff=0.1,
        consistency_loss_coeff=2,
        mse_loss_coeff=1,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.latent_dim = latent_dim
        self.batch_size = batch_size
        self.target_shape = (image_size, image_size, 1)
        self.source_shape = (image_size, image_size, 2)
        self.vgg_feature_loss_coeff = vgg_feature_loss_coeff
        self.kl_divergence_loss_coeff = kl_divergence_loss_coeff
        self.consistency_loss_coeff = consistency_loss_coeff
        self.mse_loss_coeff = mse_loss_coeff

        self.generator = build_generator(
            self.source_shape, latent_dim=self.latent_dim, alpha=0.2
        )
        self.encoder = build_encoder(
            self.source_shape,
            encoder_downsample_factor=64,
            latent_dim=self.latent_dim,
            alpha=0.2,
            dropout=0.5,
        )
        self.sampler = GaussianSampler(batch_size, self.latent_dim)

        self.total_loss_val = tf.keras.metrics.Mean(name="total_loss")
        self.mse_loss_val = tf.keras.metrics.Mean(name="mse_loss")
        self.vgg_loss_val = tf.keras.metrics.Mean(name="vgg_loss")
        self.kl_loss_val = tf.keras.metrics.Mean(name="kl_loss")
        self.cons_loss_val = tf.keras.metrics.Mean(name="cons_loss")

        self.total_loss_trn = tf.keras.metrics.Mean(name="total_loss")
        self.mse_loss_trn = tf.keras.metrics.Mean(name="mse_loss")
        self.vgg_loss_trn = tf.keras.metrics.Mean(name="vgg_loss")
        self.kl_loss_trn = tf.keras.metrics.Mean(name="kl_loss")
        self.cons_loss_trn = tf.keras.metrics.Mean(name="cons_loss")

    @property
    def val_metrics(self):
        return [
            self.total_loss_val,
            self.mse_loss_val,
            self.vgg_loss_val,
            self.kl_loss_val,
            self.cons_loss_val,
        ]
    
    @property
    def trn_metrics(self):
        return [
            self.total_loss_trn,
            self.mse_loss_trn,
            self.vgg_loss_trn,
            self.kl_loss_trn,
            self.cons_loss_trn,
        ]

    def compile(self, gen_lr: float = 1e-4, disc_lr: float = 4e-4, **kwargs):
        super().compile(**kwargs)
        self.generator_optimizer = optimizers.Adam(gen_lr, beta_1=0.0, beta_2=0.999)
        self.consistency_loss = ConsistencyLoss()
        self.vgg_loss = VGGFeatureMatchingLoss()
        self.mse_loss = MSE()

    def train_generator(
        self, source, target
    ):
        # Generator learns through the signal provided by the discriminator. During
        # backpropagation, we only update the generator parameters.
        with tf.GradientTape() as tape:
            mean, variance = self.encoder(source)
            latent_vector = self.sampler([mean, variance])
            fake_image = self.generator([latent_vector, source])

            # Compute generator losses.
            kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
            mse_loss = self.mse_loss_coeff * self.mse_loss(fake_image, target)
            vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(tf.repeat(target,3,-1), tf.repeat(fake_image,3,-1))
            consistency_loss = self.consistency_loss_coeff * self.consistency_loss(fake_image, target)
            total_loss = kl_loss + vgg_loss + consistency_loss + mse_loss

        all_trainable_variables = (
            self.generator.trainable_variables + self.encoder.trainable_variables
        )

        gradients = tape.gradient(total_loss, all_trainable_variables,)
        self.generator_optimizer.apply_gradients(
            zip(gradients, all_trainable_variables,)
        )
        return total_loss, mse_loss, vgg_loss, kl_loss, consistency_loss, fake_image

    def train_step(self, source, target):
        (total_loss, mse_loss, vgg_loss, kl_loss, cons_loss, fake_image) = self.train_generator(
            source, target
        )

        # Report progress.
        self.total_loss_trn.update_state(total_loss)
        self.vgg_loss_trn.update_state(vgg_loss)
        self.kl_loss_trn.update_state(kl_loss)
        self.cons_loss_trn.update_state(cons_loss)
        self.mse_loss_trn.update_state(mse_loss)
        results = {m.name: m.result() for m in self.trn_metrics}
        return results, fake_image

    def val_step(self, source, target):
        # Obtain the learned moments of the real image distribution.
        mean, variance = self.encoder(source)

        # Sample a latent from the distribution defined by the learned moments.
        latent_vector = self.sampler([mean, variance])

        # Generate the fake images,
        fake_images = self.generator([latent_vector, source])

        # Calculate the losses.
        fake_image = self.generator([latent_vector, source])
        kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
        vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(tf.repeat(target,3,-1), tf.repeat(fake_images,3,-1))
        consistency_loss = self.consistency_loss_coeff * self.consistency_loss(fake_image, target)
        mse_loss = self.mse_loss_coeff * self.mse_loss(fake_image, target)
        total_generator_loss = kl_loss + vgg_loss + consistency_loss + mse_loss

        # Report progress.
        self.total_loss_val.update_state(total_generator_loss)
        self.vgg_loss_val.update_state(vgg_loss)
        self.kl_loss_val.update_state(kl_loss)
        self.cons_loss_val.update_state(consistency_loss)
        self.mse_loss_val.update_state(mse_loss)
        results = {m.name: m.result() for m in self.val_metrics}
        return results, fake_images

    def call(self, source):
        mean, variance = self.encoder(source)
        latent_vector = self.sampler([mean, variance])
        return self.generator([latent_vector, source])

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        self.generator.save(
            os.path.join(filepath, "generator"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.encoder.save(
            os.path.join(filepath, "encoder"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

    def load(self, generator_filepath: str, encoder_filepath: str):
        self.generator = models.load_model(generator_filepath)
        self.encoder = models.load_model(encoder_filepath)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.generator.save_weights(
            os.path.join(filepath, "generator-checkpoints"),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.encoder.save_weights(
            os.path.join(filepath, "encoder-checkpoints"),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.generator.load_weights(
            os.path.join(filepath, "generator-checkpoints"),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
        self.encoder.load_weights(
            os.path.join(filepath, "encoder-checkpoints"),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
