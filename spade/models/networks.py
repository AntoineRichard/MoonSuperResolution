import tensorflow as tf
from tensorflow.keras import layers
from tensorflow.keras import Input, Model

from .blocks import downsample_block, ResidualBlock


def build_encoder(
    image_shape,
    encoder_downsample_factor: int,
    latent_dim: int,
    alpha: float,
    dropout: float,
) -> Model:
    input_image = Input(shape=image_shape)
    x = downsample_block(
        encoder_downsample_factor, 3, apply_norm=False, alpha=alpha, dropout=dropout
    )(input_image)
    x = downsample_block(
        2 * encoder_downsample_factor, 3, alpha=alpha, dropout=dropout
    )(x)
    x = downsample_block(
        4 * encoder_downsample_factor, 3, alpha=alpha, dropout=dropout
    )(x)
    x = downsample_block(
        8 * encoder_downsample_factor, 3, alpha=alpha, dropout=dropout
    )(x)
    x = downsample_block(
        8 * encoder_downsample_factor, 3, alpha=alpha, dropout=dropout
    )(x)
    x = layers.Flatten()(x)
    mean = layers.Dense(latent_dim, name="mean")(x)
    variance = layers.Dense(latent_dim, name="variance")(x)
    return Model(input_image, [mean, variance], name="encoder")


def build_generator(source_shape, latent_dim: int, alpha: float) -> Model:
    latent = Input(shape=(latent_dim))
    source = Input(shape=source_shape)
    sw = source_shape[0]//2**6
    x = layers.Dense(16*sw*sw*64)(latent)
    x = layers.Reshape((sw, sw, 1024))(x)
    x = ResidualBlock(filters=1024, alpha=alpha)(x, source)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(filters=1024, alpha=alpha)(x, source)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(filters=1024, alpha=alpha)(x, source)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(filters=512, alpha=alpha)(x, source)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(filters=256, alpha=alpha)(x, source)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(filters=128, alpha=alpha)(x, source)
    x = layers.UpSampling2D((2, 2))(x)
    x = tf.nn.leaky_relu(x, 0.2)
    output_image = layers.Conv2D(1, 4, padding="same")(x)
    return Model([latent, source], output_image, name="generator")


def build_discriminator(
    source_shape, target_shape, downsample_factor: int, alpha: float, dropout: float
) -> Model:
    input_image_A = Input(shape=source_shape, name="discriminator_source")
    input_image_B = Input(shape=target_shape, name="discriminator_target")
    x = layers.Concatenate()([input_image_A, input_image_B])
    x1 = downsample_block(
        downsample_factor, 4, apply_norm=False, alpha=alpha, dropout=dropout
    )(x)
    x2 = downsample_block(2 * downsample_factor, 4, alpha=alpha, dropout=dropout)(x1)
    x3 = downsample_block(4 * downsample_factor, 4, alpha=alpha, dropout=dropout)(x2)
    x4 = downsample_block(
        8 * downsample_factor, 4, strides=1, alpha=alpha, dropout=dropout
    )(x3)
    x5 = layers.Conv2D(1, 4)(x4)
    outputs = [x1, x2, x3, x4, x5]
    return Model([input_image_A, input_image_B], outputs)
