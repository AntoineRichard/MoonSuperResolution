import tensorflow as tf
from tensorflow.keras import losses, applications, Model


def generator_loss(y):
    return -tf.reduce_mean(y)

def kl_divergence_loss(mean, variance):
    return -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance))

def gradient_loss(y_true, y_pred):
    gx_true, gy_true = tf.image.image_gradients(y_true)
    gx_pred, gy_pred = tf.image.image_gradients(y_pred)
    return tf.reduce_mean(tf.abs(gx_true - gx_pred) + tf.abs(gy_true - gy_pred))

def normal_loss(y_true, y_pred):
    gx_true, gy_true = tf.image.image_gradients(y_true)
    gx_pred, gy_pred = tf.image.image_gradients(y_pred)
    one = tf.ones_like(gx_true)
    n_true = tf.concat([-gx_true, -gy_true, one],axis=-1)
    n_pred = tf.concat([-gx_pred, -gy_pred, one],axis=-1)
    normal_error = (tf.reduce_sum(tf.multiply(n_pred, n_true),-1)) / (tf.sqrt(tf.reduce_sum(tf.multiply(n_true, n_true),-1))*tf.sqrt(tf.reduce_sum(tf.multiply(n_pred, n_pred),-1)))
    return tf.reduce_mean(1 - normal_error)

class ConsistencyLoss(losses.Loss):
    def __init__(self, upscaling=10, **kwargs):
        super().__init__(**kwargs)
        self.mse = losses.MeanSquaredError()
        # Using an average pool to downscale the image. Not sure if tf.image allows for gradient propagation.
        self.pool = tf.keras.layers.AveragePooling2D(pool_size=(upscaling, upscaling),strides=upscaling,padding='valid')

    def call(self, y_true, y_pred):
        return self.mse(self.pool(y_true), self.pool(y_pred))

class MSE(losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mse = losses.MeanSquaredError()

    def call(self, y_true, y_pred):
        return self.mse(y_true, y_pred)


class FeatureMatchingLoss(losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.mae = losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        loss = 0
        for i in range(len(y_true) - 1):
            loss += self.mae(y_true[i], y_pred[i])
        return loss


class VGGFeatureMatchingLoss(losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.encoder_layers = [
            "block1_conv1",
            "block2_conv1",
            "block3_conv1",
            "block4_conv1",
            "block5_conv1",
        ]
        self.weights = [1.0 / 32, 1.0 / 16, 1.0 / 8, 1.0 / 4, 1.0]
        vgg = applications.VGG19(include_top=False, weights="imagenet")
        layer_outputs = [vgg.get_layer(x).output for x in self.encoder_layers]
        self.vgg_model = Model(vgg.input, layer_outputs, name="VGG")
        self.mae = losses.MeanAbsoluteError()

    def call(self, y_true, y_pred):
        y_true = applications.vgg19.preprocess_input(127.5 * (y_true + 1))
        y_pred = applications.vgg19.preprocess_input(127.5 * (y_pred + 1))
        real_features = self.vgg_model(y_true)
        fake_features = self.vgg_model(y_pred)
        loss = 0
        for i in range(len(real_features)):
            loss += self.weights[i] * self.mae(real_features[i], fake_features[i])
        return loss


class DiscriminatorLoss(losses.Loss):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.hinge_loss = losses.Hinge()

    def call(self, y, is_real):
        label = 1.0 if is_real else -1.0
        return self.hinge_loss(label, y)
