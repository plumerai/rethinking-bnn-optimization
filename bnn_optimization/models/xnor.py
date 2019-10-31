from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import optimizers


@registry.register_model
def xnornet(hparams, input_shape, num_classes):

    kwargs = dict(
        kernel_quantizer=None,
        input_quantizer="ste_sign",
        kernel_constraint=None,
        use_bias=False,
    )
    img_input = tf.keras.layers.Input(shape=input_shape)

    x = tf.keras.layers.Conv2D(
        96,
        (11, 11),
        strides=(4, 4),
        padding="same",
        use_bias=False,
        input_shape=input_shape,
        kernel_regularizer=hparams.kernel_regularizer,
    )(img_input)

    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-5)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = lq.layers.QuantConv2D(256, (5, 5), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = lq.layers.QuantConv2D(384, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = lq.layers.QuantConv2D(256, (3, 3), padding="same", **kwargs)(x)
    x = tf.keras.layers.MaxPool2D(pool_size=(3, 3), strides=(2, 2))(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)
    x = lq.layers.QuantConv2D(4096, (6, 6), padding="valid", **kwargs)(x)
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-4)(x)

    # Equivalent to a dense layer
    x = lq.layers.QuantConv2D(4096, (1, 1), strides=(1, 1), padding="valid", **kwargs)(
        x
    )
    x = tf.keras.layers.BatchNormalization(momentum=0.9, scale=True, epsilon=1e-3)(x)
    x = tf.keras.layers.Activation("relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(
        num_classes, use_bias=False, kernel_regularizer=hparams.kernel_regularizer
    )(x)
    x = tf.keras.layers.Activation("softmax")(x)

    return tf.keras.models.Model(img_input, x)


@registry.register_hparams(xnornet)
class bop(HParams):
    epochs = 100
    batch_size = 1024

    threshold = 1e-8

    gamma_start = 1e-4
    gamma_end = 1e-6

    lr_start = 2.5e-3
    lr_end = 5e-6

    regularization_quantity = 5e-7

    @property
    def kernel_regularizer(self):
        return tf.keras.regularizers.l2(self.regularization_quantity)

    @property
    def optimizer(self):
        decay_step = self.epochs * 1281167 // self.batch_size
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_step, end_learning_rate=self.lr_end, power=1.0
        )
        gamma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.gamma_start, decay_step, end_learning_rate=self.gamma_end, power=1.0
        )
        return optimizers.Bop(
            tf.keras.optimizers.Adam(lr), threshold=self.threshold, gamma=gamma
        )
