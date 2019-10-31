import tensorflow as tf
import larq as lq
from zookeeper import registry, HParams


@registry.register_model
def binary_alexnet(hparams, input_shape, num_classes):

    kwhparams = dict(
        input_quantizer="ste_sign",
        kernel_quantizer=None,
        kernel_constraint=None,
        use_bias=False,
        activation=None,
    )

    def conv_block(x, features, kernel_size, strides=1, pool=False, first_layer=False):
        x = lq.layers.QuantConv2D(
            features,
            kernel_size=kernel_size,
            strides=strides,
            padding="same",
            input_quantizer=None if first_layer else "ste_sign",
            kernel_quantizer=None,
            kernel_constraint=None,
            use_bias=False,
            activation=None,
        )(x)
        if pool:
            x = tf.keras.layers.MaxPool2D(pool_size=3, strides=2)(x)
        x = tf.keras.layers.BatchNormalization(scale=True, momentum=0.9)(x)
        return x

    def dense_block(x, units, final_layer=False):
        if final_layer:
            kwhparams["activation"] = None
        x = lq.layers.QuantDense(units, **kwhparams)(x)
        x = tf.keras.layers.BatchNormalization(scale=True, momentum=0.9)(x)
        return x

    # get input
    img_input = tf.keras.layers.Input(shape=input_shape)

    # feature extractor
    out = conv_block(
        img_input, features=64, kernel_size=11, strides=4, pool=True, first_layer=True
    )
    out = conv_block(out, features=192, kernel_size=5, pool=True)
    out = conv_block(out, features=384, kernel_size=3)
    out = conv_block(out, features=384, kernel_size=3)
    out = conv_block(out, features=256, kernel_size=3, pool=True)

    # classifier
    out = tf.keras.layers.Flatten()(out)
    out = dense_block(out, units=4096)
    out = dense_block(out, units=4096)
    out = dense_block(out, num_classes, final_layer=True)
    out = tf.keras.layers.Activation("softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out)


@registry.register_hparams(binary_alexnet)
class bop(HParams):
    epochs = 150
    batch_size = 1024

    threshold = 1e-8

    gamma_start = 1e-4
    gamma_end = 1e-6

    lr_start = 2.5e-3
    lr_end = 5e-6

    @property
    def optimizer(self):
        decay_step = self.epochs * 1281167 // self.batch_size
        lr = tf.keras.optimizers.schedules.PolynomialDecay(
            self.lr_start, decay_step, end_learning_rate=self.lr_end, power=1.0
        )
        gamma = tf.keras.optimizers.schedules.PolynomialDecay(
            self.gamma_start, decay_step, end_learning_rate=self.gamma_end, power=1.0
        )
        return lq.optimizers.Bop(
            tf.keras.optimizers.Adam(lr), threshold=self.threshold, gamma=gamma
        )
