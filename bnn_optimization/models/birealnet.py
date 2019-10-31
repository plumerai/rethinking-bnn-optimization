from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import optimizers


@registry.register_model
def birealnet(hparams, input_shape, num_classes):
    def residual_block(x, double_filters=False, filters=None):
        assert not (double_filters and filters)

        # compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = filters or in_filters if not double_filters else 2 * in_filters

        shortcut = x
        if in_filters != out_filters:
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
            shortcut = tf.keras.layers.Conv2D(
                out_filters, 1, kernel_initializer="glorot_normal", use_bias=False,
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.8)(shortcut)

        x = lq.layers.QuantConv2D(
            out_filters,
            3,
            strides=1 if out_filters == in_filters else 2,
            padding="same",
            input_quantizer="approx_sign",
            kernel_quantizer=None,
            kernel_initializer="glorot_normal",
            kernel_constraint=None,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        return tf.keras.layers.add([x, shortcut])

    img_input = tf.keras.layers.Input(shape=input_shape)

    # layer 1
    out = tf.keras.layers.Conv2D(
        64,
        7,
        strides=2,
        kernel_initializer="glorot_normal",
        padding="same",
        use_bias=False,
    )(img_input)
    out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
    out = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(out)

    # layer 2
    out = residual_block(out, filters=64)

    # layer 3 - 5
    for _ in range(3):
        out = residual_block(out)

    # layer 6 - 17
    for _ in range(3):
        out = residual_block(out, double_filters=True)
        for _ in range(3):
            out = residual_block(out)

    # layer 18
    out = tf.keras.layers.GlobalAvgPool2D()(out)
    out = tf.keras.layers.Dense(num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out)


@registry.register_hparams(birealnet)
class bop(HParams):
    epochs = 300
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
        return optimizers.Bop(
            tf.keras.optimizers.Adam(lr), threshold=self.threshold, gamma=gamma
        )
