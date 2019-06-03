from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import utils


@registry.register_model
def birealnet(args, dataset, input_tensor=None, include_top=True):
    def residual_block(x, double_filters=False, filters=None):
        assert not (double_filters and filters)

        # compute dimensions
        in_filters = x.get_shape().as_list()[-1]
        out_filters = filters or in_filters if not double_filters else 2 * in_filters

        shortcut = x
        if in_filters != out_filters:
            shortcut = tf.keras.layers.AvgPool2D(2, strides=2, padding="same")(shortcut)
            shortcut = tf.keras.layers.Conv2D(
                out_filters,
                1,
                kernel_initializer=args.kernel_initializer,
                use_bias=False,
            )(shortcut)
            shortcut = tf.keras.layers.BatchNormalization(momentum=0.8)(shortcut)

        x = lq.layers.QuantConv2D(
            out_filters,
            3,
            strides=1 if out_filters == in_filters else 2,
            padding="same",
            input_quantizer=args.input_quantizer,
            kernel_quantizer=args.kernel_quantizer,
            kernel_initializer=args.kernel_initializer,
            kernel_constraint=args.kernel_constraint,
            use_bias=False,
        )(x)
        x = tf.keras.layers.BatchNormalization(momentum=0.8)(x)
        return tf.keras.layers.add([x, shortcut])

    img_input = tf.keras.layers.Input(shape=dataset.input_shape)

    # layer 1
    out = tf.keras.layers.Conv2D(
        args.filters,
        7,
        strides=2,
        kernel_initializer=args.kernel_initializer,
        padding="same",
        use_bias=False,
    )(img_input)
    out = tf.keras.layers.BatchNormalization(momentum=0.8)(out)
    out = tf.keras.layers.MaxPool2D(3, strides=2, padding="same")(out)

    # layer 2 - 5
    out = residual_block(out, filters=args.filters)
    for _ in range(1, 5):
        out = residual_block(out)

    # layer 6 - 17
    for i in range(1, 4):
        out = residual_block(out, double_filters=True)
        for _ in range(1, 4):
            out = residual_block(out)

    # layer 18
    if include_top:
        out = tf.keras.layers.GlobalAvgPool2D()(out)
        out = tf.keras.layers.Dense(dataset.num_classes, activation="softmax")(out)

    return tf.keras.Model(inputs=img_input, outputs=out)


@registry.register_hparams(birealnet)
class default(HParams):
    filters = 64
    learning_rate = 5e-3
    decay_schedule = "linear"
    batch_size = 512
    input_quantizer = "approx_sign"
    kernel_quantizer = "magnitude_aware_sign"
    kernel_constraint = "weight_clip"
    kernel_initializer = "glorot_normal"

    @property
    def optimizer(self):
        if self.decay_schedule == "linear_cosine":
            lr = tf.keras.experimental.LinearCosineDecay(self.learning_rate, 750684)
        elif self.decay_schedule == "linear":
            lr = tf.keras.optimizers.schedules.PolynomialDecay(
                self.learning_rate, 750684, end_learning_rate=0, power=1.0
            )
        else:
            lr = self.learning_rate
        return tf.keras.optimizers.Adam(lr)
