from zookeeper import registry, HParams
import larq as lq
import tensorflow as tf
from bnn_optimization import utils


@registry.register_model
def binarynet(hparams, dataset):
    kwhparams = dict(
        input_quantizer="ste_sign",
        kernel_quantizer="ste_sign",
        kernel_constraint="weight_clip",
        use_bias=False,
    )
    return tf.keras.models.Sequential(
        [
            # don't quantize inputs in first layer
            lq.layers.QuantConv2D(
                hparams.filters,
                hparams.kernel_size,
                kernel_quantizer="ste_sign",
                kernel_constraint="weight_clip",
                use_bias=False,
                input_shape=dataset.input_shape,
            ),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(
                hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(
                2 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(
                2 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(
                4 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantConv2D(
                4 * hparams.filters, hparams.kernel_size, padding="same", **kwhparams
            ),
            tf.keras.layers.MaxPool2D(pool_size=(2, 2), strides=(2, 2)),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.Flatten(),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantDense(hparams.dense_units, **kwhparams),
            tf.keras.layers.BatchNormalization(scale=False),
            lq.layers.QuantDense(dataset.num_classes, **kwhparams),
            tf.keras.layers.BatchNormalization(scale=False),
            tf.keras.layers.Activation("softmax"),
        ]
    )


@registry.register_hparams(binarynet)
class default(HParams):
    filters = 64
    dense_units = 4096
    kernel_size = 3
    batch_size = 256
    optimizer = tf.keras.optimizers.Adam(5e-3)
