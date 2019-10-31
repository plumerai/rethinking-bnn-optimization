from zookeeper import registry
from bnn_optimization.data.problem_definitions import ImageClassification
import tensorflow as tf


@registry.register_preprocess("cifar10")
class resize_and_flip(ImageClassification):
    def inputs(self, data, training):
        image = data["image"]
        if training:
            image = tf.image.resize_with_crop_or_pad(image, 40, 40)
            image = tf.image.random_crop(image, [32, 32, 3])
            image = tf.image.random_flip_left_right(image)

        return tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0
