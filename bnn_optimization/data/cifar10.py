import tensorflow as tf
from zookeeper import registry


@registry.register_preprocess("cifar10")
def default(image):
    return tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0


@registry.register_preprocess("cifar10")
def resize_and_flip(image, training):
    image = tf.cast(image, tf.float32) / (255.0 / 2.0) - 1.0
    if training:
        image = tf.image.resize_image_with_crop_or_pad(image, 40, 40)
        image = tf.image.random_crop(image, [32, 32, 3])
        image = tf.image.random_flip_left_right(image)
    return image
