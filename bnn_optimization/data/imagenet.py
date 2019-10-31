"""ImageNet preprocessing.

Code adapted from https://github.com/tensorflow/tpu/blob/master/models/official/efficientnet/preprocessing.py
"""
from zookeeper import registry
from bnn_optimization.data.problem_definitions import ImageClassification
import tensorflow_datasets as tfds
import tensorflow as tf

IMAGE_SIZE = 224
CROP_PADDING = 32

MEAN_RGB = [0.485 * 255, 0.456 * 255, 0.406 * 255]
STDDEV_RGB = [0.229 * 255, 0.224 * 255, 0.225 * 255]


def distorted_bounding_box_crop(
    image_bytes,
    bbox,
    min_object_covered=0.1,
    aspect_ratio_range=(0.75, 1.33),
    area_range=(0.05, 1.0),
    max_attempts=100,
    scope=None,
):
    """Generates cropped_image using one of the bboxes randomly distorted.

    See `tf.image.sample_distorted_bounding_box` for more documentation.

    Args:
      image_bytes: `Tensor` of binary image data.
      bbox: `Tensor` of bounding boxes arranged `[1, num_boxes, coords]`
          where each coordinate is [0, 1) and the coordinates are arranged
          as `[ymin, xmin, ymax, xmax]`. If num_boxes is 0 then use the whole
          image.
      min_object_covered: An optional `float`. Defaults to `0.1`. The cropped
          area of the image must contain at least this fraction of any bounding
          box supplied.
      aspect_ratio_range: An optional list of `float`s. The cropped area of the
          image must have an aspect ratio = width / height within this range.
      area_range: An optional list of `float`s. The cropped area of the image
          must contain a fraction of the supplied image within in this range.
      max_attempts: An optional `int`. Number of attempts at generating a cropped
          region of the image of the specified constraints. After `max_attempts`
          failures, return the entire image.
      scope: Optional `str` for name scope.
    Returns:
      cropped image `Tensor`
    """
    with tf.name_scope(scope or "distorted_bounding_box_crop"):
        shape = tf.image.extract_jpeg_shape(image_bytes)
        sample_distorted_bounding_box = tf.image.sample_distorted_bounding_box(
            shape,
            bounding_boxes=bbox,
            min_object_covered=min_object_covered,
            aspect_ratio_range=aspect_ratio_range,
            area_range=area_range,
            max_attempts=max_attempts,
            use_image_if_no_bounding_boxes=True,
        )
        bbox_begin, bbox_size, _ = sample_distorted_bounding_box

        # Crop the image to the specified bounding box.
        offset_y, offset_x, _ = tf.unstack(bbox_begin)
        target_height, target_width, _ = tf.unstack(bbox_size)
        crop_window = tf.stack([offset_y, offset_x, target_height, target_width])
        image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)

        return image


def _at_least_x_are_equal(a, b, x):
    """At least `x` of `a` and `b` `Tensors` are equal."""
    match = tf.equal(a, b)
    match = tf.cast(match, tf.int32)
    return tf.greater_equal(tf.reduce_sum(match), x)


def _decode_and_random_crop(image_bytes, image_size):
    """Make a random crop of image_size."""
    bbox = tf.constant([0.0, 0.0, 1.0, 1.0], dtype=tf.float32, shape=[1, 1, 4])
    image = distorted_bounding_box_crop(
        image_bytes,
        bbox,
        min_object_covered=0.1,
        aspect_ratio_range=(3.0 / 4, 4.0 / 3.0),
        area_range=(0.08, 1.0),
        max_attempts=10,
        scope=None,
    )
    original_shape = tf.image.extract_jpeg_shape(image_bytes)
    bad = _at_least_x_are_equal(original_shape, tf.shape(image), 3)

    image = tf.cond(
        bad,
        lambda: _decode_and_center_crop(image_bytes, image_size),
        lambda: tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0],
    )

    return image


def _decode_and_center_crop(image_bytes, image_size):
    """Crops to center of image with padding then scales image_size."""
    shape = tf.image.extract_jpeg_shape(image_bytes)
    image_height = shape[0]
    image_width = shape[1]

    padded_center_crop_size = tf.cast(
        (
            (image_size / (image_size + CROP_PADDING))
            * tf.cast(tf.minimum(image_height, image_width), tf.float32)
        ),
        tf.int32,
    )

    offset_height = ((image_height - padded_center_crop_size) + 1) // 2
    offset_width = ((image_width - padded_center_crop_size) + 1) // 2
    crop_window = tf.stack(
        [offset_height, offset_width, padded_center_crop_size, padded_center_crop_size]
    )
    image = tf.image.decode_and_crop_jpeg(image_bytes, crop_window, channels=3)
    image = tf.compat.v1.image.resize_bicubic([image], [image_size, image_size])[0]
    return image


def _normalize(image, mean_rgb=MEAN_RGB, stddev_rgb=STDDEV_RGB):
    """Normalizes images to variance 1 and mean 0 over the whole dataset"""
    # TODO: Evaluate if it makes sense to use this as a first layer
    # and do the computation on the GPU instead

    image -= tf.broadcast_to(mean_rgb, tf.shape(image))
    image /= tf.broadcast_to(stddev_rgb, tf.shape(image))

    return image


def preprocess_image(
    image_bytes, is_training=False, use_bfloat16=False, image_size=IMAGE_SIZE,
):
    """Preprocesses the given image.

    Args:
      image_bytes: `Tensor` representing an image binary of arbitrary size.
      is_training: `bool` for whether the preprocessing is for training.
      use_bfloat16: `bool` for whether to use bfloat16.
      image_size: image size.

    Returns:
      A preprocessed and normalized image `Tensor`.
    """
    if is_training:
        image = _decode_and_random_crop(image_bytes, image_size)
        image = tf.image.random_flip_left_right(image)
    else:
        image = _decode_and_center_crop(image_bytes, image_size)

    image = tf.reshape(image, [image_size, image_size, 3])
    image = tf.cast(image, dtype=tf.bfloat16 if use_bfloat16 else tf.float32)
    image = _normalize(image, mean_rgb=MEAN_RGB, stddev_rgb=STDDEV_RGB)

    return image


class ImageNetClassification(ImageClassification):
    @property
    def kwargs(self):
        return {**super().kwargs, "input_shape": (224, 224, 3)}


@registry.register_preprocess("imagenet2012")
class default(ImageNetClassification):
    decoders = {"image": tfds.decode.SkipDecoding()}

    def inputs(self, data, training):
        return preprocess_image(data["image"], is_training=training, image_size=224,)
