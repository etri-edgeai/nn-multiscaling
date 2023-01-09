""" BespokeLoss """

import tensorflow as tf

from tensorflow.keras import losses
from tensorflow.keras import backend

class BespokeTaskLoss(losses.Loss):
    """BespokeTaskLoss for block-level transfer learning.

    """

    def __init__(self, label_smoothing=0.1):
        super(BespokeTaskLoss, self).__init__()
        self.mute = False
        self.label_smoothing = label_smoothing

    def call(self, y_true, y_pred):
        """ Call function imple.

        """
        if self.mute:
            return 0.0
        else:
            return losses.categorical_crossentropy(y_true, y_pred, label_smoothing=self.label_smoothing)

def accuracy(y_true, y_pred):
  """Calculates how often predictions match one-hot labels.

  Standalone usage:
  >>> y_true = [[0, 0, 1], [0, 1, 0]]
  >>> y_pred = [[0.1, 0.9, 0.8], [0.05, 0.95, 0]]
  >>> m = tf.keras.metrics.categorical_accuracy(y_true, y_pred)
  >>> assert m.shape == (2,)
  >>> m.numpy()
  array([0., 1.], dtype=float32)

  You can provide logits of classes as `y_pred`, since argmax of
  logits and probabilities are same.

  Args:
    y_true: One-hot ground truth values.
    y_pred: The prediction values.

  Returns:
    Categorical accuracy values.
  """
  return tf.cast(
      tf.equal(
          tf.compat.v1.argmax(y_true, axis=-1), tf.compat.v1.argmax(y_pred, axis=-1)),
      backend.floatx())
