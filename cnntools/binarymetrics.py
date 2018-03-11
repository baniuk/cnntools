"""Binary metrics for Keras.

This module contains some popular binary metrics. Most of the code is grabbed from [1]_ and [2]_

References
----------

.. [1] https://github.com/keras-team/keras/issues/5400
.. [2] https://gist.github.com/wassname/7793e2058c5c9dacb5212c0ac0b18a8a
.. [3] https://arxiv.org/pdf/1606.04797v1.pdf
"""

from keras import backend as K

smooth = 1.

# TODO Add https://gist.github.com/wassname/f1452b748efcbeb4cb9b1d059dce6f96


def dice_coef(y_true, y_pred):
    """
    Dice coefficient [3]_, code taken from [2]_.

    This implementation assumes that input and output are binary in range 0-1.

    .. math:: D=\\frac{2*\\sum_{i=1}^{N} (x_i*y_i)}{\\sum_{i=1}^{N} x_{i}^{2}+\\sum_{i=1}^{N} y_{i}^{2}}
        :label: dice

    Args:
        y_true (:obj:`numpy`): Ground truth values
        y_pred (:obj:`numpy`): Predicted values

    Returns:
        :obj:`tf.Tensor`: Dice coefficient according to :eq:`dice`. Returned is one single number.

    Example:
        Use the following to evaluate returned tensor:

        .. code-block:: python

            import numpy as np
            from keras import backend as K
            y1 = np.random.rand(2,4,4,1)
            y2 = np.random.rand(2,4,4,1)
            r = dice_coef(y1,y2)
            K.eval(r)

    See Also:
        * :func:`dice_coef_loss`
    """
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(K.abs(y_true_f * y_pred_f))
    return (2. * intersection + smooth) / (K.sum(K.square(y_true_f)) + K.sum(K.square(y_pred_f)) + smooth)


def dice_coef_loss(y_true, y_pred):
    """
    Loss function based on :func:`dice_coef`.

    Defined as :math:`loss=1-D`.

    See Also:
        * :func:`dice_coef`
    """
    return 1 - dice_coef(y_true, y_pred)


def mcor(y_true, y_pred):
    """Matthews_correlation."""
    y_pred_pos = K.round(K.clip(y_pred, 0., 1.))
    y_pred_neg = 1 - y_pred_pos

    y_pos = K.round(K.clip(y_true, 0., 1.))
    y_neg = 1 - y_pos

    tp = K.sum(y_pos * y_pred_pos)
    tn = K.sum(y_neg * y_pred_neg)

    fp = K.sum(y_neg * y_pred_pos)
    fn = K.sum(y_pos * y_pred_neg)

    numerator = (tp * tn - fp * fn)
    denominator = K.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

    return numerator / (denominator + K.epsilon())


def precision(y_true, y_pred):
    """Precision metric.

    Only computes a batch-wise average of precision. Computes the precision, a metric for multi-label classification of
    how many selected items are relevant.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0., 1.)))
    predicted_positives = K.sum(K.round(K.clip(y_pred, 0., 1.)))
    precision = true_positives / (predicted_positives + K.epsilon())
    return precision


def recall(y_true, y_pred):
    """Recall metric.

    Only computes a batch-wise average of recall. Computes the recall, a metric for multi-label classification of
    how many relevant items are selected.
    """
    true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
    possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
    recall = true_positives / (possible_positives + K.epsilon())
    return recall


def f1(y_true, y_pred):
    """F1 metric."""
    def recall(y_true, y_pred):
        """Recall metric.

        Only computes a batch-wise average of recall.

        Computes the recall, a metric for multi-label classification of
        how many relevant items are selected.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        possible_positives = K.sum(K.round(K.clip(y_true, 0, 1)))
        recall = true_positives / (possible_positives + K.epsilon())
        return recall

    def precision(y_true, y_pred):
        """Precision metric.

        Only computes a batch-wise average of precision.

        Computes the precision, a metric for multi-label classification of
        how many selected items are relevant.
        """
        true_positives = K.sum(K.round(K.clip(y_true * y_pred, 0, 1)))
        predicted_positives = K.sum(K.round(K.clip(y_pred, 0, 1)))
        precision = true_positives / (predicted_positives + K.epsilon())
        return precision
    precision = precision(y_true, y_pred)
    recall = recall(y_true, y_pred)
    return 2 * ((precision * recall) / (precision + recall))
