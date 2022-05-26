"""" Script to compute different loss functions in Keras based on tensorflow.

This script compute dice loss, binary cross entropy loss, focal loss, and their combinations.
It also computes hard and soft dice metric as well as loss.

"""

# Import libraries
from numpy.random import seed
from typing import List, Tuple
from keras import backend as K
import tensorflow as tf
from numpy import ndarray

# seed random number generator
seed(1)


class LossMetric:
    """ compute loss and metrics

    Attributes:
        y_true: the reference value,
        y_predicted: the predicted value to compare with y_true.

    Returns:
        Returns the loss or metric.

    """
    def __init__(self, y_true: List[float] = None, y_predicted: List[float] = None):
        self.y_true = y_true
        self.y_predicted = y_predicted

    @staticmethod
    def dice_metric(y_true: ndarray = None, y_predicted: ndarray = None, soft_dice: bool = False,
            threshold_value: float = 0.5, smooth=1) -> float:
        """compute the dice coefficient between the reference and target
        Threshold dice similarity coefficient

        Args:
            y_true: reference target.
            y_predicted:  predicted target by the model.
            soft_dice: apply soft dice or not.
            threshold_value:  thresholding value for soft-dice application.
            smooth: avoid division by zero values.

        Returns:
            Returns dice similarity coefficient, with threshold predicted values

        """
        y_true = K.flatten(y_true)
        y_predicted = K.flatten(y_predicted)
        # prevent from log(0)
        y_true = K.clip(y_true, 10e-8, 1. - 10e-8)
        y_predicted = K.clip(y_predicted, 10e-8, 1. - 10e-8)

        # soft dice
        if soft_dice:
            y_predicted = K.cast(K.greater(y_predicted, threshold_value), dtype='float32')

        intersection = K.sum(y_true * y_predicted)

        # smooth: avoid by zero division
        dice = (2. * intersection + smooth) / (K.sum(y_true) + K.sum(y_predicted) + smooth)

        return dice

    def dice_loss(self, y_true: ndarray, y_predicted: ndarray) -> float:
        """ Compute the dice loss

        Args:
            y_true: reference target.
            y_predicted:  predicted target by the model.

        Returns:
            Returns dice loss.

        """
        return 1 - self.dice_metric(y_true, y_predicted)

    @staticmethod
    def binary_cross_entropy_loss(y_true: ndarray = None, y_predicted: ndarray = None) -> float:
        """ compute the binary cross entropy loss

        Args:
            y_true: reference target.
            y_predicted:  predicted target by the model.

        Returns:
            Returns binary cross entropy between the target and predicted value.

        """
        # prevent from log(0)
        y_true = K.clip(y_true, 10e-8, 1. - 10e-8)
        y_predicted = K.clip(y_predicted, 10e-8, 1. - 10e-8)

        return K.binary_crossentropy(y_true, y_predicted)

    @staticmethod
    def binary_focal_loss(y_true: ndarray = None, y_predicted: ndarray = None, gamma: int = 2,
            alpha: float = .25) -> float:
        """ computes the focal loss

        Args:
            y_true: reference target.
            y_predicted: predicted target by the model.
            gamma: constant value
            alpha: constant value

        Returns:
            Returns focal loss.

        """
        y_true = K.cast(y_true, dtype='float32')
        # Define epsilon so that the back-propagation will not result in NaN for 0 divisor case
        epsilon = K.epsilon()
        # Add the epsilon to prediction value
        y_predicted = y_predicted + epsilon
        # Clip the prediction value
        y_predicted = K.clip(y_predicted, epsilon, 1.0 - epsilon)

        # Calculate p_t
        p_t = tf.where(K.equal(y_true, 1), y_predicted, 1 - y_predicted)
        # Calculate alpha_t
        alpha_factor = K.ones_like(y_true) * alpha
        alpha_t = tf.where(K.equal(y_true, 1), alpha_factor, 1 - alpha_factor)
        # Calculate cross entropy
        cross_entropy = -K.log(p_t)
        weight = alpha_t * K.pow((1 - p_t), gamma)
        # Calculate focal loss f
        loss = weight * cross_entropy
        # Sum the losses in mini_batch
        loss = K.mean(K.sum(loss, axis=1))
        return loss

    @staticmethod
    def focal_loss(y_true: ndarray = None, y_predicted: ndarray = None, gamma: int = 2, alpha: float = .25) -> float:
        """ computes the focal loss

        Adapted from: https://www.kaggle.com/code/bigironsphere/loss-function-library-keras-pytorch/notebook
        """

        y_true = K.flatten(y_true)
        y_predicted = K.flatten(y_predicted)

        bce = K.binary_crossentropy(y_true, y_predicted)
        bce_exp = K.exp(-bce)
        focal_loss = K.mean(alpha * K.pow((1 - bce_exp), gamma) * bce)

        return focal_loss

    def dice_plus_binary_cross_entropy_loss(self, y_true, y_predicted):
        """ compute the average of the sum of dice and binary cross entropy loss.

        Args:
            y_true: reference target.
            y_predicted:  predicted target by the model.

        Returns:
            Returns the average of the sum of dice and binary cross entropy losses.

        """
        loss = 0.5 * (self.dice_loss(y_true, y_predicted) + self.binary_cross_entropy_loss(y_true=y_true,
            y_predicted=y_predicted))

        return loss

    def dice_plus_focal_loss(self, y_true: ndarray, y_predicted: ndarray) -> float:
        """ compute the sum of the dice and focal loss

        Args:
            y_true: reference target.
            y_predicted:  predicted target by the model.

        Returns:
            Returns the sum of the dice and focal loss.

        """
        return self.dice_loss(y_true, y_predicted) + self.binary_focal_loss(y_true, y_predicted)

    @staticmethod
    def iou_loss(y_true, y_predicted, smooth=1e-8):
        """ compute the intersection over union loss.

        Args:
            y_true: reference target.
            y_predicted:  predicted target by the model.
            smooth: avoid division by zero.

        Returns:
            Returns intersection over union loss.

        """
        y_true = K.flatten(y_true)
        y_predicted = K.flatten(y_predicted)

        intersection = K.sum(K.dot(y_true, y_predicted))
        total = K.sum(y_true) + K.sum(y_predicted)

        union = total - intersection

        iou = (intersection + smooth) / (union + smooth)

        return iou
