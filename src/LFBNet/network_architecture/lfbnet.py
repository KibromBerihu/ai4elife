#######################################################################################################################

# ---------------------------------->general LFBNet---------------------------implementation --------------------------#

#######################################################################################################################
# import libraries
import os
import sys
import numpy as np
from numpy import ndarray
from copy import deepcopy

import logging
logging.getLogger('tensorflow').disabled = True
import tensorflow as tf
from keras.models import Model
from keras.layers import Conv2D
from keras.layers import Conv3D
from keras.layers import Dropout
from keras.layers import Input
from keras.layers import concatenate, BatchNormalization, Add
from keras.layers import MaxPooling2D, MaxPooling3D

# locate parent directory for absolute import
p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)

# specify coda visible dice if necessary
# CUDA_VISIBLE_DEVICES = 1
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# local import
from src.LFBNet.network_architecture.get_conv_blocks import StackedConvLayerABlock, UpConvLayer
from src.LFBNet.losses.losses import LossMetric


# function to set/configure default parameters for lfbnet.
def get_default_config(dimension: int = 2, dropout_ratio: float = 0.5, non_linear_activation: str = 'elu',
        batch_norm: bool = True, strides: int = 1, pooling: bool = True, pool_size:int =2, default_skips: bool = True,
        kernel_size: int = 3, kernel_initializer: str = 'he_normal', use_bias: bool = False, padding: str = 'same',
        num_conv_per_block: int = 2, skip_encoder=None, use_residual: bool = True,
        apply_dropout_subblock: bool = False) -> dict:

    """ Setup/configure default network configurations

    Args:
        dimension:  dimension of the intended neural network. 2 for 2D, and 3 for 3D network
        dropout_ratio: ratio of drop out
        non_linear_activation: non-linear activation function
        batch_norm: non-linear activation function
        strides:  striding value for the convolution operation
        pooling: apply pooling operation or not
        pool_size: sampling rate
        default_skips: use skip connection between encoder and decoder or not
        kernel_size:  Convolutional operation kernel size
        kernel_initializer: initialization approach, e.g. he_normal
        use_bias: use biase or not
        padding: specify padding operation in  convolution
        num_conv_per_block: number of convolution, activation and batch normalization layers per given block
        skip_encoder: store the skipped connections from the encoder to the decoder network
        use_residual: apply residual connection at each convolutional operations
        apply_dropout_subblock: apply dropout operation on consecutive convolutional operations or convolutional blocks

    Returns:
        Returns dictionary with configured values for the network, LFBNet
    """

    config = {'kernel_size': kernel_size, 'kernel_initializer': kernel_initializer, 'use_bias': use_bias,
              'padding': padding, 'merging_strategy': concatenate, 'skip_encoder': skip_encoder,
              'dropout_ratio': dropout_ratio, 'strides': strides, 'pool_size': pool_size,
              'activation': non_linear_activation, 'default_skips': default_skips,
              'num_conv_per_block': num_conv_per_block, 'use_residual': use_residual, 'dropout': Dropout,
              'apply_dropout_subblock': apply_dropout_subblock}

    if dimension == 2:
        config['conv'] = Conv2D
        config["2D_3D"] = '2D'
        if pooling:
            config['pooling_op'] = MaxPooling2D

    elif dimension == 3:
        config['conv'] = Conv3D
        config["2D_3D"] = '3D'
        if pooling:
            config['pooling_op'] = MaxPooling3D

    else:
        raise Exception("Please use either 2D or 3D CNN, NOT IMPLEMENTED! \n")

    if batch_norm:
        config['batch_norm'] = BatchNormalization

    # use skip of encoder he after combining with the feedback latent hf
    if skip_encoder is None:
        config['skip_encoder'] = Add()

    return config


class LfbNet:
    """configurable lfbnet and returns the forward and feedback system.
    """

    def __init__(self, input_image_shape: ndarray = None, num_output_class: int = 1, base_num_features: int = 32,
            conv_config: dict = None, conv_kernel_sizes:int=3, default_skips: bool = True, num_layers: int = 4,
            use_skip: bool = True, num_classes: int = 1, decoder_input_shape=None, skipped_input=None,
            num_conv_per_block: int = 2):
        """ set parameters to configure LFBNet
        Args:
            input_image_shape: dimension of the input images to the network, e.g,. [128, 256, 1]
            num_output_class: desired number of output classes, e.g., single label segmentation 1, and three level 3
            base_num_features: Number of features at the first block, e.g. 2^x recommended for better computational
            efficiency. But it could be any feature number.
            conv_config: a dictionary to configure LFBNet, e.g.m please see the function get_default_config()
             above
            conv_kernel_sizes: Convolutional operation kernel size
            default_skips: use default skip connections between the encoder and decoder networks
            num_layers: Number of convolutional blocks until the bottleneck or from bottleneck up to the end of decoder.
             Assuming there is pooling operation there will be num_layers-1 number of pooling operations to reach the
             bottleneck.
            use_skip: use skip connection between the encoder and decoder.
            num_classes: Desired number of output classes. if it is >1 softmax activation will be used at the end of
             LFBNet.
            decoder_input_shape: input shape to the decoder or the output of the encoder block.
            skipped_input: skipped values from the encoder and to be connected to the decoder.
            num_conv_per_block: a series of consecutive convolution, batch normalization, activation operations.
        """

        if input_image_shape is None:
            input_image_shape = [128, 256, 1]

        self.img_shape = input_image_shape
        self.channels_out = num_output_class
        self.base_num_features = base_num_features

        latent_dim_input_ratio = 2 ** (num_layers - 1)
        # input_sahpe[-1] is the channel, it will be replaced by base_num_features*latent_dim_input_ratio
        self.latent_dim = [int(dim / latent_dim_input_ratio) for dim in input_image_shape]

        # add the at last the number of features : base_num_features * latent_dim_input_ratio in feature space
        self.latent_dim[-1] = int(base_num_features * latent_dim_input_ratio)

        self.optimizer = tf.keras.optimizers.Adam(lr=3e-4)

        # if conv_config is not given: take the default values
        if conv_config is None:
            self.conv_config = deepcopy(get_default_config())

        self.base_num_features = base_num_features

        # forward network parameters
        self.conv_kernel_sizes = conv_kernel_sizes
        self.input_image_shape = input_image_shape
        self.default_skips = default_skips
        self.base_num_features = base_num_features
        self.num_layers = num_layers

        # forward decoder network
        self.num_classes = num_classes

        decoder_input_shape = [int(bottleneck_dim / (2 ** (num_layers - 1))) for bottleneck_dim in
                               self.input_image_shape]
        # multiply the last channel with the base number of features
        decoder_input_shape[-1] = base_num_features * (2 ** (num_layers - 1))

        '''
        print("decoder input shape \n")
        print(decoder_input_shape)
        '''

        if skipped_input is None:
            # skipp connections
            skipped_input = []
            for stage in range(num_layers):
                skipped_input.append(
                    [int(decoder_input_shape[0] * (2 ** stage)), int(decoder_input_shape[1] * (2 ** stage)),
                     int(base_num_features * (2 ** (num_layers - (1 + stage))))])

        # print("skipped_connections setup")
        # print(skipped_input)

        self.skipped_input = skipped_input

        if use_skip:
            # select the connection strategy
            if not self.conv_config['merging_strategy']:
                self.conv_config['merging_strategy'] = concatenate
        else:
            self.conv_config['merging_strategy'] = None

            if use_skip:
                # The number of skip inputs should be the same as the num of decoder stages, except bottleneck
                # skipped_input consists the skip connections from encoder, and the bottleneck output
                assert self.num_layers == (len(self.skipped_input) - 1)

        # losses
        self.loss_metric = LossMetric()

        """

        """
        # define forward encoder network
        self.forward_encoder = self.define_forward_encoder()

        # print("forward encoder summary\n")
        # self.forward_encoder.summary()

        self.forward_decoder = self.define_forward_decoder()

        # print("forward decoder summary\n")
        # self.forward_decoder.summary()

        self.forward_decoder.compile(optimizer=self.optimizer,
                                     loss=self.loss_metric.dice_plus_binary_cross_entropy_loss,
                                     metrics=[self.loss_metric.dice_metric])

        """

        """

        # combine the encoder and decoder
        # input image
        img_input = Input(shape=self.img_shape, name='input')
        # encoder outputs
        encoder_output = self.forward_encoder(img_input)

        # h0 and ht  input the decoder, feedback_latent
        img_input_latent = Input(shape=self.latent_dim, name='input_latent')

        encoder_output = encoder_output[::-1]
        encoder_output.insert(1, img_input_latent)
        decoder_output = self.forward_decoder([encoder_output[i] for i in range(len(encoder_output))])

        """
        
        """

        # combined model training both encoder and decoder together
        self.combine_and_train = Model(inputs=[img_input, img_input_latent], outputs=[decoder_output])

        # print('Forward Encoder and decoder network combined summary: \n ')
        # self.combine_and_train.summary()

        self.combine_and_train.compile(loss=self.loss_metric.dice_plus_binary_cross_entropy_loss,
                                       optimizer=self.optimizer, metrics=[self.loss_metric.dice_metric])

        """

        """

        # FCN
        self.fcn_feedback = self.define_feedback_fcn_network()

        # # compile model
        # print('Feedback FCN network summary  \n')
        # self.fcn_feedback.summary()

        self.feedback_latent = Model(inputs=[self.fcn_feedback.input],
                                     outputs=[self.fcn_feedback.get_layer('latent_space_fcn').output])

    """

    """

    def define_forward_encoder(self):
        """ forward system's encoder model.

        Returns:
            Returns a forward system's encoder block that contains series of convolutions, max-pooling operations. It
            also returns the skip values for decoder.

        """
        # the output of the forward encoder layer
        skips = []
        inputs = Input(shape=self.input_image_shape, name='input_forward_encoder')
        current_stage = inputs
        # consecutive convolution, batch normalization, and activation blocks, and skipp connections until bottleneck
        for stage in range(self.num_layers):
            current_output_num_features = int(self.base_num_features * (2 ** stage))
            current_stage = StackedConvLayerABlock(current_stage, current_output_num_features,
                                                   conv_config=self.conv_config, num_conv_per_block=2).conv_block()
            # Pooling operation when the stage is before the bottleneck
            if stage != (self.num_layers - 1):
                # keep the skipp connections before the pooling operation, and the final bottleneck layer of the Encoder
                if self.default_skips:
                    skips.append(current_stage)
                current_stage = self.conv_config['pooling_op'](pool_size=self.conv_config['pool_size'])(current_stage)

        # bottleneck layer of the Encoder, if no skip is required self.skips will have only one output, bottleneck
        skips.append(current_stage)

        return Model(inputs=[inputs], outputs=[skips[index] for index in range(len(skips))])

    """

    """

    def define_forward_decoder(self):
        """ forward system's decoder model.

        Returns:
            Returns forward system's decoder model. It consists a series of up sampling, concatenation layer,
            and convolutional blocks.

        """
        # direct input from encoder, bottleneck
        inputs = Input(shape=np.asarray(self.skipped_input[0]), name="input_from_decoder")
        # set the two inputs from the two encoders
        inputs_forward_encoder = inputs
        inputs_feedback_encoder = Input(shape=self.latent_dim, name='input_from_feedback')

        # change the input dimension into input tensors
        skip_input = []
        for skip in range(len(self.skipped_input) - 1):
            skip_input.append(
                Input(shape=np.asarray(self.skipped_input[skip + 1]), name='input_from_encoder' + str(skip)))

        '''
        The bottleneck need to do the feedback connection: 
        To use U-net-based segmentation jump or comment this block 
        '''

        concatenate_encoder_feedback = self.conv_config['merging_strategy'](
            [inputs_forward_encoder, inputs_feedback_encoder])

        fused_bottle_neck = StackedConvLayerABlock(concatenate_encoder_feedback,
                                                   int(self.base_num_features * (2 ** (self.num_layers - 1))),
                                                   conv_config=self.conv_config, num_conv_per_block=self.conv_config[
                'num_conv_per_block']).conv_block()

        fused_bottle_neck = Add()([fused_bottle_neck, inputs_forward_encoder])
        # fused_bottle_neck = BatchNormalization()(fused_bottle_neck)

        fused_bottle_neck = StackedConvLayerABlock(fused_bottle_neck,
                                                   int(self.base_num_features * (2 ** self.num_layers)),
                                                   conv_config=self.conv_config, num_conv_per_block=self.conv_config[
                'num_conv_per_block']).conv_block()

        # apply drop out at the bottleneck
        fused_bottle_neck = Dropout(self.conv_config['dropout_ratio'])(fused_bottle_neck)

        current_up_conv = fused_bottle_neck

        for decoder_stage in range(self.num_layers - 1):
            # decrease the number of features per block:  (self.num_decoder_stage-decoder_stage)
            num_output_features = int(self.base_num_features * (2 ** (self.num_layers - (2 + decoder_stage))))
            current_up_conv = UpConvLayer(current_up_conv, num_output_features=num_output_features,
                                          conv_upsampling="2D").Up_conv_layer()
            # Need skipp connections:
            if self.conv_config['merging_strategy'] is not None:
                skipped_ = skip_input[decoder_stage]
                current_up_conv = self.conv_config['merging_strategy']([current_up_conv, skipped_])

            # convolution blocks
            current_up_conv = StackedConvLayerABlock(current_up_conv, num_output_features, conv_config=self.conv_config,
                                                     num_conv_per_block=self.conv_config[
                                                         'num_conv_per_block']).conv_block()
        """

        """
        # final output layer
        if self.num_classes == 1:
            activation = 'sigmoid'
        elif self.num_classes > 1:
            activation = 'softmax'
        else:
            raise Exception("\n Not known output activation function \n")

        current_up_conv = self.conv_config['conv'](self.num_classes, kernel_size=1, activation=activation,
                                                   kernel_initializer='he_normal', use_bias=False, padding='same',
                                                   name='final_output_layer')(current_up_conv)

        skip_input.insert(0, inputs)
        skip_input.insert(1, inputs_feedback_encoder)

        return Model(inputs=[inputs for inputs in skip_input], outputs=[current_up_conv])

    def define_feedback_fcn_network(self):
        """ define the feedback system of the lfbnet.

        Returns:
            Returns the feedback system model.

        """
        # No need of skipp connection for the time being
        inputs = Input(shape=self.input_image_shape, name='input_feedback_encoder')
        current_stage = inputs
        # Encoder part
        # consecutive convolution, batch normalization, and activation blocks,
        for stage in range(self.num_layers):
            current_output_num_features = int(self.base_num_features * (2 ** stage))
            current_stage = StackedConvLayerABlock(current_stage, current_output_num_features,
                                                   conv_config=self.conv_config, num_conv_per_block=2).conv_block()
            # Pooling operation when the stage is before the bottleneck
            if stage != (self.num_layers - 1):
                current_stage = self.conv_config['pooling_op'](pool_size=self.conv_config['pool_size'])(current_stage)
            else:  # bottleneck
                current_stage = Dropout(self.conv_config['dropout_ratio'], name='latent_space_fcn')(current_stage)

        # Decoder part: up sampling
        current_up_conv = current_stage
        for decoder_stage in range(self.num_layers - 1):
            # set number of features
            num_output_features = int(self.base_num_features * (2 ** (self.num_layers - (2 + decoder_stage))))

            # up convolution block
            current_up_conv = UpConvLayer(current_up_conv, num_output_features=num_output_features,
                                          conv_upsampling="2D").Up_conv_layer()

            # convolution blocks
            current_up_conv = StackedConvLayerABlock(current_up_conv, num_output_features, conv_config=self.conv_config,
                                                     num_conv_per_block=self.conv_config[
                                                         'num_conv_per_block']).conv_block()

        # final output layer
        if self.num_classes == 1:
            activation = 'sigmoid'
        elif self.num_classes > 1:
            activation = 'softmax'
        else:
            raise Exception("\n Not known output activation function \n")

        current_up_conv = self.conv_config['conv'](self.num_classes, kernel_size=1, activation=activation,
                                                   kernel_initializer='he_normal', use_bias=False, padding='same',
                                                   name='fcn_output_layer')(current_up_conv)

        fcn_feedback_model = Model(inputs=[inputs], outputs=[current_up_conv])

        fcn_feedback_model.compile(loss=self.loss_metric.dice_plus_binary_cross_entropy_loss, optimizer=self.optimizer,
                                   metrics=[self.loss_metric.dice_metric])
        return fcn_feedback_model


if __name__ == '__main__':
    print("default config")
    props = get_default_config()
    model = LfbNet()
    print("network summary \n")


