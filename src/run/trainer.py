""" trainer python script

This script allows training the proposed lfbnet model.
This script requires to specify the directory path to the preprocessed PET MIP images. It could read the patient ids
from
the given directory path, or it could accept patient ids as .xls and .csv files. please provide the directory path to
the
csv or xls file. It assumes the csv/xls file have two columns with level 'train' and 'valid' indicating the training and
validation patient ids respectively.

Please see the _name__ == '__main__': as example which is equivalent to:

e.g.train_valid_data_dir = r"E:\LFBNet\data\remarc_default_MIP_dir/"
    train_valid_ids_path_csv = r'E:\LFBNet\data\csv\training_validation_indexs\remarc/'
    train_ids, valid_ids = get_training_and_validation_ids_from_csv(train_valid_ids_path_csv)

    trainer = NetworkTrainer(
        folder_preprocessed_train=train_valid_data_dir, folder_preprocessed_valid=train_valid_data_dir,
        ids_to_read_train=train_ids,
        ids_to_read_valid=valid_ids
        )
    trainer.train()
"""
# Import libraries
import os
import glob
import sys
import time
from datetime import datetime

import numpy as np
from numpy.random import seed
from random import randint
from tqdm import tqdm
from typing import Tuple, List
from numpy import ndarray
from copy import deepcopy
from medpy.metric import binary
import matplotlib.pyplot as plt
from keras import backend as K
import re

# make LFBNet as parent directory, for absolute import libraries. local application import.
p = os.path.abspath('../..')
if p not in sys.path:
    sys.path.append(p)

# import LFBNet modules
from src.LFBNet.data_loader import DataLoader
from src.LFBNet.network_architecture import lfbnet
from src.LFBNet.losses import losses
from src.LFBNet.preprocessing import save_nii_images
from src.LFBNet.utilities import train_valid_paths
from src.LFBNet.postprocessing import remove_outliers_in_sagittal
# choose cuda gpu
CUDA_VISIBLE_DEVICES = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "1"

# set randomness repetable across experiments.
seed(1)

# Define the parameters of the data to process
K.set_image_data_format('channels_last')


def default_training_parameters(
        num_epochs: int = 5000, batch_size: int = 16, early_stop: int = None, fold_number: int = None,
        model_name_save: List[str] = None, loss: str = None, metric: str = None
        ) -> dict:
    """ Configure default parameters for training.
    Training parameters are setted here. For other options, the user should modifier these values.
    Parameters
    ----------
    num_epochs: int, maximum number of epochs to train the model.
    batch_size: int, number of images per batch
    early_stop: int, the number of training epochs the model should train while it is not improving the accuracy.
    fold_number: int, optional, fold number while applying cross-validation-based training.
    model_name_save: str, model name to save
    loss: str, loss funciton
    metric: str, specify the metric, such as dice

    Returns
    -------
    Returns configured dictionary for the training.

    """
    if early_stop is None:
        # early stop 50 % of the maximum number of epochs
        early_stop = int(num_epochs * 0.5)

    if fold_number is None:
        fold_number = 'fold_run_at_' + str(time.time())

    if model_name_save is None:
        model_name_save = ["forward_" + str(time.time()), "feedback_" + str(time.time())]

    if loss is None:
        loss = losses.LossMetric.dice_plus_binary_cross_entropy_loss

    if metric is None:
        metric = losses.LossMetric.dice_metric

    config_trainer = {'num_epochs': num_epochs, 'batch_size': batch_size, 'num_early_stop': early_stop,
                      'fold_number': fold_number, 'model_name_save_forward': model_name_save[0],
                      'model_name_save_feedback': model_name_save[1], "custom_loss": loss, "custom_dice": metric}

    return config_trainer


def get_training_and_validation_ids_from_csv(path):
    """ Get training and validation ids from a given csv or xls file. Assuming the training ids are given with column
    name 'train' and validation ids in 'valid'

    Parameters
    ----------
    path: directory path to the csv or xls file.

    Returns
    -------
    Returns training and validation patient ids.


    """
    ids = train_valid_paths.read_csv_train_valid_index(path)
    train, valid = ids[0], ids[1]
    return train, valid


def get_train_valid_ids_from_folder(path_train_valid, ratio_valid_data=0.25):
    """ Returns the randomly split training and validation patient ids. The percentage of validation is given by the
    ratio_valid_data.

    Parameters
    ----------
    path_train_valid
    ratio_valid_data

     Returns
    -------
    Returns training patient id and validation patient ids respectively as in two array.

    """
    # given training and validation data on one folder, random splitting with .25% : train, valid
    if len(path_train_valid) == 1:
        all_cases_id = os.listdir(str(path_train_valid))  # all patients id

        # make permutation in the given list
        case_ids = np.array(all_cases_id)
        indices = np.random.permutation(len(case_ids))
        num_valid_data = int(ratio_valid_data * len(all_cases_id))

        train, valid = indices[num_valid_data:], indices[:num_valid_data]
        return [train, valid]


class NetworkTrainer:
    """
    class to train the lfb net
    """
    # keep the best loss and dice while training : Value shared across all instances, methods
    BEST_METRIC_VALIDATION = 0  # KEEP THE BEST VALIDATION METRIC SUCH AS THE DICE METRIC (BEST_DICE)
    BEST_LOSS_VALIDATION = 100  # KEEP THE BEST VALIDATION LOSS SUCH AS THE LOSS VALUES (BEST_LOSS)
    EARLY_STOP_COUNT = 0  # COUNTS THE NUMBER OF TRAINING ITERATIONS THE MODEL DID NOT INCREASE, TO COMPARE WITH THE
    now = datetime.now()  # current time, date, month,
    TRAINED_MODEL_IDENTIFIER = re.sub('[ :]', "_", now.ctime())

    # EARLY STOP CRITERIA

    def __init__(
            self, config_trainer: dict = None, folder_preprocessed_train: str = '../data/train/',
            folder_preprocessed_valid: str = '../data/valid/', ids_to_read_train: ndarray = None,
            ids_to_read_valid: ndarray = None, task: str = 'valid', predicted_directory: str = '../data/predicted/',
            save_predicted: bool = False
            ):
        """

        :param config_trainer:
        :param folder_preprocessed_train:
        :param folder_preprocessed_valid:
        :param ids_to_read_train:
        :param ids_to_read_valid:
        :param task:
        :predicted_directory:
        :save_predicted
        """

        if config_trainer is None:
            self.config_trainer = deepcopy(default_training_parameters())

        # training data
        self.folder_preprocessed_train = folder_preprocessed_train
        if ids_to_read_train is None:
            ids_to_read_train = os.listdir(folder_preprocessed_train)

        self.ids_to_read_train = ids_to_read_train

        # validation data
        self.folder_preprocessed_valid = folder_preprocessed_valid
        if ids_to_read_valid is None:
            ids_to_read_valid = os.listdir(folder_preprocessed_valid)
        self.ids_to_read_valid = ids_to_read_valid

        # save predicted directory:
        self.save_all = save_predicted
        self.predicted_directory = predicted_directory
        # load the lfb_network architecture
        self.model = lfbnet.LfbNet()
        self.task = task

        # forward network decoder

        # latent feedback at zero time: means no feedback from feedback network
        self.latent_dim = self.model.latent_dim
        self.h_at_zero_time = np.zeros(
            (int(self.config_trainer['batch_size']), int(self.latent_dim[0]), int(self.latent_dim[1]),
             int(self.latent_dim[2])), np.float32
            )

    @staticmethod
    def load_dataset(directory_: str = None, ids_to_read: List[str] = None):
        """

        :param ids_to_read:
        :param directory_:
        """
        # load batch of data
        data_loader = DataLoader(data_dir=directory_, ids_to_read=ids_to_read)
        image_batch_ground_truth_batch = data_loader.get_batch_of_data()

        batch_input_data, batch_output_data = image_batch_ground_truth_batch[0], image_batch_ground_truth_batch[1]
        # expand dimension for the channel
        batch_output_data = np.expand_dims(batch_output_data, axis=-1)
        batch_input_data = np.expand_dims(batch_input_data, axis=-1)

        return batch_input_data, batch_output_data

    def load_latest_weight(self):
        """ loads the weights of the model with the latest saved weight in the folder ./weight
        """
        # load the last trained weight in the folder weight
        folder_path = r'./weight/'
        file_type = r'\*.h5'
        files = glob.glob(folder_path + file_type)
        try:
            max_file = max(files, key=os.path.getctime)
        except:
            raise Exception("weight could not found !")

        base_name = str(os.path.basename(max_file))
        print(base_name)
        self.model.combine_and_train.load_weights('./weight/forward_system' + str(base_name.split('system')[1]))
        # f
        self.model.fcn_feedback.load_weights('./weight/feedback_system' + str(base_name.split('system')[1]))

    def train(self):
        """Train the model
        """

        batch_size = self.config_trainer['batch_size']
        # self.load_latest_weight()
        # training
        if self.task == 'train':
            # training
            for current_epoch in range(self.config_trainer['num_epochs']):
                feedback_loss_dice = []
                forward_loss_dice = []
                forward_decoder_loss_dice = []

                # shuffle the index of the training data
                index_read = np.random.permutation(int(len(self.ids_to_read_train)))
                # read data
                for selected_patient in range(len(index_read)):
                    # get index of batch of data
                    start = selected_patient * batch_size
                    idx_list_batch = index_read[start:start + batch_size]
                    # if there are still elements in the given batch
                    if idx_list_batch.size > 0:
                        # get index of Why not ? kk = indx_list_batch
                        kk = [str(k) for i, k in enumerate(self.ids_to_read_train) if i in idx_list_batch]

                        batch_input_data, batch_output_data = self.load_dataset(
                            directory_=self.folder_preprocessed_train, ids_to_read=kk
                            )

                        assert len(batch_input_data) > 0, "batch of data not loaded correctly"

                        # shuffle within the batch
                        index_batch = np.random.permutation(int(batch_input_data.shape[0]))
                        batch_input_data = batch_input_data[index_batch]
                        batch_output_data = batch_output_data[index_batch]

                        # batches per epoch: Selected batch might as in id could have more images than the batch size
                        batch_per_epoch = int(batch_input_data.shape[0] / batch_size)
                        for batch_per_epoch_ in range(batch_per_epoch):
                            batch_input = batch_input_data[
                                          batch_per_epoch_ * batch_size:(batch_per_epoch_ + 1) * batch_size]
                            batch_output = batch_output_data[
                                           batch_per_epoch_ * batch_size:(batch_per_epoch_ + 1) * batch_size]

                            # Train forward models
                            if current_epoch % 2 == 0:
                                # step 1: train the forward network encoder and decoder
                                loss, dice = self.model.combine_and_train.train_on_batch(
                                    [batch_input, self.h_at_zero_time], [batch_output]
                                    )  # self.h_at_zero_time
                                forward_loss_dice.append([loss, dice])

                            else:
                                predicted_decoder = self.model.combine_and_train.predict(
                                    [batch_input, self.h_at_zero_time]
                                    )  # , self.h_at_zero_time

                                # step 2: train the feedback network, considering the output of the forward network
                                loss, dice = self.model.fcn_feedback.train_on_batch(predicted_decoder, batch_output)
                                feedback_loss_dice.append([loss, dice])

                                # Step 3: train the forward decoder, considering the trained
                                feedback_latent_result = self.model.feedback_latent.predict([predicted_decoder])
                                forward_encoder_output = self.model.forward_encoder.predict([batch_input])

                                # forward_encoder_output.insert(1, feedback_latent_result)
                                forward_encoder_output = forward_encoder_output[::-1]  # bottleneck should be first
                                forward_encoder_output.insert(1, feedback_latent_result)
                                loss, dice = self.model.forward_decoder.train_on_batch(
                                    [output for output in forward_encoder_output], [batch_output]
                                    )
                                forward_decoder_loss_dice.append([loss, dice])

                forward_loss_dice = np.array(forward_loss_dice)
                feedback_loss_dice = np.array(feedback_loss_dice)
                forward_decoder_loss_dice = np.array(forward_decoder_loss_dice)

                if current_epoch % 2 == 0:
                    loss, dice = np.mean(forward_loss_dice, axis=0)
                    print(
                        'Training_forward_system: >%d, '
                        ' fwd_loss = %.3f, fwd_dice=%0.3f, ' % (current_epoch, loss, dice)
                        )

                else:
                    loss_forward, dice_forward = np.mean(forward_decoder_loss_dice, axis=0)
                    loss_feedback, dice_feedback = np.mean(feedback_loss_dice, axis=0)

                    print(
                        'Training_forward_decoder_and_feedback_system: >%d, '
                        'fwd_decoder_loss=%03f, '
                        'fwd_decoder_dice=%0.3f '

                        'fdb_loss=%03f, '
                        'fdb_dice=%.3f ' % (current_epoch, loss_forward, dice_forward, loss_feedback, dice_feedback)
                        )
                # validation test:
                self.validation(current_epoch=current_epoch)

                # CHECK TRAINING STOPPING CRITERIA:  maximum number of epochs (epoch - 1), meet early stop
                if NetworkTrainer.EARLY_STOP_COUNT == self.config_trainer['num_early_stop']:
                    # save model with early stop identification
                    self.model.combine_and_train.save(
                        'weight/forward_system_early_stopped_' + str(NetworkTrainer.TRAINED_MODEL_IDENTIFIER) + '_.h5'
                        )
                    self.model.fcn_feedback.save(
                        'weight/feedback_system_early_stopped_' + str(NetworkTrainer.TRAINED_MODEL_IDENTIFIER) + '_.h5'
                        )
                    break  # STOP TRAINING WITH BREAK, OR EXIT TRAINING

        # if not training load the last saved weights, and check validation
        elif self.task == 'valid':
            # load the last trained weight in the folder weight
            self.load_latest_weight()
            self.validation(current_epoch=self.config_trainer['num_epochs'])

    def validation(self, verbose: int = 0, current_epoch: int = None):
        """
        Compute the validation dice, loss of the training from the validation data
        """
        # path to the validation data, if not specified, the default path ../data/valid/ would be considered

        folder_preprocessed = self.folder_preprocessed_valid

        # image folder names, or identifier: if not specified the default values would be the name of the folder inside
        # the directory "folder processed" or the self.folder_processed_valid :

        valid_identifier = self.ids_to_read_valid

        '''
        WE CAN IMPLEMENT THE EVALUATION METHOD AS BATCH BASED, PATIENT BASED, OR THE WHOLE-VALIDATION DATA BASED. FOR
        THE LAST OPTION WE NEED TO IMPLEMENT THE EVALUATION() FUNCTION HERE. 
        '''

        ''''
        declare variables to return: 
                        forward loss and dice with h0 (no feedback), 
                        feedback network loss and dice
                        forward decoder loss and dice,
                        forward loss and dice with ht (with feedback latent space)
        '''
        loss_dice = {'loss_fwd_h0': [], 'dice__fwd_h0': [], 'loss_fdb_h0': [], 'dice_fdb_h0': [],
                     'loss_fwd_decoder': [], 'dice_fwd_decoder': [], 'loss_fwd_ht': [], 'dice_fwd_ht': []}

        all_dice_sen_sep = {'dice': [], 'specificity': [], 'sensitivity': []}

        # load the dataset,
        # get the validation ids
        for id_to_validate in valid_identifier:
            try:
                id_to_validate = str(id_to_validate).split('.')[0]
            except:
                pass

            valid_input, valid_output = self.load_dataset(directory_=folder_preprocessed, ids_to_read=[id_to_validate])

            if len(valid_input) == 0:
                print("data %s not read" % id_to_validate)
                continue

            results, dice_sen_sep = self.evaluation(
                input_image=valid_input.copy(), ground_truth=valid_output.copy(), case_name=str(id_to_validate)
                )

            # append all loss to loss and dice to dice from all cases in valid identifiers
            for keys in results.keys():
                loss_dice[str(keys)].append(results[str(keys)][0])

            for keys in dice_sen_sep.keys():
                all_dice_sen_sep[str(keys)].append(dice_sen_sep[str(keys)][0])

        print("\n Dice, sensitivity, specificity \t")
        for k, v in all_dice_sen_sep.items():
            print('%s :  %0.3f ' % (k, np.mean(list(v), axis=0)), end=" ")
        print("\n")

        """
        print the mean of the validation loss and validation dice
        """

        # FOR STOPPING CRITERIA WE ARE USING THE MODEL AT THE 3RD STEP
        dice_mean = np.mean(loss_dice['dice_fwd_ht'])
        loss_mean = np.mean(loss_dice['loss_fwd_ht'])

        # at the first epoch
        if current_epoch == 0:
            NetworkTrainer.BEST_METRIC_VALIDATION = dice_mean
            NetworkTrainer.BEST_LOSS_VALIDATION = loss_mean

        # compare the current dice and loss with the previous epoch's loss and dice:
        # NOW CONSIDER DICE AS OPTIMIZATION METRIC
        print("Current validation loss and metrics at epoch %d: >> " % current_epoch, end=" ")
        for k, v in loss_dice.items():
            print('%s :  %0.3f ' % (k, np.mean(v)), end=" ")
        print("\n")

        if NetworkTrainer.BEST_METRIC_VALIDATION <= dice_mean:
            # reset early stop count, best dice, and best loss values
            NetworkTrainer.BEST_LOSS_VALIDATION = loss_mean
            NetworkTrainer.BEST_METRIC_VALIDATION = dice_mean
            NetworkTrainer.EARLY_STOP_COUNT = 0

            # save the best model weights
            if not os.path.exists('./weight'):
                os.mkdir('./weight')

            self.model.combine_and_train.save(
                'weight/forward_system_' + str(NetworkTrainer.TRAINED_MODEL_IDENTIFIER) + '_.h5'
                )
            self.model.fcn_feedback.save(
                'weight/feedback_system_' + str(NetworkTrainer.TRAINED_MODEL_IDENTIFIER) + '_.h5'
                )
        else:  # just print the current validation metric (dice) and loss, and count early stop
            # Increase the early stop count per epoch
            NetworkTrainer.EARLY_STOP_COUNT += 1

        print(
            '\n Best model on validation data : %0.3f :  Dice: %0.3f \n' % (
                NetworkTrainer.BEST_LOSS_VALIDATION, NetworkTrainer.BEST_METRIC_VALIDATION)
            )

    def evaluation(
            self, verbose: int = 0, input_image: ndarray = None, ground_truth: ndarray = None,
            validation_or_test: str = 'test', case_name: str = None
            ):
        """

        :param case_name:
        :param validation_or_test:
        :param verbose:
        :param input_image:
        :param ground_truth:

        Parameters
        ----------
        save_all
        """
        ''''
        declare variables to return: 
                        forward loss and dice with h0 (no feedback), 
                        feedback network loss and dice
                        forward decoder loss and dice,
                        forward loss and dice with ht (with feedback latent space)
        '''
        all_loss_dice = {'loss_fwd_h0': [], 'dice__fwd_h0': [], 'loss_fdb_h0': [], 'dice_fdb_h0': [],
                         'loss_fwd_decoder': [], 'dice_fwd_decoder': [], 'loss_fwd_ht': [], 'dice_fwd_ht': []}

        dice_sen_sp = {'dice': [], 'specificity': [], 'sensitivity': []}

        # latent  feedback variable h0
        # replace the first number of batches with the number of input images from the first channel
        h0_input = np.zeros(
            (len(input_image), int(self.latent_dim[0]), int(self.latent_dim[1]), int(self.latent_dim[2])), np.float32
            )

        # step 0:
        # Loss and dice on the validation of the forward system
        loss, dice = self.model.combine_and_train.evaluate([input_image, h0_input], [ground_truth], verbose=verbose)
        all_loss_dice['loss_fwd_h0'].append(loss), all_loss_dice['dice__fwd_h0'].append(dice)

        # predict from the forward system
        predicted = self.model.combine_and_train.predict([input_image, h0_input])

        # step 2:
        # Loss and dice on the validation of the feedback system
        loss, dice = self.model.fcn_feedback.evaluate([predicted], [ground_truth], verbose=verbose)
        all_loss_dice['loss_fdb_h0'].append(loss), all_loss_dice['dice_fdb_h0'].append(dice)

        # step 3:
        feedback_latent = self.model.feedback_latent.predict(predicted)  # feedback: hf
        forward_encoder_output = self.model.forward_encoder.predict([input_image])  # forward system's encoder output

        forward_encoder_output = forward_encoder_output[::-1]  # bottleneck should be first
        forward_encoder_output.insert(1, feedback_latent)
        loss, dice = self.model.forward_decoder.evaluate(
            [output for output in forward_encoder_output], [ground_truth], verbose=verbose
            )
        all_loss_dice['loss_fwd_decoder'].append(loss), all_loss_dice['dice_fwd_decoder'].append(dice)

        # loss and dice from the combined and feed back latent space : input  [input_image, fdb_latent_space]
        loss, dice = self.model.combine_and_train.evaluate(
            [input_image, feedback_latent], [ground_truth], verbose=verbose
            )
        all_loss_dice['loss_fwd_ht'].append(loss), all_loss_dice['dice_fwd_ht'].append(dice)
        """
        For the testing time, we use defined metrics on the predicted images instead of using model.evaluate during 
        the validation cases 
        """
        predicted = self.model.combine_and_train.predict([input_image, feedback_latent])

        # binary.dc, sen, and specificty works only on binary images
        dice_sen_sp['dice'].append(
            binary.dc(NetworkTrainer.threshold_image(predicted), NetworkTrainer.threshold_image(ground_truth))
            )
        dice_sen_sp['sensitivity'].append(
            binary.sensitivity(NetworkTrainer.threshold_image(predicted), NetworkTrainer.threshold_image(ground_truth))
            )
        dice_sen_sp['specificity'].append(
            binary.specificity(NetworkTrainer.threshold_image(predicted), NetworkTrainer.threshold_image(ground_truth))
            )
        # all = np.concatenate((ground_truth, predicted, input_image), axis=0)
        # display_image(all)

        # Sometimes save predictions
        if self.save_all:
            predicted = self.model.combine_and_train.predict([input_image, feedback_latent])
            save_nii_images(
                [predicted, ground_truth, input_image], identifier=str(case_name),
                name=[case_name + "_predicted", case_name + "_ground_truth", case_name + "_image"], path_save=self.predicted_directory
                )
        else:

            n = randint(0, 10)
            if n % 3 == 0:
                predicted = self.model.combine_and_train.predict([input_image, feedback_latent])
                save_nii_images(
                    [predicted, ground_truth, input_image], identifier=str(case_name),
                    name=[case_name + "_predicted", case_name + "_ground_truth", case_name + "_image"], path_save=self.predicted_directory
                    )

        return all_loss_dice, dice_sen_sp

    @staticmethod
    def display_image(im_display: ndarray):
        """ display given images

        :param all: 2D image arrays to display
        :returns: display images
        """
        plt.figure(figsize=(10, 8))
        plt.subplots_adjust(hspace=0.5)
        plt.suptitle("Daily closing prices", fontsize=18, y=0.95)
        # loop through the length of tickers and keep track of index
        for n, im in enumerate(im_display):
            # add a new subplot iteratively
            plt.subplot(3, 2, n + 1)
            plt.imshow(im)  # chart formatting
        plt.show()

    @staticmethod
    # binary.dc, sen, and specificty works only on binary images
    def threshold_image(im_: ndarray, thr_value: float = 0.5) -> ndarray:
        """ threshold given input array with the given thresholding value

        :param im_: ndarray of images
        :param thr_value: thresholding value
        :return: threshold array image
        """
        im_[im_ > thr_value] = 1
        im_[im_ < thr_value] = 0
        return im_


class ModelTesting:
    """ performs prediction on a given data set. It predicts the segmentation results, and save the results, calculate
    the clinical metrics such as TMTV, Dmax, sTMTV, sDmax.

    """
    now = datetime.now()  # current time, date, month,
    TRAINED_MODEL_IDENTIFIER = re.sub('[ :]', "_", now.ctime())
    print("current directory", os.getcwd())

    def __init__(
            self, config_test: dict = None, preprocessed_dir: str = '../data/test/', data_list: List[str] = None,
            predicted_dir: str = "../data/predicted"
            ):
        """

        :param config_trainer:
        :param folder_preprocessed_train:
        :param folder_preprocessed_valid:
        :param ids_to_read_train:
        :param ids_to_read_valid:
        :param task:
        :param predicted_dir:
        """

        if config_test is None:
            self.config_test = deepcopy(default_training_parameters())

        # training data
        self.preprocessed_dir = preprocessed_dir
        self.predicted_dir = predicted_dir

        # if the list of testing cases are not given, get from the directory
        if data_list is None:
            data_list = os.listdir(preprocessed_dir)

        self.data_list = data_list

        # load the lfb_network architecture
        self.model = lfbnet.LfbNet()

        # latent feedback at zero time: means no feedback from feedback network
        self.latent_dim = self.model.latent_dim

        # load the last trained weight in the folder weight
        print(os.getcwd())
        folder_path = os.path.join(os.getcwd(), 'src/weight')
        print(folder_path)

        full_path = [path_i for path_i in glob.glob(str(folder_path) + '/*.h5')]

        print("files \n", full_path)
        try:
            max_file = max(full_path, key=os.path.getctime)
        except:
            raise Exception("weight could not found !")

        base_name = str(os.path.basename(max_file))
        print(base_name)
        self.model.combine_and_train.load_weights(
            str(folder_path) + '/forward_system' + str(base_name.split('system')[1])
            )
        # f
        self.model.fcn_feedback.load_weights(str(folder_path) + '/feedback_system' + str(base_name.split('system')[1]))

        self.test()

    def test(self):
        """
                   Compute the validation dice, loss of the training from the validation data
           """
        # path to the validation data, if not specified, the default path ../data/valid/ would be considered
        #
        folder_preprocessed = self.preprocessed_dir
        # image folder names, or identifier: if not specified the default values would be the name of the folder inside
        # the directory "folder processed" or the self.folder_processed_valid :
        test_identifier = self.data_list

        ''''
        declare variables to return if there is a reference segmentation or ground truth : 
                        forward loss and dice with h0 (no feedback), 
                        feedback network loss and dice
                        forward decoder loss and dice,
                        forward loss and dice with ht (with feedback latent space)
        '''
        loss_dice = {'loss_fwd_h0': [], 'dice__fwd_h0': [], 'loss_fdb_h0': [], 'dice_fdb_h0': [],
                     'loss_fwd_decoder': [], 'dice_fwd_decoder': [], 'loss_fwd_ht': [], 'dice_fwd_ht': []}

        # get the validation ids
        test_output = []
        for id_to_test in tqdm(list(test_identifier)):
            test_input, test_output = NetworkTrainer.load_dataset(
                directory_=folder_preprocessed, ids_to_read=[id_to_test]
                )

            if len(test_input) == 0:
                print("data %s not read" % id_to_test)
                continue

            '''
            if there is a ground truth segmentation (gt), and you would like to compare with the predicted segmentation
            by the deep learning model
            '''

            if len(test_output):
                results = self.evaluation_test(
                    input_image=test_input.copy(), ground_truth=test_output.copy(), case_name=str(id_to_test)
                    )

                # append all loss to loss and dice to dice from all cases in valid identifiers
                for keys in results.keys():
                    loss_dice[str(keys)].append(results[str(keys)][0])

                print("Results (sagittal and coronal) for case id: %s  : >> " % id_to_test, end=" ")
                for k, v in loss_dice.items():
                    print('%s :  %0.3f ' % (k, np.mean(v)), end=" ")
                print("\n")

            # Predict the segmentation and save in the folder predicted, dataset identifier
            else:
                self.prediction(input_image=test_input.copy(), case_name=str(id_to_test))

        """
        print the mean of the testing loss and dice if there is a ground truth, for all cases 
        """
        if len(test_output):
            print("Total dataset metrics:  : >> ", end=" ")
            for k, v in loss_dice.items():
                print('%s :  %0.3f ' % (k, np.mean(v)), end=" ")
            print("\n")

    def evaluation_test(
            self, verbose: int = 0, input_image: ndarray = None, ground_truth: ndarray = None,
            validation_or_test: str = 'validate', case_name: str = None
            ):
        """

        :param case_name:
        :param validation_or_test:
        :param verbose:
        :param input_image:
        :param ground_truth:
        """
        ''''
        declare variables to return: 
                        forward loss and dice with h0 (no feedback), 
                        feedback network loss and dice
                        forward decoder loss and dice,
                        forward loss and dice with ht (with feedback latent space)
        '''
        all_loss_dice = {'loss_fwd_h0': [], 'dice__fwd_h0': [], 'loss_fdb_h0': [], 'dice_fdb_h0': [],
                         'loss_fwd_decoder': [], 'dice_fwd_decoder': [], 'loss_fwd_ht': [], 'dice_fwd_ht': []}
        # latent  feedback variable h0
        # replace the first number of batches with the number of input images from the first channel
        h0_input = np.zeros(
            (len(input_image), int(self.latent_dim[0]), int(self.latent_dim[1]), int(self.latent_dim[2])), np.float32
            )

        # step 0:
        # Loss and dice on the validation of the forward system
        loss, dice = self.model.combine_and_train.evaluate([input_image, h0_input], [ground_truth], verbose=verbose)
        all_loss_dice['loss_fwd_h0'].append(loss), all_loss_dice['dice__fwd_h0'].append(dice)

        # predict from the forward system
        predicted = self.model.combine_and_train.predict([input_image, h0_input])

        # step 2:
        # Loss and dice on the validation of the feedback system
        loss, dice = self.model.fcn_feedback.evaluate([predicted], [ground_truth], verbose=verbose)
        all_loss_dice['loss_fdb_h0'].append(loss), all_loss_dice['dice_fdb_h0'].append(dice)

        # step 3:
        feedback_latent = self.model.feedback_latent.predict(predicted)  # feedback: hf
        forward_encoder_output = self.model.forward_encoder.predict([input_image])  # forward system's encoder output

        forward_encoder_output = forward_encoder_output[::-1]  # bottleneck should be first
        forward_encoder_output.insert(1, feedback_latent)
        loss, dice = self.model.forward_decoder.evaluate(
            [output for output in forward_encoder_output], [ground_truth], verbose=verbose
            )
        all_loss_dice['loss_fwd_decoder'].append(loss), all_loss_dice['dice_fwd_decoder'].append(dice)

        # loss and dice from the combined and feed back latent space : input  [input_image, fdb_latent_space]
        loss, dice = self.model.combine_and_train.evaluate(
            [input_image, feedback_latent], [ground_truth], verbose=verbose
            )
        all_loss_dice['loss_fwd_ht'].append(loss), all_loss_dice['dice_fwd_ht'].append(dice)

        """
        For the testing time, we use defined metrics on the predicted images instead of using model.evaluate during 
        the validation cases 
        """
        if validation_or_test == "test":
            # return [dice, specificity, and sensitivity
            return {'dice': binary.dc(predicted, ground_truth),
                    'specificity': binary.specificity(predicted, ground_truth),
                    'sensitivity': binary.sensitivity(predicted, ground_truth)}

        predicted = self.model.combine_and_train.predict([input_image, feedback_latent])
        predicted = remove_outliers_in_sagittal(predicted)
        save_nii_images(
            [predicted, ground_truth, input_image], identifier=str(case_name),
            name=[case_name + "_predicted", case_name + "_ground_truth", case_name + "_pet"],
            path_save= os.path.join(str(self.predicted_dir), 'predicted_data')
            )

        return all_loss_dice

    def prediction(self, input_image: ndarray = None, case_name: str = None):
        """
        :param case_name:
        :param input_image:
        """
        # latent  feedback variable h0
        # replace the first number of batches with the number of input images from the first channel
        h0_input = np.zeros(
            (len(input_image), int(self.latent_dim[0]), int(self.latent_dim[1]), int(self.latent_dim[2])), np.float32
            )

        # STEP 1: forward system prediction
        # predict from the forward system
        predicted = self.model.combine_and_train.predict([input_image, h0_input])

        # step 2: Feedback system prediction
        feedback_latent = self.model.feedback_latent.predict(predicted)  # feedback: hf

        predicted = self.model.combine_and_train.predict([input_image, feedback_latent])
        predicted = remove_outliers_in_sagittal(predicted)
        save_nii_images(
            image=[predicted, input_image], identifier=str(case_name), name=[case_name + "_predicted",
                                                                             case_name + "_pet"],
            path_save= os.path.join(str(self.predicted_dir), 'predicted_data')
            )


if __name__ == '__main__':
    train_valid_data_dir = r"E:\LFBNet\data\remarc_default_MIP_dir/"
    train_valid_ids_path_csv = r'E:\LFBNet\data\csv\training_validation_indexs\remarc/'
    train_ids, valid_ids = get_training_and_validation_ids_from_csv(train_valid_ids_path_csv)

    trainer = NetworkTrainer(
        folder_preprocessed_train=train_valid_data_dir, folder_preprocessed_valid=train_valid_data_dir,
        ids_to_read_train=train_ids, ids_to_read_valid=valid_ids
        )
    trainer.train()
