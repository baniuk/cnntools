"""
Load training and test pairs.

Usage:
    From command line to produce numpy files:
    preparedataset.py 'in_prefix' 'target_prefix' percent_of_train rescale_factor
    - percent_of_train can be 0
    - images are not scaled if rescale factor is 1.0

    From API
    Use load_train_data and load_test_data to load previously saved numpy arrays (saves time)

    split_random - splits dataset (input-output pairs) from raw_path folder to test_path folder and train_path
    create_train_data - scan train_path and create separate numpy arrays from input and output images
    create_test_data - scan test_path and create separate numpy arrays from input and output images

    _load_images_from_folder - loads images according to suffix from specified folder
    _load_training_pairs - same as above but scan for input and output images

    Module expects folder structure:
    training-data
        raw
        train
        test

    Output files are saved in training-data

"""

import os
import sys
import numpy as np

from skimage.io import imread
from skimage.transform import rescale
import random
import shutil
import glob


class PrepareDataSets:
    """
    This class provides methods for reading separate images into numpy arrays.

    It splits large dataset into testing and training subsets. Module assumes that all images exist in 'raw_path'
    sub-folder and they follow certain naming convention like 'CORE_XXX.ext', where CORE is common for all images and
    XXX.ext denotes input and target image.
    """

    def __init__(self,
                 data_path,  # Root folder of data, raw, train, test expected here
                 image_rows=256,
                 image_cols=256,
                 out_prefix='imgs'):
        self.data_path = data_path
        self.image_rows = image_rows
        self.image_cols = image_cols
        self.out_prefix = out_prefix
        # path where all training pairs sit (input and output images)
        self.raw_path = os.path.join(data_path, 'raw')
        # pairs selected to be used during training
        self.train_path = os.path.join(data_path, 'train')
        # pairs selected to be used for tests
        self.test_path = os.path.join(data_path, 'test')
        # name of numpy file for training inputs
        self.in_train_name = out_prefix + '_train.npy'
        # name of numpy file for training targets
        self.out_train_name = out_prefix + '_mask_train.npy'
        # name of numpy file for test inputs
        self.in_test_name = out_prefix + '_test.npy'
        # name of numpy file for test targets
        self.out_test_name = out_prefix + '_mask_test.npy'

    def _load_images_from_folder(self,
                                 suffix,
                                 from_folder,
                                 rescale_factor=None):
        """
        Load images with specified suffix (can be only extension) from specified folder with optional scaling.

        If suffix is list it is expected to be list of files to load frm_folder.

        Args:
            suffix :            {str} - suffix, can be extension only or alternatively it can be list of files to load
            rescale_factor :    {float} - rescale factor (0-1), None to skip

        Return:
            img :   {numpy} - Array of [num, x, y] with loaded images
            names : {str}   - List of names of loaded images in order they appear in img
        """
        if isinstance(suffix, (list,)):
            in_images = [os.path.join(from_folder, x) for x in suffix]
        else:
            in_images = glob.glob(os.path.join(from_folder, "*" + suffix))
        total = int(len(in_images))
        if total == 0:
            print("No images found in ", from_folder)
            return None

        img = imread(in_images[0], as_grey=True)
        if rescale_factor:
            self.image_rows = (int)(img.shape[0] * rescale_factor)
            self.image_cols = (int)(img.shape[1] * rescale_factor)

        imgs = np.ndarray((total, self.image_rows, self.image_cols), dtype=np.uint8)

        i = 0
        print('-' * 30)
        if isinstance(suffix, (list,)):
            print('Collecting images from {0} ...'.format('list: ' + suffix[0]))
        else:
            print('Collecting images {0} ...'.format(suffix))
        print('-' * 30)
        for in_image_name in in_images:
            # in_images and out_images are not in the same order - regenerate out from in
            in_img = imread(in_image_name, as_grey=True)
            if rescale_factor:
                in_img = rescale(in_img, rescale_factor, preserve_range=True, mode='reflect')
            in_img = np.array([in_img])
            imgs[i] = in_img
            if i % 100 == 0:
                print('Done: {0}/{1} images'.format(i, total))
            i += 1
        print('Loading of {0} images done. Output image size is [{1},{2}].'.format(i, self.image_rows, self.image_cols))
        return imgs, [os.path.split(x)[1] for x in in_images]

    def _load_training_pairs(self,
                             in_suffix,
                             out_suffix,
                             from_folder,
                             rescale_factor=None):
        """
        Load training pairs from given folder into numpy array.

        It assuems input and target (segmented) images in from_folder folder. Returned input-output pairs match
        indexes in output arrays where particular images are.

        Args:
            in_suffix :         {str} - suffix for input images with extension (e.g. _input.png for image_1_input.png)
            out_suffix :        {str} - suffix for target images with extension (e.g. _output.png for image_1_output.png)
                                        If set to None, only images defined by in_suffix are loaded and out_img_train is
                                        returned None
            rescale_factor :    {float} - rescale factor, None to skip

        Return:
            in_img_train :   {numpy} - Array of [num, x, y] with input images from train dataset (in_suffix)
            out_img_train :  {numpy} - Array of [num, x, y] with target images from train dataset )out_suffix)
        """
        img, names = self._load_images_from_folder(in_suffix, from_folder, rescale_factor)
        if out_suffix:
            # load out_images in the same order, generate out names from in names
            names_out = [s.replace(in_suffix, out_suffix) for s in names]
            img_mask, _ = self._load_images_from_folder(names_out, from_folder, rescale_factor)
            if not img.shape == img_mask.shape:
                raise ValueError("Number of input and output images does not agree")

        return img, img_mask

    def create_train_data(self,
                          in_suffix,
                          out_suffix,
                          rescale_factor=None):
        """
        Load training data images from train_path into numpy array and save it.

        Assuems input and output (segmented) images in train_path folder. Save imgs_train.npy and imgs_mask_train.npy files
        in current folder. File imgs_train.npy contains images with suffix in_suffix and imgs_mask_train.npy those with
        out_suffix.

        Args:
            in_suffix :         {str} - suffix for input images with extension (e.g. _1.png for image_1.png)
            out_suffix :        {str} - suffix for target images with extension (e.g. _2.png for image_2.png)
            rescale_factor :    {float} - rescale factor

        Return:
            Create numpy files.

        """
        imgs, imgs_mask = self._load_training_pairs(in_suffix, out_suffix, self.train_path, rescale_factor)
        if imgs is not None:
            np.save(os.path.join(self.data_path, self.in_train_name), imgs)
            print('Saving to {0} files done.'.format(os.path.join(self.data_path, self.in_train_name)))
            if imgs_mask is not None:
                np.save(os.path.join(self.data_path, self.out_train_name), imgs_mask)
                print('Saving to {0} files done.'.format(os.path.join(self.data_path, self.out_train_name)))
        else:
            print("Train data not saved - no images")

    def load_train_data(self):
        """
        Load trainging data saved by create_train_data() function.

        Training dataset contains pairs of input and target images.

        Return:
            in_img_train :   {numpy} - Array of [num, x, y] with input images from train dataset
            out_img_train :  {numpy} - Array of [num, x, y] with target images from train dataset
        """
        imgs_train = np.load(os.path.join(self.data_path, self.in_train_name))
        imgs_mask_train = np.load(os.path.join(self.data_path, self.out_train_name))
        return imgs_train, imgs_mask_train

    def create_test_data(self,
                         in_suffix,
                         out_suffix,
                         rescale_factor=None):
        """
        Load test data images from test_path into numpy array and save it.

        Assuems input and output (segmented) images in train_path folder. Save imgs_train.npy and imgs_mask_train.npy files
        in current folder. File imgs_test.npy contains images with suffix in_suffix and imgs_mask_test.npy those with
        out_suffix.

        Args:
            in_suffix :         {str} - suffix for input images with extension (e.g. _1.png for image_1.png)
            out_suffix :        {str} - suffix for target images with extension (e.g. _2.png for image_2.png)
            rescale_factor :    {float} - rescale factor

        """
        imgs, imgs_mask = self._load_training_pairs(in_suffix, out_suffix, self.test_path, rescale_factor)

        if imgs is not None:
            np.save(os.path.join(self.data_path, self.in_test_name), imgs)
            print('Saving to {0} files done.'.format(os.path.join(self.data_path, self.in_test_name)))
            if imgs_mask is not None:
                np.save(os.path.join(self.data_path, self.out_test_name), imgs_mask)
                print('Saving to {0} files done.'.format(os.path.join(self.data_path, self.out_test_name)))
        else:
            print("Train data not saved - no images")

    def load_test_data(self):
        """
        Load test data saved by create_test_data() function.

        Test dataset contains pairs of input and target images.

        Return:
            in_img_test :   {numpy} - Array of [num, x, y] with input images from test dataset
            out_img_test :  {numpy} - Array of [num, x, y] with target images from test dataset
        """
        imgs_test = np.load(os.path.join(self.data_path, self.in_test_name))
        imgs_mask_test = np.load(os.path.join(self.data_path, self.out_test_name))
        return imgs_test, imgs_mask_test

    def split_random(self,
                     in_suffix,
                     out_suffix,
                     percent):
        """
        Split training pairs from raw_path folder to two exclusive groups, for training and testsing.

        Content of raw_path folder is not modified, pairs are copied to train_path and test_path.

        Args:
            in_suffix :     {str} - suffix for input images with extension (e.g. _1.png for image_1.png)
            out_suffix :    {str} - suffix for target images with extension (e.g. _2.png for image_2.png)
            percent :       {float} - percent (0-1) of images copied to test folder (copied are always pairs of images
                                        (in, out))
        """
        in_images = glob.glob(os.path.join(self.raw_path, "*" + in_suffix))
        total = len(in_images)
        print("Found {0} images in {1}".format(total, self.raw_path))
        random.seed()
        # split input images to two groups
        testing_in = random.sample(in_images, round(total * percent))  # selected for tests
        training_in = [x for x in in_images if x not in testing_in]  # remaining (without test)
        # generate also output file names from splited inputs
        testing_out = [s.replace(in_suffix, out_suffix) for s in testing_in]
        training_out = [s.replace(in_suffix, out_suffix) for s in training_in]
        # copy images from raw folder to test and train folders
        shutil.rmtree(self.train_path)
        os.makedirs(self.train_path)
        shutil.rmtree(self.test_path)
        os.makedirs(self.test_path)
        for (ti, to) in zip(testing_in, testing_out):
            shutil.copy(ti, self.test_path)
            shutil.copy(to, self.test_path)
        for (ti, to) in zip(training_in, training_out):
            shutil.copy(ti, self.train_path)
            shutil.copy(to, self.train_path)
        print("Number of training and testing pairs: {0}/{1}".format(len(training_in), len(testing_in)))


if __name__ == '__main__':
    """
    Expect four parameters in_prefix test_prefix percent (0-1) scale.
    """
    args = sys.argv
    if len(args) is not 5:
        raise ValueError("Need 4 parameters")
    ob = PrepareDataSets('training-data')
    ob.split_random(args[1], args[2], float(args[3]))
    ob.create_train_data(args[1], args[2], float(args[4]))
    ob.create_test_data(args[1], args[2], float(args[4]))
