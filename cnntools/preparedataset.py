"""
Prepare training and testing data-pairs for use with CNN.

The purpose of :class:`PrepareDataSets` is to randomly split specified dataset into two exclusive subsets containing
training pairs and testing pairs. The input and output image within each pair is defined by common **core_name** and
different **suffix**, e.g.:

- **file_1.png** - can be input image
- **file_2.png** - can be output image
- **file_3.png** - can be other input

In above example *_1.png* and *_2.png* stand for unique suffixes used for specifying input and output image in pair. This
module expects certain folder structure under *root* folder. The *root* has to be specified by user on creation of
:class:`PrepareDataSets` object. Expected structure is as follows:

* *root*
    * *raw* - contain all available training data, not necessarily training pairs. Crucial is uniform file naming.
    * *train* - here randomly selected training pairs from *raw* wil be copied by :class:`PrepareDataSets`
    * *test* - here randomly selected testing pairs will be copied by :class:`PrepareDataSets`

At the end module produces *.npz* files in *root* folder with content of *train* and *test* folders (for sake of speed
up of loading). Input and output images are saved to separate files (in total 4 files are created, each two for train
and testing datasets). Additionally, these files contain filenames of images they were produced from in order they
appear in :obj:`numpy` arrays.

Warning:
    1. *train* and *test* folders are deleted on each run of module.
    2. Images are saved as *npz* files without processing as [sample height width] arrays.
    3. Images are read in alphabetical order, note that if images names contain index at the end, it must have the same
       number of characters in each name to be sorted properly.

Example:
    The module can be called from :func:__main__ or from API. Calling from command line can look like follows:

    .. code-block:: sh

        python preparedataset.py 'in_suffix' 'target_suffix' percent_of_train rescale_factor

    Where:
        * in_suffix - is suffix of images in *raw* folder used as inputs
        * target_suffix - is suffix of images in *raw* folder used as outputs
        * percent_of_train - percent of images in *raw* folder to be copied to *train* folder, remaining images will
          be copied to *test* folder
        * rescale_factor - image rescale factor, 1.0 to not rescale.

    Preparation of datasets, access from API:

    .. code-block:: python

        from preparedataset import PrepareDataSets
        data_path = 'training-data'
        ob = PrepareDataSets(data_path)
        ob.split_random('_1.png', '_2.png', 0.8)
        ob.create_test_data('_2.png', '_1.png') # will save npz file
        ob.create_train_data('_2.png', '_1.png') # will save npz file

    Then saved *npz* files can be easelly loaded in other part of code:

    .. code-block:: python

        from preparedataset import PrepareDataSets
        data_path = 'training-data'
        ob = PrepareDataSets(data_path)
        imgs_in_train, imgs_out_train = ob.load_train_data()
        imgs_in_test, imgs_out_test = ob.load_test_data()
        # normalise, except _out, they are in range 0-1 and they are binary
        imgs_out_train = imgs_out_train[..., np.newaxis].astype('float32') / 255
        imgs_in_train = tools.normEach(imgs_in_train)[..., np.newaxis]
        imgs_out_test = imgs_out_test[..., np.newaxis].astype('float32') / 255
        imgs_in_test = tools.normEach(imgs_in_test)[..., np.newaxis]
"""
# TODO Update examples after supporting name saving

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
    Main class forming functionality of module :mod:`preparedataset`.

    Read documentation of :mod:`preparedataset` for details.

    Args:
        data_path (str):    path to *root* folder with sub-folders structure as described in :mod:`preparedataset`
        image_rows (int, optional):   height of the image
        image_cols (int, optional):   width of the image
        out_prefix (str, optionsl):   prefix added to ouput *npz* files.
    """

    def __init__(self,
                 data_path,  # Root folder of data, raw, train, test expected here
                 image_rows=256,
                 image_cols=256,
                 out_prefix='imgs'):
        """Main constructor. Create :class:`PrepareDataSets` object.

        See :class:`PrepareDataSets` for parameters description.
        """
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
        self.in_train_name = out_prefix + '_train.npz'
        # name of numpy file for training targets
        self.out_train_name = out_prefix + '_mask_train.npz'
        # name of numpy file for test inputs
        self.in_test_name = out_prefix + '_test.npz'
        # name of numpy file for test targets
        self.out_test_name = out_prefix + '_mask_test.npz'

    def load_images_from_folder(self,
                                suffix,
                                from_folder,
                                rescale_factor=None):
        """
        Load images with specified suffix (can be only extension) from specified folder with optional scaling.

        If suffix is list it is expected to be list of names of files to be loaded ``from_folder``.

        Args:
            suffix (str):   suffix, can be extension only or list of files to be loaded
            from_folder (str):  folder within *root* folder (:class'PrepareDataSets) to load files from
            rescale_factor (float, optional): rescale factor (0-1), :obj:`None` to skip

        Returns:
            (tuple): tuple containing:

                - images (:obj:`numpy.array`):   Array of [sample, height, width] with loaded images
                - names (str):    List of names of loaded images in order they appear in returned :obj:`numpy`
        """
        if isinstance(suffix, (list,)):
            in_images = [os.path.join(from_folder, x) for x in suffix]
        else:
            in_images = glob.glob(os.path.join(from_folder, "*" + suffix))
            in_images.sort()
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
        Load training pairs from given folder into numpy arrays.

        It assuems input and target (segmented) images in ``from_folder`` folder. Returned input-output pairs match
        indexes in output arrays where particular images are.

        Args:
            in_suffix (str):    suffix for input images with extension (e.g. _input.png for image_1_input.png)
            out_suffix (str):   suffix for target images with extension (e.g. _output.png for image_1_output.png)
                                If set to :obj:`None`, only images defined by in_suffix are loaded and out_img_train is
                                returned :obj:`None`
            rescale_factor (float): rescale factor (0-1), :obj:`None` to skip

        Returns:
            (tuple): tuple containing:

                - in_img_train (:obj:`numpy.array`):    Array of [num, x, y] with input images from train dataset
                                                        (in_suffix)
                - out_img_train (:obj:`numpy.array`):   Array of [num, x, y] with target images from train dataset
                                                        (out_suffix)
                - in_names (:obj:list):                 List of names of files that forms ``in_img_train`` array
                - out_names (:obj:list):                List of names of files that forms ``out_img_train`` array
        """
        img, names = self.load_images_from_folder(in_suffix, from_folder, rescale_factor)
        img_mask = None
        if out_suffix:
            # load out_images in the same order, generate out names from in names
            names_out = [s.replace(in_suffix, out_suffix) for s in names]
            img_mask, _ = self.load_images_from_folder(names_out, from_folder, rescale_factor)
            if not img.shape == img_mask.shape:
                raise ValueError("Number of input and output images does not agree")

        return img, img_mask, names, names_out

    def create_train_data(self,
                          in_suffix,
                          out_suffix,
                          rescale_factor=None):
        """
        Load training images from *train* sub-folder into :obj:`numpy.array` array and save it under prefixed name.

        Save *prefix_train.npz* and *prefix_mask_train.npz* files infolder specified in ::class::PrepareDataSets.
        File *prefix_train.npz* contains images with ``in_suffix`` whereas *prefix_mask_train.npz* those with
        ``out_suffix``. Both files contain two arrays, data and file names, both under keys **data** and **names**
        respectivelly.

        Args:
            in_suffix (str):    suffix for input images with extension (e.g. _1.png for image_1.png)
            out_suffix (str):   suffix for target images with extension (e.g. _2.png for image_2.png),
                                can be :obj:`None`
            rescale_factor (float): rescale factor (0-1). :obj:`None` to skip rescaling
        """
        imgs, imgs_mask, names, names_out = self._load_training_pairs(
            in_suffix, out_suffix, self.train_path, rescale_factor)
        if imgs is not None:
            np.savez_compressed(os.path.join(self.data_path, self.in_train_name), data=imgs, names=names)
            print('Saving to {0} files done.'.format(os.path.join(self.data_path, self.in_train_name)))
            if imgs_mask is not None:
                np.savez_compressed(os.path.join(self.data_path, self.out_train_name), data=imgs_mask, names=names_out)
                print('Saving to {0} files done.'.format(os.path.join(self.data_path, self.out_train_name)))
        else:
            print("Train data not saved - no images")

    def load_train_data(self):
        """
        Load training data files saved by :func:`create_train_data` function.

        Returns:
            (tuple): tuple containing:

                - in_img_train (:obj:`numpy.array`):    Array of [num, x, y] with input images from *train* folder
                - out_img_train (:obj:`numpy.array`):   Array of [num, x, y] with target images from *train* folder

        Note:
            Note that returned :obj:`numpy.array` arrays are in the same format as images saved on disk. Perhaps further
            scaling is necessary. Check :mod:`preparedataset` description.
        """
        try:
            imgs_train = np.load(os.path.join(self.data_path, self.in_train_name))
        except FileNotFoundError:
            print("File", self.in_train_name, "not found")
            imgs_train = None
        try:
            imgs_mask_train = np.load(os.path.join(self.data_path, self.out_train_name))
        except FileNotFoundError:
            print("File", self.out_train_name, "not found")
            imgs_mask_train = None
        return imgs_train, imgs_mask_train

    def create_test_data(self,
                         in_suffix,
                         out_suffix,
                         rescale_factor=None):
        """
        Load testing images from *test* sub-folder into numpy array and save it under prefixed name.

        Save *prefix_test.npz* and *prefix_mask_test.npz* files infolder specified in ::class::PrepareDataSets.
        File *prefix_test.npz* contains images with ``in_suffix`` whereas *prefix_mask_test.npz* those with
        ``out_suffix``. Both files contain two arrays, data and file names, both under keys **data** and **names**
        respectivelly.

        Args:
            in_suffix (str):    suffix for input images with extension (e.g. _1.png for image_1.png)
            out_suffix (str):   suffix for target images with extension (e.g. _2.png for image_2.png)
            rescale_factor (float): rescale factor (0-1). :obj:`None` to skip rescaling
        """
        imgs, imgs_mask, names, names_out = self._load_training_pairs(
            in_suffix, out_suffix, self.test_path, rescale_factor)

        if imgs is not None:
            np.savez_compressed(os.path.join(self.data_path, self.in_test_name), data=imgs, names=names)
            print('Saving to {0} files done.'.format(os.path.join(self.data_path, self.in_test_name)))
            if imgs_mask is not None:
                np.savez_compressed(os.path.join(self.data_path, self.out_test_name), data=imgs_mask, names=names_out)
                print('Saving to {0} files done.'.format(os.path.join(self.data_path, self.out_test_name)))
        else:
            print("Train data not saved - no images")

    def load_test_data(self):
        """
        Load testing data files saved by :func:`create_test_data` function.

        Returns:
            (tuple): tuple containing:

                - in_img_train (:obj:`numpy.array`):    Array of [num, x, y] with input images from *train* folder
                - out_img_train (:obj:`numpy.array`):   Array of [num, x, y] with target images from *train* folder

        Note:
            Note that returned :obj:`numpy.array` arrays are in the same format as images saved on disk. Perhaps further
            scaling is necessary. Check :mod:`preparedataset` description.
        """
        try:
            imgs_test = np.load(os.path.join(self.data_path, self.in_test_name))
        except FileNotFoundError:
            print("File", self.in_test_name, "not found")
            imgs_test = None
        try:
            imgs_mask_test = np.load(os.path.join(self.data_path, self.out_test_name))
        except FileNotFoundError:
            print("File", self.out_test_name, "not found")
            imgs_mask_test = None
        # TODO return https://stackoverflow.com/questions/354883/how-do-you-return-multiple-values-in-python
        return imgs_test, imgs_mask_test

    def split_random(self,
                     in_suffix,
                     out_suffix,
                     percent):
        """
        Split training pairs from *raw* folder to two exclusive groups, and copy them to *train* and *test* folder.

        Content of *raw* folder is not modified. This method finds common base in file names and copies all files
        with the same base. Therefore, *file_1.png*, *file_2.png*, *file_3.png* will be always copied together assuming
        that they relate to the same frame but differ in presentation (e.g. DIC, Mask, Fluorescent).

        Args:
            in_suffix (str):    suffix for input images with extension (e.g. _1.png for image_1.png)
            out_suffix (str):   suffix for target images with extension (e.g. _2.png for image_2.png). Can be :obj:`None`
            percent (float):    percent (0-1) of images copied to test folder
        """
        in_images = glob.glob(os.path.join(self.raw_path, "*" + in_suffix))
        total = len(in_images)
        print("Found {0} images in {1}".format(total, self.raw_path))
        random.seed()
        # split input images to two groups
        testing_in = random.sample(in_images, round(total * percent))  # selected for tests
        training_in = [x for x in in_images if x not in testing_in]  # remaining (without test)
        # generate also output file names from splited inputs
        if out_suffix is not None:
            testing_out = [s.replace(in_suffix, out_suffix) for s in testing_in]
            training_out = [s.replace(in_suffix, out_suffix) for s in training_in]
        else:
            testing_out = [None] * len(testing_in)
            training_out = [None] * len(training_in)  # fake array of out data, must be same length as in
        # copy images from raw folder to test and train folders
        if os.path.isdir(self.train_path):
            shutil.rmtree(self.train_path)
        os.makedirs(self.train_path)
        if os.path.isdir(self.test_path):
            shutil.rmtree(self.test_path)
        os.makedirs(self.test_path)
        for (ti, to) in zip(testing_in, testing_out):
            shutil.copy(ti, self.test_path)
            if to:  # if not None
                shutil.copy(to, self.test_path)
        for (ti, to) in zip(training_in, training_out):
            shutil.copy(ti, self.train_path)
            if to:
                shutil.copy(to, self.train_path)
        print("Number of training and testing pairs: {0}/{1}".format(len(training_in), len(testing_in)))


if __name__ == '__main__':
    """
    Expect four parameters: in_prefix test_prefix percent (0-1) scale.

    See:
        :mod:'preparedataset'
    """
    args = sys.argv
    if len(args) is not 5:
        raise ValueError("Need 4 parameters")
    ob = PrepareDataSets('training-data')
    ob.split_random(args[1], args[2], float(args[3]))
    ob.create_train_data(args[1], args[2], float(args[4]))
    ob.create_test_data(args[1], args[2], float(args[4]))
