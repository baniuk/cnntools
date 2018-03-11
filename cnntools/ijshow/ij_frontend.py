"""Show image using Fiji backend.

Basics
------

This package uses Fiji to display :obj:`numpy.array`. Temporary folder is used as intermediate transport medium, each
array is saved as *tiff* image before displaying and then *Fiji* is called with this file as parameter. A
reference (as file name) to displayed image is returned back to the caller allowing to re-read this image after
modification i *Fiji*. There is random folder created in system *tmp* on each new :class:`IJShow` object.
This solution is rather recommended to be used with IPython.

Usage
-----

Example:
    d



"""
# TODO Finish Examples of usage

import os
import os.path
import shutil
import logging
import subprocess as subp
import tempfile as tmp
from tifffile import imsave


class IJShow:
    """
    Construct *Fiji* based backend for showing arrays from Python.

    Read documentation of :mod:`ij_frontend` for details and examples.

    Args:
        ijpath (str):    path to ImageJ or Fiji executable
        persistent (bool, optional):   if ``True`` either temporary folder for this session or image files will not
                                       be deleted on object destruction.
    """

    ijpath = ""

    def __init__(self,
                 ijpath,
                 persistent=True):
        """Construct IJShow object."""
        self.persistent = persistent
        IJShow.ijpath = ijpath
        self.images = []
        self.tmpfolder = tmp.mkdtemp(prefix='IJShow')  # tmp folder for this session

    @staticmethod
    def runIjandShow(image_name):
        """Run Fiij with file name to open."""
        if os.path.isfile(image_name):
            subp.Popen([IJShow.ijpath, image_name],
                       shell=False, stdin=None, stdout=None, stderr=None, close_fds=True)
        else:
            logging.warning('Specified file ' + image_name + " does not exist")

    def __del__(self):
        """Clean temporary folder."""
        if not self.persistent:
            for i in self.images:
                i.valid(False)
            shutil.rmtree(self.tmpfolder)

    def __repr__(self):
        """Info string."""
        return 'Tmp folder {0}, hold {1} objects'.format(self.tmpfolder, len(self.images))

    def imshow(self, image, convert=True):
        """Show the image."""
        _, image_name = tmp.mkstemp(dir=self.tmpfolder)
        if convert:
            image = image.astype('float32')
        imsave(image_name, image)  # save under tmp folder
        ob = IJImage(image_name)  # create output object
        self.images.append(ob)  # store in internal list FIXME not shure why yet
        ob.display()  # show the image
        return ob


class IJImage:
    """Hold image object."""

    # TODO Implement: retrieve from file to numpy

    def __init__(self, image_name):
        """
        Assign file name to this object.

        Assumes that file exists.
        """
        self._image_name = image_name
        self._valid = True

    @property
    def image_name(self):
        """Return file name of the image."""
        return self._image_name

    @property
    def valid(self):
        """
        Property defining status of the object.

        Invalid file is assumed to be not existing. Invalidating always delete the file.
        """
        return self._valid

    @valid.setter
    def valid(self, isvalid):
        if not isvalid:
            os.remove(self._image_name)
        self._valid = isvalid

    def display(self):
        """Show the image if it is valid."""
        if self.valid:
            IJShow.runIjandShow(self.image_name)

    def delete(self):
        """Delete the image and invalidate it."""
        self.valid - False

    def __del__(self):
        """Delete the image and invalidate it."""
        self.valid = False

    def __repr__(self):
        """Return name of file and its status."""
        return 'Image file {0}, is valid = {1}'.format(self.image_name, self.valid)
