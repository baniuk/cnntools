"""Setup file for project."""

from setuptools import setup
from setuptools import find_packages
from cnntools import __version__ as ver

setup(name='cnntools',
      version=ver,
      description='Collection of tools used with CNN',
      url='http://github.com/baniuk/cnntools',
      author='Piotr Baniukiewicz',
      author_email='baniuk1@gmail.com',
      license='MIT',
      packages=find_packages(),
      install_requires=[
          'numpy',
          'keras',
          'tifffile',
      ],
      zip_safe=False)
