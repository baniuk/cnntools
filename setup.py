"""Setuptools."""

from setuptools import setup

setup(name='cnntools',
      version='0.1',
      description='Collection of tools used with CNN',
      url='http://github.com/baniuk/cnntools',
      author='Piotr Baniukiewicz',
      author_email='baniuk1@gmail.com',
      license='MIT',
      packages=['cnntools'],
      install_requires=[
          'numpy',
          'keras',
      ],
      zip_safe=False)
