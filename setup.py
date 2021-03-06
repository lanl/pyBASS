import setuptools
from setuptools import setup

setup(name='pyBASS',
      version='0.1',
      description='Bayesian Adaptive Spline Surfaces',
      url='http://www.github.com/lanl/pyBASS',
      author='Devin Francom',
      author_email='',
      license='BSD-3',
      packages=setuptools.find_packages(),
      zip_safe=False,
      python_requires='>=3.6',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy',
          'pathos'
      ]
      )