from setuptools import setup, find_packages

setup(
   name='interiorize',
   version='0.1.0',
   author='Benjamin Idini',
   author_email='bidiniza@caltech.edu',
   packages=find_packages(),
   url='https://github.com/bidini/interiorize',
   license='LICENSE.txt',
   description='1D axisymmetric profiles of planets',
   install_requires=[
       'scipy',
       'numpy',
       'matplotlib',
       'pygyre',
       'sympy',
       'pytest',
   ],
)
