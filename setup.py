from setuptools import setup, find_packages

setup(
   name='interiorize',
   version='0.1.0',
   author='Benjamin Idini',
   author_email='benjamin.idini@gmail.com',
   packages=find_packages(),
   url='https://github.com/bidini/interiorize',
   license='LICENSE.txt',
   description='dynamical tides of planetary bodies',
   install_requires=[
       'scipy',
       'numpy',
       'matplotlib',
       'pygyre',
       'sympy',
       'pytest',
   ],
)
