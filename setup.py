import os
from setuptools import setup, find_packages


# Utility function to read the README file.
def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name='sps4lat',
    version='0.0.0',
    author=('Benjamin Beringue', 'Davide Poletti'),
    author_email=('bb510@cam.ac.uk', 'davide.pole@gmail.com'),
    description='Semi-Parametric Component Separation for the LAT',
    license='GPLv3',
    keywords='',
    url='https://github.com/simonsobs/sps4lat',
    packages=find_packages(),
    long_description=read('README.rst'),
    install_requires=[
        'numpy', 'healpy'],
)
