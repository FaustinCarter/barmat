from setuptools import setup, find_packages
import os

version_file = open(os.path.join('.', 'VERSION.txt'))
version_number = version_file.read().strip()
version_file.close()

setup(
    name = 'barmat',
    description = 'Mattis-Bardeen surface impedance calculator.',
    version = version_number,
    author = 'Faustin Carter',
    author_email = 'faustin.carter@gmail.com',
    license = 'MIT',
    url = 'http://github.com/FaustinCarter/barmat',
    packages = ['barmat'],
    long_description = open('README.rst').read(),
    install_requires = [
        'numpy',
        'scipy',
        'numba',
        'joblib',
    ],
    classifiers = [
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python',
        'Topic :: Scientific/Engineering :: Physics'
    ]

)
