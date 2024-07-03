barmat: the surface impedance calculator you never knew you always wanted
=========================================================================

Some functions for calculating Mattis-Bardeen surface impedance at different
temperatures and frequencies.

See Example.ipynb for an example!

Installation:
-------------
At some point this will be installable from PyPI and conda-forge, but for now just clone it and
install it in editable mode.

Step 0: (Optional if using conda) Create a conda env with everything you need to run the example notebook::

  conda create -n barmat numpy scipy numba joblib matplotlib seaborn jupyter
  conda activate barmat

Step 1: Copy the code to your machine::

  git clone https://github.com/FaustinCarter/barmat.git

Step 2: Install locally with::

  pip install -e /path/where/you/cloned/barmat

License:
--------
barmat is licensed under the MIT license, which means you can basically do
whatever you like as long as you include the license and copyright!
