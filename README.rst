barmat: the surface impedance calculator you never knew you always wanted
=========================================================================

Some functions for calculating Mattis-Bardeen surface impedance at different
temperatures and frequencies.

See Example.ipynb for an example!

Installation:
-------------
At some point this will be installable from PyPI, but for now just clone it and
install it in editable mode.

Step 1: Copy the code to your machine::

  git clone https://github.com/FaustinCarter/barmat.git

Step 2: Install locally with::

  pip install -e /path/where/you/cloned/barmat

Known Issues:
-------------
There may be issues with running barmat on Windows. This will definitely be
true if you are using Anaconda on Windows. To fix the problem, uninstall numpy
and scipy (on conda use `conda remove --force` to keep things that depend on
numpy and scipy from getting removed also). Then reinstall numpy and scipy (in
that order) from http://www.lfd.uci.edu/%7Egohlke/pythonlibs/. Barmat should
work fine after that!

License:
--------
barmat is licensed under the MIT license, which means you can basically do
whatever you like as long as you include the license and copyright!
