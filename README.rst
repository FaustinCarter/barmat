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
There is something weird about running barmat under Windows 10 (and maybe other
versions). In Windows 10 with Anaconda, huge floating point errors have been
noted. However, running within Ubuntu Bash on the Windows 10 Subystem for Linux
with Anaconda 2 presents no problems. OSX is also fine. See Issue #1 in the
github tracker.

License:
--------
barmat is licensed under the MIT license, which means you can basically do
whatever you like as long as you include the license and copyright!
