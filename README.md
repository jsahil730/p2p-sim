# p2p-sim
CS 765 assgn - P2P sim

## Installation
Need to have python3.7 or higher installed.

`sudo apt install python3 python3-pip` on Debian/Ubuntu machines would allow installation of python3 as well as pip(pip3).

After having installed `pip`, the python dependencies can be installed as `pip3 install -r ./requirements.txt`

## Running the code
After having installed all dependencies, the code can simply be run as `python3 p2p-sim.py` which outputs certain logging information to `stderr` and at the end of the simulation, block-dumps are created in files named `i.log` where `i` is the number referring to `i`-th node, `i` is indexed from `0`.