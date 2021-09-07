# p2p-sim
CS 765 assgn - P2P sim

## Installation
Need to have python3.7 or higher installed.

`sudo apt install python3 python3-pip` on Debian/Ubuntu machines would allow installation of python3 as well as pip(pip3).

After having installed `pip`, the python dependencies can be installed as `pip3 install -r ./requirements.txt`

## Running the code
After having installed all dependencies, the code can simply be run as `python3 p2p-sim.py` or `./p2p-sim.py` which outputs certain logging information to `stderr` and at the end of the simulation, block-dumps are created in files named `i.log` where `i` is the number referring to `i`-th node, `i` is indexed from `0`. Also the blockchain tree at the end is saved in a png image.

```
usage: p2p-sim.py [-h] [--nodes NODES] [--z Z] [--edge EDGE]
                  [--invalid INVALID] [--Ttx TTX]
                  [--sim_time SIM_TIME] [--seed SEED]
                  [--ltk LTK] [--utk UTK] [--img IMG]

optional arguments:
  -h, --help           show this help message and exit
  --nodes NODES        nodes in P2P network
  --z Z                probability of each node being fast
  --edge EDGE          probability of each edge being present or
                       absent
  --invalid INVALID    probability of each block being invalid
  --Ttx TTX            mean time for txn gen
  --sim_time SIM_TIME  maximum simulation time
  --seed SEED          seed for random functions
  --ltk LTK            lower bound on Tk
  --utk UTK            upper bound on Tk
  --img IMG            output image name, png image is
                       generated
```