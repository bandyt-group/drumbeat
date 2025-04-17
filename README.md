# Distribution package for DRUMBEAT: Dynamically Resolved Universal Model for BayEsiAn network Tracking 



## Contents

- [Overview](#overview)
- [System Requirements](#system-requirements)
- [Installation Guide](#installation-guide)
- [Demo](#demo)
- [Instructions for use](#instructions-for-use)

# Overview

DRUMBEAT (Dynamically Resolved Universal Model for BayEsiAn network Tracking): software package that implements a temporally resolved machine learning analysis via Bayesian network. This package was prepared and released along with the publication: [Temporally Resolved and Interpretable Machine Learning Model of GPCR conformational transition]. This software builds on a previously published code (BaNDyt) which allows for Bayesian netowkr analysis of protein dynamics trajectories ([BaNDyT github](https://github.com/bandyt-group/bandyt))

# System Requirements

## Hardware Requirements

DRUMBEAT requires only a standard computer with enough RAM to support the operations defined by a user. For minimal performance, this will be a computer with about 16 GB of RAM. For optimal performance, we recommend a computer with the following specs:

RAM: 16+ GB  
CPU: 4+ cores, 2.6+ GHz/core

The runtimes associated with the demo are generated using a computer with the recommended specs (16 GB RAM, 4 cores@2.6 GHz) and internet of speed 25 Mbps.

Note: Running DRUMBEAT on large trajectory data will may require much more RAM (~100Gb) and therefore may require being run on an HPC.

## Software Requirements

The DRUMBEAT package was developed and tested on *Linux* operating systems:

Linux: Arch 6.5.3, Red 


Before setting up the DRUMBEAT  package, users should have the following python packages installed:

```
pip install numpy
pip install matplotlib
pip install networkx
pip install matplotlib.pyplot
```


# Installation Guide

No particular installation procedure is necessary. Simply import the DRUMBEAT code and as long as all packages listed above are installed, the software will work.

# Demo

For interactive demo of DRUMBEAT, please check out the the [DRUMBEAT Demo](https://github.com/bandyt-group/drumbeat/tree/main/drumbeat_demo). Within that folder there are 5 test trajectory files along with a jupyter notebook to help in running the demo. Timings for each step in the calculation are also included. 

# Instructions for Use 

Using DRUMBEAT for MD data is best outlined by looking at the jupyter notebook in the demo folder. A short outline is as follows:

1. Obtain [getContacts](https://github.com/getcontacts/getcontacts) output using trajectories of interest
2. Load the trajectories and perform any neccessary feature selection
3. Build a universal dataset using sampling/concatenation
4. Using BaNDyT software, build the universal graph via the Bayesian network
5. Scan each of the trajectories and obtain the DRUMBEAT output
6. Analyze output and extract graph measures of interest (i.e top weighted degree nodes)
