=================================================================
:scissors: Distributed Deep Learning Simulator :desktop_computer:
=================================================================

--------------------------------------------------------------------------------------------------------------------------------------------

Implementation of PAC-ML (partitioning for asynchronous computing with machine
learning) and the associated distributed deep learning simulation of a RAMP optical architecture 
as reported in `Partitioning Distributed Compute Jobs with Reinforcement Learning and Graph Neural Networks <https://arxiv.org/abs/2205.14345>`_.

.. figure:: assets/rl_gnn_partitioning_methodology.drawio.png

--------------------------------------------------------------------------------------------------------------------------------------------


Setup
=====

Open your command line. Change the current working directory to the location where you want to clone this project, and run::

    $ git clone https://github.com/cwfparsonson/ddls

In the project's root directory, run::

    $ python setup.py install

Then, still in the root directory, install the required packages with conda (env name defined at top of .yaml file)::

    $ conda env create -f requirements/environment.yaml


Re-Running the Paper's Experiments
==================================
TODO


Citing this work
================
If you find this project or the associated paper useful, please cite our work::

    article{parsonson2023reinforcement,
          title = {Partitioning Distributed Compute Jobs with Reinforcement Learning and Graph Neural Networks},
          author = {Parsonson, Christopher W. F. and Shabka, Zacharaya and Ottino, Alessandro and Zervas, Georgios},
          journal = {arXiv},
          year = {2023}
        }
