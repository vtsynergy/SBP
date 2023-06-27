# Accelerated Stochastic Block Partitioning

Despite the title of the repository, to date there is no GPU code in this repository (though it is planned for the future).

Stochastic block partitioning (SBP) code based on the reference implementation in the [Graph Challenge](http://graphchallenge.org)

Sequential, shared memory parallel, and distributed memory, multi-node SBP code is in src/main.cpp.

A C++ translation of the divide-and-conquer approach based on [iHeartGraph's implementation](https://github.com/iHeartGraph/GraphChallenge) and [Scalable Stochastic Block Partition paper](https://ieeexplore.ieee.org/document/8091050) is found in src/DivideAndConquerSBP.cpp.
