# ibex

This fork is a stripped down version of Brian Matejek's [original](https://github.com/bmatejek/ibex) ibex repository. It provides tools to generate skeletons from a segmentation. Additional functionalities over the original ibex include:
* Generation of skeleton edges along with nodes
* Adjacency matrix extraction
* Skeleton length calculation
* Skeleton plotting

## Installation

Clone the repository.

```git clone --recursive https://github.com/srujanm/ibex.git```

If you are using this for parallel fiber analysis in the [cerebellum](https://github.com/srujanm/cerebellum.git) repository, make sure you switch to the relevant branch. If not, proceed to the build step below.

```git checkout cereb-compat```

Locally build the transforms and skeletonization modules by running this command.

```python setup.py build_ext --inplace```