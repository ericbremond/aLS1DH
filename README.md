# aLS1DH
The adaptative *linearly scaled one-parameter double-hybrid* model.

## Description
aLS1DH is the first adaptive double-hybrid model to combine the nonempirical LS1DH framework with machine learning, thereby establishing a physically motivated, system-specific approach to double-hybrid density-functional approximations.

## Installation

### Requirements

The code has been developed for Python 3.9 and uses the Conda package manager.

### 1. Clone the repository

```
git clone https://github.com/ericbremond/aLS1DH.git
cd aLS1DH
```

### 2. Create the Conda environment

```
conda env create -f aLS1DH.yml
```

### 3. Activate the environment

```
conda activate aLS1DH
```

The environment is now ready to use.

## Usage

### Running the program

The program takes one or several ORCA input files (.inp) as command-line arguments.

For example:
```
python aLS1DH.py water.inp
```

The program reads the molecular geometry and charge from the ORCA input file, generates the corresponding molecular descriptors and uses the pretrained KRR model to predict the aLS1DH parameters.

A new ORCA input file is then generated.

### Input file format

The input must be an ORCA input file containing the molecular geometry in XYZ format.

For example, a water molecule:

```
  ! PBE0-DH def2-QZVPP RIJCOSX AutoAux TightSCF DEFGRID3
  
  *xyz 0 1
  O       0.000000    0.000000    0.000000
  H       0.000000    0.757000    0.586000
  H       0.000000   -0.757000    0.586000
  end
```

The first two numbers following *xyz correspond to the total molecular charge and spin multiplicity, respectively.

### Output
For an input file named:

```
water.inp
```

the program generates:

```
water_aLS1DH.inp
```

The generated file contains the predicted aLS1DH functional parameters and the molecular geometry.

### Processing multiple molecules

Several input files can be processed in a single execution:

```
python aLS1DH.py molecule1.inp molecule2.inp molecule3.inp
```

A corresponding output file will be generated for each molecular system.

## Machine-learning model

The aLS1DH parameter prediction relies on a pretrained [Kernel Ridge Regression](https://github.com/dkhan42/qml2) model.

The model is stored in:

```
KRR_qm5_qm9.npz
```

This file is required to run the program and is included in the repository.

The molecular descriptors are generated using the [cMBDF](https://github.com/dkhan42/cMBDF) representation.

The training and test datasets are also provided in the repository.

## References

```
@article{
doi:10.26434/chemrxiv.15012736/v1,
author = {Hong Nhung Vu  and Angel Jose Pérez-Jiménez  and Juan Carlos Sancho-García  and Carlo Adamo  and Eric Brémond },
title = {Nonempirical Double-hybrid Functionals Entering the AI Field: the Adaptative LS1DH model},
journal = {ChemRxiv},
volume = {2026},
number = {1124},
pages = {},
year = {2026},
doi = {10.26434/chemrxiv.15012736/v1},
URL = {https://chemrxiv.org/doi/abs/10.26434/chemrxiv.15012736/v1},
eprint = {https://chemrxiv.org/doi/pdf/10.26434/chemrxiv.15012736/v1},
abstract = {We introduce a machine-learning-assisted nonempirical double-hybrid density-functional approximation, dubbed adaptative LS1DH or aLS1DH, which combines the theoretical constraints of the linearly scaled one-parameter double-hybrid (LS1DH) model with system-specific flexibility. Based on kernel ridge regression (KRR) and convolutional many-body distribution function (cMBDF) representations, aLS1DH predicts optimal system-dependent fractions of nonlocal exchange and correlation while preserving size consistency. Its performance is assessed on the extended GMTKN55 and on the recently developed CYCLO70. The aLS1DH model consistently improves upon its parent functional, PBE0-DH, rivaling the latest machine-learning-based functionals such as skala functional. These results highlight the potential of machine learning to improve the accuracy of nonempirical double hybrids while retaining their underlying theoretical constraints.}}
```


Additional references concerning the [cMBDF](https://github.com/dkhan42/cMBDF) molecular representation and [Kernel Ridge Regression](https://github.com/dkhan42/qml2) should also be acknowledged where appropriate.


