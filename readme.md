# Trex 
**Tr**usted **Ex**perts

## Overview
Trex is a Decision Rule Learner based on a ensemble learning.
It uses highly customizable `HDTree` as a base learner and `RuleLoE` for managing Decision rules. 

## Features
- Decision Rule Learner
- Adjustable Rule System with custom constraints using `HDTree` (check below)
- Adjustable Rule complexity 
  - Amount and complexity of rules can be steered easily
- Optional rule pruning
  - Simplifying rules without compromising accuracy
- Support for categorical and numerical data
  - Support for missing values may be added soon
- Easy to use API
  - Installation may be streamlined soon

## Installation
Checkout this repository and the related repositories. The related repositories must be deployed into the Python Path.
When all packages are in the Python path, you can install the package by running:

```pip install -e .```
This installs the package in `editable` mode, so it will not actually be copied into your environment but linked instead.

## Usage
See `notebooks` for examples on how to use the package.


## Related Repositories
- [HDTree](https://github.com/Mereep/HDTree)
- [RuleLoE](https://github.com/Mereep/rule_loe)
- [LoE](https://github.com/Mereep/loe)
