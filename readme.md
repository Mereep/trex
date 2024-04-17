# Trex

**Tr**usted **Ex**perts

## Overview

Trex is a Decision Rule Learner based on a ensemble learning build for explainability (`xAI`).
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
This installs the package in `editable` mode, so it will not actually be copied into your environment but linked
instead.

## Usage

See `notebooks` for examples on how to use the package.

# How does Trex work?

Trex is an ensemble learner, that means it build multiple models for learning your data.
There is a bunch of ensemble algorithms. Well known instances are
Random Forests or Gradient Boosting. However, those are hare hard to interpret as they generally train a huge amount of
models within their ensemble.
Even if these are interpretable on their own, the decisions are based by combining the ensemble members, which is hard
to interpret. 
<br /><br /> 
Trex uses a different approach. It builds models after each other (called `Expert`) and will only use **one** model later to predict data.
<br /> <br />

**Algorithm outline**
<br /><br />
Training:
- Each model is trained on the errors of the previous model, until each data point is classified correctly (or the maximum
amount of desired models is reached).  So each model becomes an expert of a different subset of the data.
- Another set of models is trained (called `assignment models`) which learn on which subset each model is an expert in.
- An rule system (based on `RuleLoE`) is built which combines the experts and their assignment trees
  - This is based on the path `assignment models -> experts -> decision`.

Prediction:
  - The resulting decision rule system is searched for the best fitting entry (most conditions are met) and the corresponding decision is returned

## Related Repositories

- [HDTree](https://github.com/Mereep/HDTree)
- [RuleLoE](https://github.com/Mereep/rule_loe)
- [LoE](https://github.com/Mereep/loe)
