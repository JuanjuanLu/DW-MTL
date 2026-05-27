# Dynamic Weighted Multi-Task Learning for Regulatory Toxicity Value Prediction

This repository provides an implementation of a dynamic weighted multi-task learning (DW-MTL) framework for predicting regulatory toxicity values (RTVs). 

## Overview

Regulatory toxicity values are essential quantitative benchmarks for human health risk assessment, but experimentally or expert-derived RTVs are available for only a limited number of chemicals. Multi-task learning can improve predictive performance by transferring information across related endpoints. However, conventional multi-task learning may suffer from negative transfer when different tasks have unequal data quality, different learning difficulty, or weak mechanistic relatedness.

This project introduces a dynamic weighting strategy to prioritize main RTV prediction tasks while still allowing auxiliary toxicological or pharmacokinetic tasks to contribute useful information during training.

## Model Concept

The DW-MTL framework consists of the following major components:

1. **Molecular graph input**
   - Molecules are represented as graph structures.
   - Atoms are treated as nodes and chemical bonds are treated as edges.
   - Atom and bond features are used as model inputs.

2. **RGCN-based molecular feature extraction**
   - A relational graph convolutional network (RGCN) is used to learn molecular representations from graph-structured data.
   - The model captures atom-level and bond-type-dependent structural information.

3. **Multi-task prediction heads**
   - The model predicts multiple toxicity-related endpoints in a unified framework.
   - In the non-cancer setting, the selected tasks include:
     - CYP1A2
     - CYP2C9
     - CYP2C19
     - CYP2D6
     - CYP3A4
     - RfD
     - RfC
     - BMD
   
   - In the cancer setting, the selected tasks include:
     - APO
     - ELM
     - EPM
     - GIN
     - OXP
     - PRO
     - OSF
     - ISF
     - IUR

4. **Dynamic task weighting**
   - Prior weights are assigned to important main tasks.
   - Task weights are dynamically adjusted during training according to task-specific learning status.
   - This strategy helps the model focus on difficult or under-optimized main tasks while reducing the risk of negative transfer.

## Citation

If this code is used in academic work, please cite the corresponding study or repository once available.

## requirements：
python 3.6
anaconda
dgl 0.4.3
xgboost
rdkit
pytorch
sklearn