# Optimization in PyTorch: From Logistic Regression to Advanced Optimizers

This repository contains a comprehensive deep-dive into the fundamental concepts of optimization in machine learning, implemented from scratch using PyTorch. The project covers the end-to-end process of building a classification pipeline, from text preprocessing to comparing state-of-the-art optimization algorithms.

## Table of Contents
1. [Overview](#overview)
2. [Dataset and Preprocessing](#dataset-and-preprocessing)
3. [Logistic Regression from Scratch](#logistic-regression-from-scratch)
4. [Numerical Stability and Loss Functions](#numerical-stability-and-loss-functions)
5. [Stochastic Gradient Descent (SGD) Pipeline](#stochastic-gradient-descent-sgd-pipeline)
6. [Regularization (L1 & L2)](#regularization-l1--l2)
7. [Advanced Optimization Algorithms](#advanced-optimization-algorithms)
8. [Experimental Results and Analysis](#experimental-results-and-analysis)

---

## Overview
The goal of this hometask is to bridge the gap between theoretical optimization concepts and practical implementation. By avoiding high-level abstractions where possible, we explore how data flows through a model, how gradients are computed and applied, and how different hyperparameters influence the trajectory of a model's learning process.

## Dataset and Preprocessing
We use the **SST-2 (Stanford Sentiment Treebank)** dataset for binary sentiment classification.
- **Cleaning**: Standardizing text (lowercase, removing noise, handling hyphens).
- **Tokenization**: Splitting text into atomic word units.
- **Feature Engineering**: Implementing a **Bag-of-Words (BoW)** representation with a fixed vocabulary of the top 10,000 most frequent tokens. This transforms variable-length text into fixed-size numerical vectors suitable for neural network input.

## Logistic Regression from Scratch
A custom `LogisticRegression` class is implemented using PyTorch's `nn.Module` without using `nn.Linear`.
- **Initialization**: Supporting zero, random (small scale), and tensor-based initialization.
- **Forward Pass**: Explicitly calculating `logits = XW + b` followed by a sigmoid activation.
- **Decision Logic**: Converting probabilities to binary classes based on a 0.5 threshold.

## Numerical Stability and Loss Functions
One of the most critical lessons in this project is the importance of **numerical stability**.
- **Binary Cross-Entropy (BCE)**: Implementing a version that uses `torch.clamp` to avoid `log(0)` or `log(1)` errors which lead to `NaN` gradients.
- **Softmax vs. Sigmoid**: Exploring the relationship between binary and multi-class formulations.
- **Stable Softmax**: Understanding why subtracting the maximum value from logits (`z - max(z)`) prevents exponential overflow while keeping probabilities identical.

## Stochastic Gradient Descent (SGD) Pipeline
We build a full training loop from the ground up:
- **Mini-batching**: Efficiently processing data in chunks.
- **Shuffling**: Essential for SGD to ensure the model doesn't "memorize" the order of the dataset.
- **Metric Tracking**: Monitoring accuracy and F1-score across training and validation sets.

## Regularization (L1 & L2)
To combat overfitting and control model complexity, we implement:
- **L2 Regularization**: Penalizing large weights with a squared norm penalty, leading to smaller, more distributed weights.
- **L1 Regularization**: Using the absolute norm to encourage **sparsity**. We observe how L1 can effectively perform feature selection by driving irrelevant feature weights to exactly zero.

## Advanced Optimization Algorithms
Beyond standard SGD, we manually implement and visualize the trajectories of several advanced optimizers:
- **Momentum**: Accelerating descent by accumulating a "velocity" of past gradients, helping to navigate ravines and dampen oscillations.
- **AdaGrad**: Adjusting the learning rate for each parameter individually based on the history of squared gradients.
- **Adam (Adaptive Moment Estimation)**: The industry standard, combining the ideas of Momentum and RMSProp/AdaGrad with bias correction for the early steps of training.

## Experimental Results and Analysis
The project includes several experiments visualized through heatmaps and trajectory plots:
- **Learning Rate vs. Batch Size**: Analyzed the trade-off between convergence speed and stability. High learning rates with small batches often lead to faster but "noisier" training, while larger batches provide more stable gradient estimates.
- **Loss Landscapes**: Optimization algorithms were tested on synthetic functions:
  - **The Bowl**: A simple convex landscape.
  - **The Camel**: A more complex, non-convex landscape with multiple local minima and plateaus, testing the robustness of adaptive optimizers like Adam.
