# Assignment 1: Neural Networks from Scratch (NumPy)

This directory contains the complete implementation, experiments, and analysis for **Assignment 1** of **DA6401: Introduction to Deep Learning**.

The focus of this assignment is to **build a fully configurable Multi-Layer Perceptron (MLP) from scratch using NumPy**, implement manual forward and backward propagation, and evaluate the model on **MNIST** and **Fashion-MNIST** datasets.

---

## Assignment Objectives

The objectives of this assignment are to:

* Implement a **modular MLP** without using any automatic differentiation libraries
* Manually derive and code:

  * Forward propagation
  * Backpropagation
  * Gradient-based optimization
* Experiment with different:

  * Loss functions
  * Optimizers
  * Activation functions
  * Weight initialization strategies
* Perform systematic experimentation and analysis using **Weights & Biases**
* Ensure correctness and robustness under an **automated evaluation pipeline**

---

## Constraints & Academic Integrity

* **Frameworks prohibited**: PyTorch, TensorFlow, JAX, or any library providing automatic differentiation
* **Permitted libraries**:

  * `numpy` (core implementation)
  * `keras.datasets` (data loading)
  * `scikit-learn` (splits, metrics)
  * `matplotlib` (visualization)
  * `wandb` (experiment tracking)
* AI tools were used **only for conceptual clarification**, not for code generation
* Training and test datasets are **strictly isolated**
* Any form of data leakage or plagiarism results in **immediate disqualification**

---

## Command-Line Interface

Both `train.py` and `inference.py` are fully configurable via `argparse`.

### Required Arguments

| Argument          | Description                                          |
| ----------------- | ---------------------------------------------------- |
| `--dataset`       | `mnist` or `fashion_mnist`                           |
| `--epochs`        | Number of training epochs                            |
| `--batch_size`    | Mini-batch size                                      |
| `--loss`          | `mse` or `cross_entropy`                             |
| `--optimizer`     | `sgd`, `momentum`, `nag`, `rmsprop`, `adam`, `nadam` |
| `--learning_rate` | Initial learning rate                                |
| `--weight_decay`  | L2 regularization coefficient                        |
| `--num_layers`    | Number of hidden layers                              |
| `--hidden_size`   | Neurons per hidden layer (list)                      |
| `--activation`    | `sigmoid`, `tanh`, or `relu`                         |
| `--weight_init`   | `random` or `xavier`                                 |

---

## Implementation Notes

* **Manual Gradient Computation**

  * Each layer stores gradients explicitly:

    * `self.grad_W`
    * `self.grad_b`
* **Gradient Verification**

  * Analytical gradients match numerical gradients within tolerance `1e-7`
* **Model Serialization**

  * Exactly one model (`best_model.npy`) is submitted
  * Selection criterion: **highest test F1-score**
* **Evaluation Metrics**

  * Accuracy
  * Precision
  * Recall
  * F1-score

---

## Weights & Biases Report

A **public Weights & Biases report** accompanies this assignment and includes:

* Dataset exploration and class visualization
* Large-scale hyperparameter sweep (≥100 runs)
* Optimizer convergence comparison
* Vanishing gradient and dead neuron analysis
* Loss function comparison (MSE vs Cross-Entropy)
* Generalization gap analysis
* Confusion matrix and error visualization
* Weight initialization and symmetry-breaking study
* Fashion-MNIST transfer analysis

---

## Submission Notes

* Maximum hidden layers: **6**
* Maximum neurons per layer: **128**
* Only one final model is submitted
* Repository and W&B report are **publicly accessible** during evaluation

---

## Timeline

* **Release Date:** 9 Feb 2026
* **Deadline:** 1 Mar 2026
* **Submission Platform:** Gradescope

---

## Relation to Course Objectives

This assignment directly supports the course goals of:

* Understanding neural network fundamentals
* Implementing learning algorithms from first principles
* Analyzing optimization behavior and generalization
* Developing experimental rigor in deep learning workflows
