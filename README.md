# Matrix Factorization with Negative Sampling

This project implements a Matrix Factorization model for recommendation systems with different negative sampling techniques: **Augmented Negative Sampling (ANS)**, **Hard Negative Sampling (HNS)**, and **Random Negative Sampling (RNS)**. It demonstrates the effectiveness of various negative sampling methods for collaborative filtering tasks in recommendation systems.

## Table of Contents

- [Project Overview](#project-overview)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Results](#results)
- [License](#license)

## Project Overview

Matrix Factorization (MF) is a widely-used technique in recommendation systems that aims to predict missing values in a user-item interaction matrix. In this project, we implement a Matrix Factorization model and enhance it with various **negative sampling techniques** to improve the model's performance.

Negative sampling is an essential component for training collaborative filtering models. By generating negative samples (non-interacted user-item pairs), we can enhance the recommendation system's ability to learn useful patterns.

The techniques implemented in this project include:

- **Augmented Negative Sampling (ANS)**: Generates negative samples with slight perturbations.
- **Hard Negative Sampling (HNS)**: Selects the hardest negative samples that are closest to the actual positive samples.
- **Random Negative Sampling (RNS)**: Randomly selects negative samples from the user-item space.

## Requirements

Before running the code, you need to install the following Python libraries:

- `numpy`
- `pandas`
- `scikit-learn`

To install them, run the following command:

```bash
pip install numpy pandas scikit-learn
```

## Installation

1. Clone the repository to your local machine:

   ```bash
   git clone https://github.com/Jeffrey-1236/Matrix-Factorization-Negative-Sampling.git
   ```

2. Navigate into the project directory:

   ```bash
   cd Matrix-Factorization-Negative-Sampling
   ```

3. (Optional) Install the required dependencies:

   ```bash
   pip install -r requirements.txt
   ```

## Usage

To run the project, simply execute the main Python script:

```bash
python matrix_factorization.py
```

This script will generate a random user-item interaction dataset, train the Matrix Factorization model with different negative sampling techniques, and print evaluation metrics (Precision, Recall, and F1-score) for each method.

### Example Output

```
ANS Precision: 0.8321, Recall: 0.7485, F1-score: 0.7873
HNS Precision: 0.8119, Recall: 0.7621, F1-score: 0.7865
RNS Precision: 0.7892, Recall: 0.7283, F1-score: 0.7575
```

These metrics will help you compare the performance of the different negative sampling strategies.

## Results

In the evaluation, the following metrics are used to assess the performance of each negative sampling method:

- **Precision**: The proportion of relevant items retrieved by the model.
- **Recall**: The proportion of relevant items that were actually retrieved by the model.
- **F1-score**: The harmonic mean of Precision and Recall, providing a balanced measure of the model's performance.

The model trains for a fixed number of epochs and the results are printed after training. You can adjust the number of epochs and the learning rate for further experimentation.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
```