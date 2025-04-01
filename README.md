# Signal Classifier Using Neural Network

## Overview
This is a simple project implemented for the second "Text Mining" course assignment.
This project implements a neural network to classify space signals into three categories: **Natural**, **Artificial**, and **Unknown**. The goal is to identify potential signs of intelligence in space missions. Signals are represented by three features: frequency, amplitude, and wavelength.

## Project Structure

```bash

signal_classifier_project/
│
├── src/
│   ├── data_preparation.py   # Generate synthetic data
│   ├── model.py              # Define the neural network model
│   ├── train.py              # Train the model
│   ├── evaluate.py           # Evaluate the model
│   └── predict.py            # Predict on new data
│
├── main.py                   # Main script to run the project
│
├── requirements.txt          # List of dependencies
│
└── README.md                 # Project documentation

```

## Installation
1. Clone the repository:
```bash
git clone https://github.com/your-username/signal_classifier_project.git](https://github.com/Aminam78/signal_classifier_project_optimum.git
cd signal_classifier_project_optimum

```
2. Install the required dependencies:

```bash
pip install -r requirements.txt

```

## Usage
Run the project using the main script:
```bash
python main.py

``` 
This will:
- Generate synthetic training and test data
- Train a neural network with two hidden layers
- Evaluate the model's accuracy on the test set
- Predict the class of new signals

**Example output:**
```bash

Epoch 1/50, Loss: 1.0986
...
Epoch 50/50, Loss: 0.8765
Accuracy on test data: 0.8950
Prediction for new signal [0.5, 0.5, 0.5]: Artificial

```

## Methodology
1. **Data Preparation** (`data_preparation.py`):
   - Synthetic data is generated with three features and random labels.
2. **Model Definition** (`model.py`):
   - A neural network with two hidden layers (5 and 3 neurons) and an output layer with 3 neurons.
3. **Training** (`train.py`):
   - The model is trained using SGD with a learning rate of 0.01 for 50 epochs.
4. **Evaluation** (`evaluate.py`):
   - The model's accuracy is calculated on the test data.
5. **Prediction** (`predict.py`):
   - The model predicts the class of new signals.

## Results
- The model achieved an accuracy of approximately 90% on the test set.
- Predictions on new signals were successfully performed.

## Future Improvements
- Use real-world data for training and testing.
- Experiment with different architectures and hyperparameters.
- Add more features to improve classification accuracy.

## Dependencies
- Python 3.7+
- PyTorch
- NumPy

## Author
Amirhossein Amin Moghaddam
Master's Student in Computer Software Engineering at Iran University of Science and Technology (IUST)

Text Mining Course  
March 2025
