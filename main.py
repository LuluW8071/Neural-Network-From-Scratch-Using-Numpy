import argparse
import os
import pandas as pd
from data_loader import load_data
from train import train, predict
from neuralnet import *


def main(args):
    # Load data
    X_train, y_train, X_test = load_data(args.train_path, args.test_path)
    print("Loaded train.csv and test.csv")
    
    # Initialize weights
    input_size = X_train.shape[1]
    output_size = y_train.max() + 1
    W1, b1, W2, b2 = init_weights(input_size, 128, output_size)

    # Train the model
    W1, b1, W2, b2, histories = train(
        X_train.values, y_train, W1, b1, W2, b2, args.learning_rate, args.epochs, args.batch_size)

    # Make predictions
    y_test = predict(X_test.values, W1, b1, W2, b2)

    # Save predictions to a file
    submission = pd.DataFrame(
        {'ImageId': range(1, y_test.shape[0] + 1), 'Label': y_test})
    save_path = args.save_path if args.save_path else './'
    submission.to_csv(os.path.join(save_path, 'submission.csv'), index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network for digit recognition and make predictions.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training data CSV file.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to test data CSV file.")
    parser.add_argument("--learning_rate", type=float, default=0.01,
                        help="Learning rate for training the model.")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--save_path", type=str, default='',
                        help="Directory to save the submission file.")
    args = parser.parse_args()

    main(args)
