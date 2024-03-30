import argparse
import os
import pandas as pd
import numpy as np
from data_loader import load_data
from train import train, predict, plot_training_history
from neuralnet import init_weights


def main(args):
    # Load data
    X_train, y_train, X_test, num_classes = load_data(args.train_path, args.test_path)
    print("Loaded train.csv and test.csv")
    
    # Initialize weights
    input_size = X_train.shape[1]
    hidden_size = 128
    output_size = num_classes
    W1, b1, W2, b2 = init_weights(input_size, hidden_size, output_size)

    # Train the model
    W1, b1, W2, b2, histories = train(
        X_train.values, y_train, W1, b1, W2, b2, args.learning_rate, args.epochs, args.batch_size)

    # Make predictions
    y_test = predict(X_test.values, W1, b1, W2, b2)
    # print(y_test.shape)

    y_test =np.argmax(y_test, axis=1)
    image_ids = range(1, len(y_test) + 1)
    inference = pd.DataFrame({'ImageId': image_ids, 'Label': y_test})

    # Set the default save path if not provided
    save_path = args.save_path if args.save_path else './'
    inference.to_csv(os.path.join(save_path, 'inference.csv'), index=False)
    print("Saved inference.csv")

    # Plot graph
    if args.plot == 1:
        histories_df = pd.DataFrame(histories)
        histories_df_grouped = histories_df.groupby('epoch').agg(['mean', 'min', 'max'])
        plot_training_history(histories_df_grouped)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train a neural network for digit recognition and make predictions.")
    parser.add_argument("--train_path", type=str, required=True,
                        help="Path to training data CSV file.")
    parser.add_argument("--test_path", type=str, required=True,
                        help="Path to test data CSV file.")
    parser.add_argument("--learning_rate", type=float, default=0.001,
                        help="Learning rate for training the model.")
    parser.add_argument("--epochs", type=int, default=150,
                        help="Number of epochs for training.")
    parser.add_argument("--batch_size", type=int, default=64,
                        help="Batch size for training.")
    parser.add_argument("--plot", type=int, default=0,
                        help="Plot loss and acc curves: 1 else 0")
    parser.add_argument("--save_path", type=str, default='',
                        help="Directory to save the submission file.")
    args = parser.parse_args()

    main(args)
