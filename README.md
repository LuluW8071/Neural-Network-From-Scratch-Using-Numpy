# Neural-Network-From-Scratch-Using-Numpy

This repository contains code for building and training a neural network from scratch using Numpy, a Python library for numerical computing. The neural network is trained on the **MNIST dataset** for digit recognition.

![Linear Neural Network](https://miro.medium.com/v2/resize:fit:679/1*0WUg6f46UmDYij6nAqh-7w.gif)

## Overview

In this project, we implement a simple neural network architecture with one hidden layer using the following steps:

| Step               | Description                                                                                       |
|--------------------|---------------------------------------------------------------------------------------------------|
| **Data Loading**   | Load the **MNIST dataset** containing images of handwritten digits which can be downloaded form [here](https://www.kaggle.com/competitions/digit-recognizer)          |
| **Data Preprocessing** | Normalize the images to have pixel values between 0 and 1. **One-hot encode** labels for classification. |
| **Model Architecture** | Construct a neural network with an **input layer**, a **hidden layer** with **ReLU** activation, and an **output layer** with **softmax** activation. |
| **Training**       | Train the neural network using **batch gradient descent** with **backpropagation**. Monitor training for loss and accuracy. |
| **Evaluation**     | Evaluate the trained model on a separate test set to measure its performance.                     |
| **Prediction**     | Use the trained model to make predictions on new unseen data.                                      |

## Usage

1. Clone the repository:

```bash
git clone https://github.com/LuluW8071/Neural-Network-From-Scratch-Using-Numpy.git
```

2. Install the required dependencies:

```bash
pip install -r requirements.txt
```

3. Run the training script:

```bash
python main.py --train_path "/path/to/train.csv" --test_path "/path/to/test.csv" --plot 1
```

You can also adjust other parameters for train configs as needed. Here's a table summarizing the command-line arguments accepted by the script:

| Argument          | Description                                                                                   |
|-------------------|-----------------------------------------------------------------------------------------------|
| `--train_path`    | Path to the CSV file containing the training data.                                            |
| `--test_path`     | Path to the CSV file containing the test data.                                                |
| `--learning_rate` | Learning rate to be used for training the model. Default is `0.001`.                            |
| `--epochs`        | Number of epochs for training the model. Default is `150`.                                       |
| `--batch_size`    | Batch size to be used during training. Default is `64`.                                          |
| `--plot`          | Whether to plot the loss and accuracy curves after training. Set to `1` to plot, `0` otherwise.    |
| `--save_path`     | Directory to save the submission file. Default is `current_directory`.                           |

#### Sample Demo Run Output
```bash
(env) PS D:\Neural-Network-From-Scratch-Using-Numpy> py .\main.py --epochs 150 --plot 1 --train_path ".\train.csv" --test_path ".\test.csv"
Loaded train.csv and test.csv
100%|███████████████████████████████████████████████████████| 150/150 [07:27<00:00,  2.99s/it, accuracy=1, loss=1.3e-5]
Saved inference.csv
```

4. After training, the model will make predictions on the `test.csv` set and save the results in a `inference.csv` file.

### Accuracy and Loss Curves
![Figure_1](https://github.com/LuluW8071/Neural-Network-From-Scratch-Using-Numpy/assets/107304848/62af617d-901b-42d7-b1c9-2354dbe8bc34)

Also if you want the **CNN(Convolutional Neural Network)** implementation, you can find the notebook [here](https://www.kaggle.com/code/luluw8071/mnist-trained-on-tinyvgg-model-with-pytorch?scriptVersionId=168950513). 

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

--- 

Feel free to report any issues you encounter.</br>
<img src="https://user-images.githubusercontent.com/74038190/213844263-a8897a51-32f4-4b3b-b5c2-e1528b89f6f3.png" width="25px" /> Don't forget to star the repo :)
