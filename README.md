# 3D Equivariant Convolutional Neural Network
 
This repository implements a SE(3)-equivariant convolutional neural network using the [escnn](https://github.com/QUVA-Lab/escnn) library. The model is designed for processing 3D volumetric data with rotational equivariance, making it suitable for applications in physics, along with the implementation of lie derivative-based estimation of learned equivariance of the model using [lie-deriv](https://github.com/ngruver/lie-deriv?utm_source=catalyzex.com) library.

## Overview
The core script for training and evaluating the equivariant neural network is located in the `codes` directory with the name `main_script.py`. This script orchestrates the full training pipeline, including model initialization, loading/saving checkpoints, training, evaluation, and logging. You can run the script directly from the command line with various configurable arguments.

### ⚙️ Script Arguments

| Argument | Type | Default | Description |
| -------- | ---- | ------- | ----------- |
| `--epochs` | `int` | 10 | Number of epochs to train the model. |
| `--train_case` | `str` | - | Subfolder name inside `./../data/` used for training data. |
| `--test_case` | `str` | `plain` | Subfolder name inside `./../data/test/` used for testing data. |
| `--run_name` | `str` | `anonymous` | Name for the current training run. Used for creating a directory in `./../models/` and `./../outputs/` to save model checkpoints and progress log. |
| `--load_model` | `flag` | False | If set, loads an existing model checkpoint to resume training. |
| `--additional_epochs`  | `int` | 10 | Additional epochs to train when resuming from a checkpoint. Used when the `--load_model` flag is used in the command line. |

### 📋 Example

To train a model for 20 epochs using data in `plain` subdirectory inside `./../data/` from the beginning (i.e., no checkpoints) and test it on the data present in `plain` subdirectory inside `./../data/test/`, first change directory `cd ./codes/` and use the command  `python3 main_script.py --train_case plain --test_case plain --epochs 20 --run_name experiment1`

Now the model has already trained for 20 epochs. To resume the training from 21<sup>st</sup> epoch for additional 5 epochs, execute the command `python3 main_script.py --train_case plain --test_case plain --epochs 20 --run_name experiment1 --load_model --additional_epochs 5`

### 🔧 Installation

To have the code running, [escnn](https://github.com/QUVA-Lab/escnn) and [lie-deriv](https://github.com/ngruver/lie-deriv?utm_source=catalyzex.com) packages should be installed. Follow the installation process on their respective pages by clicking the links attached. The latest release on [escnn](https://github.com/QUVA-Lab/escnn) can be installed as `pip install escnn` or by cloning the repository by `pip install git+https://github.com/QUVA-Lab/escnn`. 
Installation of [lie-deriv](https://github.com/ngruver/lie-deriv?utm_source=catalyzex.com) is done as:
<pre>git clone --recurse-submodules https://github.com/ngruver/lie-deriv.git
cd lie-deriv
pip install -r requirements.txt </pre>

### Credits and Acknowledgements

- The credit for 3D steerable equivariant CNN go to author(s) of [escnn](https://github.com/QUVA-Lab/escnn) library.
- The credit for Lie Derivative based estimation of Equivariance Error go to the author(s) of [lie-deriv](https://github.com/ngruver/lie-deriv?utm_source=catalyzex.com) library
