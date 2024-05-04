# Deep Learning Ops

This repository contains code to train and evaluate a Vision Transformer (VIT) model on a sports dataset and run the model in ONNX (in Python) and TorchScript format in C++.

#### These steps are written keeping Linux as os, however same code can be executed in MacOS and Windows by installing same libraries for these respective operating systems.

## Setup

### 1. Creating Conda Environment

```bash
conda create --name deep-learning-ops python=3.8
conda activate deep-learning-ops
```
### 2. Installing Python Libraries
```bash
pip install -r requirements.txt
```

## Dataset
Download the Sports image dataset from Kaggle and place it in the dataset folder after unzipping it. The structure should be as follows:
dataset
```
│   ├── sportsdataset
│   │   ├── train
│   │   ├── test
│   │   └── split
```
Link for the sports dataset is `https://www.kaggle.com/datasets/gpiosenka/sports-classification`

## How to Run Python Code
- To TRAIN the model, set the `mode` in train.py as "train". Trained models will be automatically saved in the models folder.
- To TEST the trained model, change the mode to "test" in train.py.
```bash
python train.py
```
## How to covert torch model to torchscript format
- Change the mode to "torchscript" in train.py and run it
```bash
python train.py
```
## How to convert the torch model to ONNX model
- After creating the torch model, run the `convert_to_onnx.py` file,, it will save the ONNX format model to `models` folder
```bash
python convert_to_onnx.py
```

#### The trained and converted model can be found in `models` folder

## Inferance for ONNX
place the images you want to perform prediction for in `testimages` folder and run  `onnx_inferance.py`
```bash
python onnx_inferance.py
```
##Run Trochscript model in C++ on test dataset
### Install C++ in Linux environment 
- Use the link here for step by step `https://data-flair.training/blogs/install-cpp/`
  
### Install LIbtorch for C++
- Download the libtorch files from official Pytorch website as given here `https://pytorch.org/get-started/locally/`
- Unzip the file in your current folder to keep it in code level
- update the libtorch path in the CMakeLists.txt in `torch_script_inferance` of current repo.
  
### Install OpenCV in Linux, use the following link for step by step guide
- `https://www.geeksforgeeks.org/how-to-install-opencv-in-c-on-linux/`

## Run C++ code to inferance on test dataset
- To run the c++ code on complete test dataset for evaluation, keep the code as it is (becacuse the mode is already set for `test dataset`).
- If you want to infrance the model on single image, change the mode in `infrancescript.cpp` to `single image` and pass the image path in the same file ta `Line number 38` in "filename" variable

In both of above cases, C++ code will be executed by following commands 
```bash
cd torch_script_inferance
cmake .
make
./torchscript
```

