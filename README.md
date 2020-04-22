# ContinualLearning with Gated Incremental Memories for sequential data processing
[Paper](https://arxiv.org/abs/2004.04077) accepted at IJCNN 2020.

## MNIST task
Run the script `mnist.py` with your hyperparameters of choice.

## Audioset task
Download `bal.h5`, `eval.h5` and `unbal_train.h5` from [here](https://drive.google.com/drive/folders/1IlsVeAD9iAhK1Keu958RR8hXd2rcRnq5?usp=sharing) and put them in `tasks/audioset/data/`.  
Then, run `audioset_task.py` with your hyperparameters of choice.

## Devanagari task
Download Devanagari dataset from [here](https://drive.google.com/file/d/1dcP0m02bRyKGebZxwq_jMifuTZsXk5RJ/view?usp=sharing) and put `Train` and `Test` folder inside `tasks/mnist/data/Devanagari_CL/`.  
Then, run `mnist.py --devanagari` with your hyperparameters of choice.

