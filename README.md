##### Instructions and headsups
This is the modified version of RGL's git repo, which contains many improvements that I have made.
crowd_nav is the training and testing code, where crowd_sim is the simulating environment.
Follow the original RGL instructions below to install, and read the following comments.

Simply follow the command line argument list at the end of the files train.py and test.py to start training and testing.

To train a model, change the argumment "--config": the config file path, that is the main thing that governs which model to train and the environment settings.
Also specify a different "--model_dir" of your wish.
Trained model will be inside data/model_name folder that you specified, and the model files are named rl_model_x, x means the check point number.

Go to configs/icra_benchmark/ to see a list of all different models, and model_predictive_rl is the officical RGL config.
All configs inherit from config.py, which is the base config, and no change should be made to that file. You can create your own copy of your config by simpy 
inherit your config from the config.py's settings, and add only the parameters that are different from the base config.


To resume training, firstly go into the model folder, find the model checkpoint file you wish to continue from, and rename it rl_model.py. Then, simply specify the model directory as argument and add "--resume" flag to train. Resumed training produce resumed_rl_model_x, where x is check point number.

To evaluate a model, follow the argument in test.py. Note I have modified the config so that testing environemnt differs from training environment in that humans will treat robots as visible and have a larger safety distance, you can change that in config file.
Note also the model folder must have a best_val.py, you can rename any checkpoint as that file if the training has not yet produced one.

I have also added a modify_model_name.py in crowd_nav folder, which will rename all resumed_rl_model_x.py to rl_model_y.py and their ckeck point numbers will succeed the largest already trained checkpoint number. It will also rename the rl_model.py and best_val.py to the one with largest checkpoint number if you use the argument to enable the setting.

##### Below are original readme from RGL's repository

# RelationalGraphLearning
This repository contains the codes for our paper, which is accepted at IROS 2020. 
For more details, please refer to the [paper](https://arxiv.org/abs/1909.13165).


## Abstract
We present a relational graph learning approach for robotic crowd navigation using model-based deep reinforcement 
learning that plans actions by looking into the future.
Our approach reasons about the relations between all agents based on their latent features and uses a Graph Convolutional 
Network to encode higher-order interactions in each agent's state representation, which is subsequently leveraged for 
state prediction and value estimation.
The ability to predict human motion allows us to perform multi-step lookahead planning, taking into account the temporal 
evolution of human crowds.
We evaluate our approach against a state-of-the-art baseline for crowd navigation and ablations of our model to 
demonstrate that navigation with our approach is more efficient, results in fewer collisions, and avoids failure cases 
involving oscillatory and freezing behaviors.



## Method Overview
<img src="https://i.imgur.com/8unQNIv.png" width="1000" />


## Setup
1. Install [Python-RVO2](https://github.com/sybrenstuvel/Python-RVO2) library
2. Install [socialforce](https://github.com/ChanganVR/socialforce) library
2. Install crowd_sim and crowd_nav into pip
```
pip install -e .
```

## Getting Started
This repository are organized in two parts: crowd_sim/ folder contains the simulation environment and
crowd_nav/ folder contains codes for training and testing the policies. Details of the simulation framework can be found
[here](crowd_sim/README.md). Below are the instructions for training and testing policies, and they should be executed
inside the crowd_nav/ folder.


1. Train a policy.
```
python train.py --policy rgl
```
2. Test policies with 500 test cases.
```
python test.py --policy rgl --model_dir data/output --phase test
```
3. Run policy for one episode and visualize the result.
```
python test.py --policy rgl --model_dir data/output --phase test --visualize --test_case 0
```
4. Plot training curve
```
python utils/plot.py data/output/output.log
```



## Video Demo
[<img src="https://i.imgur.com/SnqVhit.png" width="70%">](https://youtu.be/U3quW30Eu3A)


## Citation
If you find the codes or paper useful for your research, please cite the following papers:
```bibtex
@inproceedings{chen2020relational,
    title={Relational Graph Learning for Crowd Navigation},
    author={Changan Chen and Sha Hu and Payam Nikdel and Greg Mori and Manolis Savva},
    year={2020},
    booktitle={IROS}
}
@inproceedings{chen2019crowdnav,
    title={Crowd-Robot Interaction: Crowd-aware Robot Navigation with Attention-based Deep Reinforcement Learning},
    author={Changan Chen and Yuejiang Liu and Sven Kreiss and Alexandre Alahi},
    year={2019},
    booktitle={ICRA}
}
```
