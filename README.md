# RABNN_TPAMI_2022

# For Stage-1 Training of a VGG model run the following command:
First in the modules folder use the stage-1 binary module in the init file and comment out the stage-2 binary module then run the command:
bash grow.sh

# For Stage-2 Training of a VGG model run the following command:
The results should print the multiplier factor to each channel for example 0.8 at layer 1. then in vgg.py file at line 153 chs = [0.8*3,....] so on:
Then in the modules folder use the stage-2 binary module in the init file and run :
bash run.sh

# To attack the VGG model run the following command:
the model will be saved in model folder vgg_rabnn.pkl. To attack this model specify the destination folder in line 239 of attack_targeted.py file and run:
bash attack.sh

## ALL the results in the paper can be found in the results folder with network specifications and attack evolutions.

Pre-trained models: https://drive.google.com/drive/folders/19iEKRhjcYCuTtCVRqEcHqb1Nzgp1rrt_?usp=sharing

To test the attack on a pre-trained model download 'vgg_rabnn.pkl' in models folder. Then run the following command:

## targeted attack
bash attack.sh

## un-targeted attack
bash attack_un.sh
