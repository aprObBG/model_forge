This is a codebase for training and forging neural network models.

## Folders
The codes for training and forging resnet/vgg on CIFAR-10 
are stored in resnet_cifar10 and vgg_cifar10, respectively.  

The codes for estimating the cryptographic performance on the
specific dataset-architecture setting are contained in the folder performance_estimation/

## Prerequisite
For model forging attacks:  
```shell
torch==1.7.0
torchvision==0.8.1
tqdm==4.42.1
numpy==1.18.1
scipy==1.4.1
ruamel-yaml==0.15.87
ipdb==0.13.4
```

For secure inference estimation:  
Packges for python:  
```shell
pycrypto==2.6.1
pycryptodomex==3.6.1
sklearn
```
Other requirements:  
SEAL 3.3.0 (Please refer to performance_estimation/SEAL/README.md for more details)  
NTTW (available at https://code.google.com/archive/p/finite-transform-library/)


## 1. To Start the Experiment on Model Forging
Go to any folder and do the following.
1. go to quantize-\* (e.g., quantize-ica)
2. Set the --data through command-line argument or editting in main.py or main\_ica.py
3. run 

    ```shell
    run.sh
    mv checkpoint trained
    ```

4. go to the corresponding attack folder (e.g., ica-attack)
5. Set the args.workspace variable in main.py
6. run 

    ```shell
    ln -s ../quantize-*/trained/ckpt.pth .
    run.sh
    ```

The above procedure will first train an oracle model to be stolen (in step 1 to 3), and then
perform model forging attack in a layer-by-layer manner. During the attack, the script will
incrementally set the front-end of the NN to be fixed and public (i.e., the encoder), and 
the back-end of the NN to be random (i.e., the backbone).

## 1. To Characterize the Performance
Refer to the readme file in performance_estimation/
