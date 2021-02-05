Automatically generate parameters and characterize cryptographic
parameters.


## 1. Installing libraries
To execute the scripts, NTTW and SEAL needs to be installed. The sources
of the required libraries can be found at:  
SEAL 3.3.0 (https://github.com/microsoft/SEAL/tree/3.3.0)  
NTTW v1.31 (https://code.google.com/archive/p/finite-transform-library/downloads)  


## 2. Model Characterization
1. produce the architecture in a python ordered dictionary format (examples for VGG, ResNet, ICA-VGG and ICA-ResNet are included)
2. open main_arch.py and instantiate the corresponding architecture on line 20
3. execute
    ```shell
    python main_arch.py
    ```
    or
    ```shell
    ./main_arch.py
    ```
