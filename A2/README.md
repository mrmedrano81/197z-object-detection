### SETUP:

<code>!pip install -r requirements.txt</code>

### TRAINING:

<code>!python train.py</code>

### EVALUATION/TESTING:

<code>!python test.py</code>

### WEBCAM DEMO:

<code>!python demo.py</code>

Notes:

-The code assumes that torch, torchvision, and NVIDIA CUDA toolkit have been setup in the machine/environment being used to run it. Hence torch and torchvision are not included in the requirements.txt file in the setup.



References:

-training and finetuning modules/libraries:
>https://github.com/mrmedrano81/vision/tree/main/references/detection

-model used:
>https://paperswithcode.com/paper/faster-r-cnn-towards-real-time-object

-utility files used in generating dictionary files for dataloader:
>https://github.com/PacktPublishing/Advanced-Deep-Learning-with-Keras/tree/master/chapter11-detection
