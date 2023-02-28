# Network Anomaly Detection System

In this project we will use the NSL-KDD dataset to train our deep-machine-learning algorythm with pytorch. The NSL-KDD dataset is only used to teach/learn the machine about the network traffic.

First we need to install PyTorch. The issue with pytorch is that the current version of python (3.11) is not supported, therefore I had to install the version 3.7 which unfortunatly does not have the performance upgrades that came with 3.11.
For working with PyTorch it is recommended having an NVidia GPU. NVidia GPUs have CUDA-cores which massively boosts PyTorch's performance. For AMd GPU owners (like myself) pytorch can use the ROCm compute platform to make use of the GPU. Unfortunatelly it is not available for Windows.

![ROCm is available on Windows](./pictures/Pytorch_ROCm.png "PyTorch ROCm")

Some people have implemented a way to still use AMD GPUs for pytorch but it was too unreliable and too hacky for me. So i had to stick with the CPU.

## Convert the dataset
After downloading this project the "convert_dataset.py" will take the included dataset and start converting it to a usable format. The converted dataset will be saved in a newly created "Data"-folder.
The reason why the dataset has to be converted is because of text values and one hot encoding. Pytorch works with numbers and it can not deal with strings very well. So for example TCP and UDP have to be converted to numbers. With the LabelEncoder we can automatically assign a numeric value to the string (for example: TCP 0, UPD 1). This will enable pytorch to work with the values but it will still not deliver great results. It also allwos for our protocoll type to be 0.5 (half TCP, half UDP) which in the real world is impossible.

### One hot encoding
One hot encoding fixes this problem. We take the protocoll type and make every single value a column. It is simmilar to a "is-a"-value. With the TCP,UDP example our one hot encoded datset will have "is-tcp"-column and a "is-udp" column. This drastically enchances the quality of our system.

The results are stored in ./Data/train_enc.csv and the Label_Encoder-values are stored in label_encode_values.txt.

## Learning

I created two python files for training. train_Modal_selection.py and train_production.py. train_Modal_selection.py is optinal and goes through a lot of tests to see which model works better for us. "A model with more layer is called a deeper model. A model with more parameters on each layer is called a wider model." (https://machinelearningmastery.com/). The implementation of the deep and wide model can be seen in the "classes"-directory. After trying both for a couple of times the deep-model works just a bit better for my usecase.

![Accurcay of the wide and deep model](./pictures/Screenshot_Modal_Selection.png "Accurcay of the wide and deep model.")

The train_production.py does the actual work in our case. We take our deep model and start teaching it with our dataset. At the end we save our neural network into data ./Data/model.pth-file for later use. It takes 99% of the data for training purposes and only 1% for testing. The reason for that is that I already did a lot of testing in the train_Modal_selection.py-file, so it's not really needed here anymore. 

## Network detection
Now that the system converted and learned the data it's time to use it on our own network. There are many ways to use this system. You could for example route every system to the server which has this script running. For simplicity I am only sniffing the network packets that this PC is receiving. To sniff the packets a tool called scapy is used. Scapy is a powerful Python-based interactive packet manipulation program and library that allows users to capture, decode, analyze, forge and inject network packets. However, just a simple pip install scappy will not work since scappy needs Npcap. On windows machines Npcap has to be downloaded and installed first in order to run scappy. 

Network_anomaly_detection.py is loading our trained machine learning model and then starts sniffing for network packages. Each packet is then getting converted to a panda.Dataframe and analyzed by pytorch. When an anomaly is detected the user will get informed (TO DO).  

## My learnings

Overall I learned a lot about machine learning while doing this project, probably even more than my machine ;). It was my first time doing a machine-learning-project and things like label-encoding, one-hot-encoding, deep vs wide neuronal networks were completly new for me. I was stuck a couple of times and going back an forth a ton of times. Even now I would change a couple of things (mostly on how to convert the dataset and which source to use). But I have/had a lot of fun programming and researching this. 

# Sources
https://www.tutorialspoint.com/python_penetration_testing/python_penetration_testing_network_packet_sniffing.htm

https://betterprogramming.pub/building-a-packet-sniffing-tool-with-python-58ea5d65ace2

https://www.binarytides.com/python-packet-sniffer-code-linux/

https://towardsdatascience.com/categorical-encoding-using-label-encoding-and-one-hot-encoder-911ef77fb5bd

https://www.gkbrk.com/2016/05/hotel-music/

https://github.com/Mamcose/NSL-KDD-Network-Intrusion-Detection/blob/master/NID.ipynb

https://machinelearningmastery.com/building-a-binary-classification-model-in-pytorch/

https://pandas.pydata.org/docs/

https://pytorch.org/tutorials/

https://www.kaggle.com/code/avk256/nsl-kdd-anomaly-detection
