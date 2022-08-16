# Classical-to-quantum convolutional neural network transfer learning
These codes are implementation of classical-to-quantum neural network (C2Q-CNN) transfer learning (TL).
C2Q-CNN is composed of convolutional neural network (CNN) pre-training model and quantum CNN (QCNN) fine-tuning model.
In this work, QCNN is composed of convolution layers and pooling layers.
This code can numerically simulate binary classification of various C2Q-CNN TL models and classical-to-classical (C2C) TL models under similar training conditions.
This code pre-train CNN model with Fashion-MNIST data, and fine-tune QCNN or CNN model with MNIST data.
Binary classification is used in this code because this code measure only one qubit of QCNN.
"CQtransfer_FtoM_bench.py" file trains and tests C2Q-CNN TL models and save results to files.
"CQtransfer_FtoM_bench_angle.py" also trains and tests C2Q-CNN TL models and save results to files.
The difference between first file and second file is encoding method.
First file use amplitude encoding method, and second file use qubit encoding method.
"CCtransfer_FtoM.py" file trains and tests C2C TL models and save results to files.
### Acknowledgement
QOSF code were forked from (https://github.com/takh04/QOSF_project) and modified to run QCNN transfer learning model.
For more information of C2Q-CNN TL, see “Classical-to-quantum convolutional neural network transfer learning” paper.
