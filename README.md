# FaceRecogImplementation
An implementation of a facial recognition algorithm using convolutional neural networks, trained on the Extended Yale Face Database B.

#Method
The network uses the Tensorflow library's python API with GPU support to classifiy faces. The Network graph is shown below but can be subject to change as issues are fixed and detection is improved. 

Currently the network is trained over 2000 steps using a learning rate of 0.0001. The image data is split 80% into training and 20% for evaluation. The system currently has an accuracy of 27% which hopefully can be improved upon.

#Network Graph
![Alt text](/graph_images/NetworkGraphA.jpg?raw=true "Network Graph") 

#Current Loss Over Training Steps
![Alt text](/graph_images/NetworkLossA.jpg?raw=true "Network Loss")
