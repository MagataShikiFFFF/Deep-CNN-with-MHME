# Deep-CNN-with-MHME
This is a novel hierarchical multilabel classifier implementing the whole set of hierarchically organized local classifiers in one deep convolution neural network (CNN) model with multiple heads and multiple ends. The proposed MHME CNN model consists of three parts: the body part of a deep CNN model shared by different local classifiers for feature extraction and feature mapping; the multiend part of a set of autoencoders performing feature fusion transforming the input vectors of different local classifiers to feature vectors with the same length so as to share the feature mapping part; and the multihead part of a set of linear multilabel classifiers.

It is built based on the framework chainer. 
The datasets are in folder n-gram.
