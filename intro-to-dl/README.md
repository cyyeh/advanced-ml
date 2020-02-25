# Introduction to Deep Learning

## Overview

The goal of this course is to give learners basic understanding of modern neural networks and their applications in computer vision and natural language understanding. The course starts with a recap of linear models and discussion of stochastic optimization methods that are crucial for training deep neural networks. Learners will study all popular building blocks of neural networks including fully connected layers, convolutional and recurrent layers. 
Learners will use these building blocks to define complex modern architectures in TensorFlow and Keras frameworks. In the course project learner will implement deep neural network for the task of image captioning which solves the problem of giving a text description for an input image.

The prerequisites for this course are: 
1) Basic knowledge of Python.
2) Basic linear algebra and probability.

Please note that this is an advanced course and we assume basic knowledge of machine learning. You should understand:
1) Linear regression: mean squared error, analytical solution.
2) Logistic regression: model, cross-entropy loss, class probability estimation.
3) Gradient descent for linear models. Derivatives of MSE and cross-entropy loss functions.
4) The problem of overfitting.
5) Regularization for linear models.

## Key Concepts by Week

Week 1, Introduction to optimization
> Welcome to the "Introduction to Deep Learning" course! In the first week you'll learn about linear models and stochatic optimization methods. Linear models are basic building blocks for many deep architectures, and stochastic optimization is used to learn every model that we'll discuss in our course.

- Use linear models for classification and regression tasks
- Train a linear model for classification or regression task using stochastic gradient descent
- Tune SGD optimization using different techniques
- Apply regularization to train better models

Week 2, Introduction to neural networks
> This module is an introduction to the concept of a deep neural network. You'll begin with the linear model and finish with writing your very first deep network.

- Explain the mechanics of basic building blocks for neural networks
- Apply backpropagation algorithm to train deep neural networks using automatic differentiation
- Implement, train and test neural networks using TensorFlow and Keras

Week 3, Deep Learning for images
> In this week you will learn about building blocks of deep learning for image input. You will learn how to build Convolutional Neural Network (CNN) architectures with these blocks and how to quickly solve a new task using so-called pre-trained models.

- Define and train a CNN from scratch
- Understand building blocks and training tricks of modern CNNs
- Use pre-trained CNN to solve a new task

Week 4, Unsupervised representation learning
> This week we're gonna dive into unsupervised parts of deep learning. You'll learn how to generate, morph and search images with deep learning.

- Understand basics of unsupervised learning of word embeddings
- Apply autoencoders for image retrieval and image morphing
- Implement and train generative adversarial networks
- Implement and train deep autoencoders
- Understand what is unsupervised learning and how you can benifit from it

Week 5, Deep learning for sequences
> In this week you will learn how to use deep learning for sequences such as texts, video, audio, etc. You will learn about several Recurrent Neural Network (RNN) architectures and how to apply them for different tasks with sequential input/output.

- Use RNNs for different types of tasks: sequential input, sequential output, sequential input and output
- Define and train an RNN from scratch
- Understand modern architectures of RNNs: LSTM, GRU

Week 6, Final Project
> In this week you will apply all your knowledge about neural networks for images and texts for the final project. You will solve the task of generating descriptions for real world images!

- Apply your skills to train an Image Captioning model