# Bayesian Methods for Machine Learning

## Overview

People apply Bayesian methods in many areas: from game development to drug discovery. They give superpowers to many machine learning algorithms: handling missing data, extracting much more information from small datasets. Bayesian methods also allow us to estimate uncertainty in predictions, which is a desirable feature for fields like medicine. 
When applied to deep learning, Bayesian methods allow you to compress your models a hundred folds, and automatically tune hyperparameters, saving your time and money.
In six weeks we will discuss the basics of Bayesian methods: from how to define a probabilistic model to how to make predictions from it. We will see how one can automate this workflow and how to speed it up using some advanced techniques. 
We will also see applications of Bayesian methods to deep learning and how to generate new images with it. We will see how new drugs that cure severe diseases be found with Bayesian methods.

## Key Concepts by Week

Week 1, Introduction to Bayesian methods & Conjugate priors
> Welcome to first week of our course! Today we will discuss what bayesian methods are and what are probabilistic models. We will see how they can be used to model real-life situations and how to make conclusions from them. We will also learn about conjugate priors — a class of models where all math becomes really simple.

- Understand Bayesian approach to statistics
- Learn how to define a probabilistic model
- Learn how to apply Bayesian inference

---

Week 2, Expectation-Maximization algorithm
> This week we will about the central topic in probabilistic modeling: the Latent Variable Models and how to train them, namely the Expectation Maximization algorithm. We will see models for clustering and dimensionality reduction where Expectation Maximization algorithm can be applied as is. In the following weeks, we will spend weeks 3, 4, and 5 discussing numerous extensions to this algorithm to make it work for more complicated models and scale to large datasets.

- Understand what is a latent variable and apply them to simplify probabilistic models
- Cluster data with Gaussian Mixture Model
- Train probabilistic models with Expectation Maximization algorithm

---

Week 3, Variational Inference & Latent Dirichlet Allocation
> This week we will move on to approximate inference methods. We will see why we care about approximating distributions and see variational inference — one of the most powerful methods for this task. We will also see mean-field approximation in details. And apply it to text-mining algorithm called Latent Dirichlet Allocation

- Understand when Variational inference is needed
- Apply variational inference for probabilistic models
- Understand variational interpretation of Latent Dirichlet Allocation
- Application of LDA to text mining

---

Week 4, Markov chain Monte Carlo
> This week we will learn how to approximate training and inference with sampling and how to sample from complicated distributions. This will allow us to build simple method to deal with LDA and with Bayesian Neural Networks — Neural Networks which weights are random variables themselves and instead of training (finding the best value for the weights) we will sample from the posterior distributions on weights.

- Train / do inference almost any probabilistic model with Markov Chain Monte Carlo
- Tweak MCMC for the problem at hand
- Know the limitations of MCMC and when it's better to apply other methods

---

Week 5, Variational Autoencoder
> Welcome to the fifth week of the course! This week we will combine many ideas from the previous weeks and add some new to build Variational Autoencoder -- a model that can learn a distribution over structured data (like photographs or molecules) and then sample new data points from the learned distribution, hallucinating new photographs of non-existing people. We will also the same techniques to Bayesian Neural Networks and will see how this can greatly compress the weights of the network without reducing the accuracy.

- Learn how and why to combine CNNs with Bayesian methods
- Use scalable variational inference to hallucinate new images with VAE
- Tune millions of hyperparameters with Bayesian Neural Networks

---

Week 6, Gaussian processes & Bayesian optimization
> Welcome to the final week of our course! This time we will see nonparametric Bayesian methods. Specifically, we will learn about Gaussian processes and their application to Bayesian optimization that allows one to perform optimization for scenarios in which each function evaluation is very expensive: oil probe, drug discovery and neural network architecture tuning.

- Understand random processes
- Train Gaussian processes
- Select hyperparameters for machine learning models