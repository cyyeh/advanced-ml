# How to Win a Data Science Competition: Learn from Top Kagglers

## Overview

If you want to break into competitive data science, then this course is for you! Participating in predictive modelling competitions can help you gain practical experience, improve and harness your data modelling skills in various domains such as credit, insurance, marketing, natural language processing, sales’ forecasting and computer vision to name a few. At the same time you get to do it in a competitive context against thousands of participants where each one tries to build the most predictive algorithm. Pushing each other to the limit can result in better performance and smaller prediction errors. Being able to achieve high ranks consistently can help you accelerate your career in data science.

In this course, you will learn to analyse and solve competitively such predictive modelling tasks. 

When you finish this class, you will:

- Understand how to solve predictive modelling competitions efficiently and learn which of the skills obtained can be applicable to real-world tasks.
- Learn how to preprocess the data and generate new features from various sources such as text and images.
- Be taught advanced feature engineering techniques like generating mean-encodings, using aggregated statistical measures or finding nearest neighbors as a means to improve your predictions.
- Be able to form reliable cross validation methodologies that help you benchmark your solutions and avoid overfitting or underfitting when tested with unobserved (test) data. 
- Gain experience of analysing and interpreting the data. You will become aware of inconsistencies, high noise levels, errors and other data-related issues such as leakages and you will learn how to overcome them. 
- Acquire knowledge of different algorithms and learn how to efficiently tune their hyperparameters and achieve top performance. 
- Master the art of combining different machine learning models and learn how to ensemble. 
- Get exposed to past (winning) solutions and codes and learn how to read them.

Disclaimer : This is not a machine learning course in the general sense. This course will teach you how to get high-rank solutions against thousands of competitors with focus on practical usage of machine learning methods rather than the theoretical underpinnings behind them.

Prerequisites: 
- Python: work with DataFrames in pandas, plot figures in matplotlib, import and train models from scikit-learn, XGBoost, LightGBM.
- Machine Learning: basic understanding of linear models, K-NN, random forest, gradient boosting and neural networks.

## Key Conceps by Week

**Week 1: Introduction & Recap**

> This week we will introduce you to competitive data science. You will learn about competitions' mechanics, the difference between competitions and a real life data science, hardware and software that people usually use in competitions. We will also briefly recap major ML models frequently used in competitions.

- Describe competition mechanics
- Compare real life applications and competitions
- Summarize reasons to participate in data science competitions
- Describe main types of ML algorithms
- Describe typical hardware and software requirements
- Analyze decision boundaries of different classifiers
- Use standard ML libraries

**Additional Resources**

- Recap of main ML algorithms
  - Overview of methods
    - [Scikit-Learn (or sklearn) library](http://scikit-learn.org/)
    - [Overview of k-NN (sklearn's documentation)](http://scikit-learn.org/stable/modules/neighbors.html)
    - [Overview of Linear Models (sklearn's documentation)](http://scikit-learn.org/stable/modules/linear_model.html)
    - [Overview of Decision Trees (sklearn's documentation)](http://scikit-learn.org/stable/modules/tree.html)
    - [Overview of algorithms and parameters in H2O documentation](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science.html)
    - [Explanation of Random Forest](https://www.datasciencecentral.com/profiles/blogs/random-forests-explained-intuitively)
    - [Explanation/Demonstration of Gradient Boosting](http://arogozhnikov.github.io/2016/06/24/gradient_boosting_explained.html)
    - [Example of kNN](https://www.analyticsvidhya.com/blog/2018/03/introduction-k-neighbours-algorithm-clustering/)
  - Additional Tools
    - [Vowpal Wabbit repository](https://github.com/JohnLangford/vowpal_wabbit)
    - [XGBoost repository](https://github.com/dmlc/xgboost)
    - [LightGBM repository](https://github.com/Microsoft/LightGBM)
    - [Interactive demo of simple feed-forward Neural Net](http://playground.tensorflow.org/)
    - Frameworks for Neural Nets: Keras,PyTorch,TensorFlow,MXNet, Lasagne
    - [Example from sklearn with different decision surfaces](http://scikit-learn.org/stable/auto_examples/classification/plot_classifier_comparison.html)
    - [Arbitrary order factorization machines](https://github.com/geffy/tffm)
- Software/Hardware requirements
  - StandCloud Computing
    - AWS, Google Cloud, Microsoft Azure
  - AWS spot option
    - [Overview of Spot mechanism](http://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)
    - [Spot Setup Guide](http://www.datasciencebowl.com/aws_guide/)
  - Stack and packages
    - Basic SciPy stack (ipython, numpy, pandas, matplotlib)
    - Jupyter Notebook
    - [Stand-alone python tSNE package](https://github.com/danielfrg/tsne)
    - Libraries to work with sparse CTR-like data: [LibFM](http://www.libfm.org/), [LibFFM](https://www.csie.ntu.edu.tw/~cjlin/libffm/)
    - Another tree-based method: RGF ([implemetation](https://github.com/baidu/fast_rgf), [paper](https://arxiv.org/pdf/1109.0887.pdf))
    - Python distribution with all-included packages: Anaconda
    - [Blog "datas-frame" (contains posts about effective Pandas usage)](https://tomaugspurger.github.io/)
- Feature preprocessing and generation with respect to models
  - Feature preprocessing
    - [Preprocessing in Sklearn](http://scikit-learn.org/stable/modules/preprocessing.html)
    - [Andrew NG about gradient descent and feature scaling](https://www.coursera.org/learn/machine-learning/lecture/xx3Da/gradient-descent-in-practice-i-feature-scaling)
    - [Feature Scaling and the effect of standardization for machine learning algorithms](http://sebastianraschka.com/Articles/2014_about_feature_scaling.html)
  - Feature generation
    - [Discover Feature Engineering, How to Engineer Features and How to Get Good at It](https://machinelearningmastery.com/discover-feature-engineering-how-to-engineer-features-and-how-to-get-good-at-it/)
    - [Discussion of feature engineering on Quora](https://www.quora.com/What-are-some-best-practices-in-Feature-Engineering)
- Feature extraction from text and images
  - Feature extraction from text
    - Bag of words
      - [Feature extraction from text with Sklearn](http://scikit-learn.org/stable/modules/feature_extraction.html)
      - [More examples of using Sklearn](https://andhint.github.io/machine-learning/nlp/Feature-Extraction-From-Text/)
    - Word2vec
      - [Tutorial to Word2vec](https://www.tensorflow.org/tutorials/word2vec)
      - [Tutorial to word2vec usage](https://rare-technologies.com/word2vec-tutorial/)
      - [Text Classification With Word2Vec](http://nadbordrozd.github.io/blog/2016/05/20/text-classification-with-word2vec/)
      - [Introduction to Word Embedding Models with Word2Vec](https://taylorwhitten.github.io/blog/word2vec)
    - NLP Libraries
      - [NLTK](http://www.nltk.org/)
      - [TextBlob](https://github.com/sloria/TextBlob)
  - Feature extraction from images
    - Pretrained models
      - [Using pretrained models in Keras](https://keras.io/applications/)
      - [Image classification with a pre-trained deep neural network](https://www.kernix.com/blog/image-classification-with-a-pre-trained-deep-neural-network_p11)
    - Finetuning
      - [How to Retrain Inception's Final Layer for New Categories in Tensorflow](https://www.tensorflow.org/tutorials/image_retraining)
      - [Fine-tuning Deep Learning Models in Keras](https://flyyufelix.github.io/2016/10/08/fine-tuning-in-keras-part2.html)


**Week 2: Exploratory Data Analysis**

> We will start this week with Exploratory Data Analysis (EDA). It is a very broad and exciting topic and an essential component of solving process. Besides regular videos you will find a walk through EDA process for Springleaf competition data and an example of prolific EDA for NumerAI competition with extraordinary findings.

- Describe the major visualization tools
- Generate hypotheses about data
- Inspect the data and find golden features
- Examine and analyze various plots and other data visualizations

**Additional Resources**

- Exploratory data analysis
    - Visualization tools
        - Seaborn
        - Plotly
        - Bokeh
        - ggplot
        - Graph visualization with NetworkX
    - Others
        - Biclustering algorithms for sorting corrplots
- Validation
    - [Validation in Sklearn](http://scikit-learn.org/stable/modules/cross_validation.html)
    - [Advices on validation in a competition](http://www.chioka.in/how-to-select-your-final-models-in-a-kaggle-competitio/)

**Week 3: Metrics Optimization**

> This week we will first study another component of the competitions: the evaluation metrics. We will recap the most prominent ones and then see, how we can efficiently optimize a metric given in a competition.

- Describe the role of correct metric optimization method in a competition
- Analyze new metrics
- Create constant baselines
- Recall the most important classification and regression metrics
- Describe what libraries can be used to optimize a particular metric

**Week 4: Hyperparameter Optimization**

> In this module we will talk about hyperparameter optimization process. We will also have a special video with practical tips and tricks, recorded by four instructors.

- List most important hyperparameters in major models; describe their impact
- Understand the hyperparameter tuning process in general
- Arrange hyperparameters by their importance

**Week 5: Competitions go through**

> For the 5th week we've prepared for you several "walk-through" videos. In these videos we discuss solutions to competitions we took prizes at. The video content is quite short this week to let you spend more time on the final project. Good luck!

- Increase your expertise by assessing solutions of other people
- Analyse winning solutions of various competitions
- Compare approaches to solving data science competitions
