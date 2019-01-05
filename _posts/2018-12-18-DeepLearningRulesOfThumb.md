---
layout: single
title:  "Deep Learning Rules of Thumb"
excerpt: "Practical tips for deep learning"
date:   2018-12-18 21:04:11 -0500
categories: post
classes: wide
header:
    image: /assets/images/deepLearningNotes.jpg
---

# Note: This is an initial draft and not the final version

When I first learned about neural networks in grad school, I asked my professor if there were any rules of thumb for choosing architectures and hyperparameters. I half expected his reply of “well, kind of, but not really” – there are a lot more choices for neural networks than there are for other machine learning algorithms after all! I kept thinking about this when reading through [Ian Goodfellow, Yoshua Bengio, and Aaaron Courville’s Deep Learning](https://www.deeplearningbook.org/) book, and decided to compile a list of rules of thumbs listed throughout this book. As it turns out, there are a lot of them (especially since there are a lot of types of neural networks and tasks they can accomplish). The funny thing is that a lot of these rules of thumb aren’t very heavily established – deep learning is still a relatively new active area of research, so a lot of ones listed below are just things that researchers may have recently discovered. Beyond that, there are a lot of areas in this book where the authors will either state (in more academic terms) “we don’t really know why this works, but we can see that it does” or “we know this isn’t the best way, but it is an active area of research and we don’t know any better ways at the moment”. 

Below are the more practical notes that I have taken throughout reading [Deep Learning](https://www.deeplearningbook.org/). I included a TL:DR at the top to hit on the most important points, and I’d recommend skipping to **Section 10: Practical Methodology** for some of the more important parts. 

This isn’t a book review for [Deep LearningA](https://www.deeplearningbook.org/), but I would personally recommend it if you’re looking to learn a more in depth understanding of the more established methodologies as well as the active areas of research (at the time of its publishing). Jeremy Howard of [fast.ai](https://course.fast.ai/) (an excellent source for learning the practical side of deep learning) criticized this book due to focusing too much on the math and theory, but I found that it did a good job explaining the intuition behind concepts and practical methodologies in addition to all of the math formulas that I skipped over. 

# TL:DR

-   Use transfer learning if possible. If not, and working on a problem that’s been studied extensively, start by copying the architecture.
    -   Network architecture should always be ultimately decided with experimentation and determined by the validation error.
    - Deeper (more layers), thinner (smaller layers) networks are more difficult to optimize but tend to yield better generalization error.
-   Always use early stopping.
    -   Two early stopping methods:
        1.  Re-train the model again with new parameters on the entire dataset, and stop when hitting same number of training steps as the previous model at the early stopping point.
        2. Keep the parameters obtained at the early stop, continue training with all the data, and stop when the average training error falls below the training error at the previous early stopping point.
-   It’s probably a good idea to use dropout.
    -   Use a 0.8 keep probability on the input layers and 0.5 for hidden layers.
    -   Dropout may require larger networks that need to be trained with more iterations.
-   ReLUs are the ideal activation function. They have flaws, so using leaky or noisy ReLUs could yield performance gains at the cost of having more parameters to tune..
-   You need at least 5,000 observations per category for acceptable performance (>=10 million for human performance or better).
    -   Use k-folds cross validation instead of train/validation/test split if you have less than 100,000 observations.
-   Use as large of a batch size as your GPU’s memory can handle.
    -   Try different batch sizes by increasing in powers of 2 starting with 32 (or 16 for really large models) and going up to 256.
-   Stochastic gradient descent with momentum and a decaying learning rate is a good optimization algorithm to start with.
    -   Common values for the  <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> hyperparameter for momentum are 0.5, 0.9, and 0.99. It can be adapted over time, starting with a small value and raising to larger values.
    -   Alternatively, use ADAM or RMSProp.
    -   Use asynchronous SGD if doing distributed deep learning.
-   The learning rate is the most important hyperpameter. If bound by time, focus on tuning it.
    -   The learning rate can be picked by monitoring learning curves that plot the objective function over time.
    -   The optimal learning rate is typically higher than the learning rate that yields the best performance after the first ~100 iterations, but not so high that it causes instability..
-   For computer vision:
    -   Use data augmentation as long as flipping the images doesn’t fundamentally change the image. Contrast normalization is another safe pre-processing step.
    -   Batch normalization, pooling, and padding are common tools to use with convolutional neural networks. Batch normalization may make dropout redundant.
-   For natural language processing:
    -   Long short term memory (LSTM) networks typically outperform other neural networks.
    -   Pre-trained word embeddings (ex. word2vec, word2glove, etc.) are powerful.
-   Random search typically converges to good hyperparameters faster than grid search.
-   Debugging strategies:
    -   __Visualize the model in action:__ Look at samples of images and what the model detects. This helps determine if the quantitative performance numbers are reasonable.
    -   __Visualize the worst mistakes:__ This can reveal problems with pre-processing or labeling.
    -   __Fit a tiny dataset when the training error is high:__ This helps determine genuine underfitting vs. software defects .
    -   __Monitor a histogram of activations and gradients:__ Do this for about one epoch. This tells us if the units saturate and how often. The gradient should be about 1% of the parameter.


# Full comprehensive notes

## I) Applied Math and Machine Learning Basics

### 1. Introduction
-   At least 5,000 observations per category are typically required for acceptable performance. <sub>(pg. 20)</sub>
    -   More than 10,000,000 observations per category are typically required for human performance or better. <sub>(pg. 20)</sub>

### 4. Numerical Computation
-   In deep learning we typically settle for local minima rather than the global minima because of complexity and non-convex problems. <sub>(pg. 81)</sub>

### 5. Machine Learning Basics 
-   If your model is at optimal capacity and there is still a gap between training and testing error, gather more data. <sub>(pg. 113)</sub>
-   20% of a training set is typically used for the validation set. <sub>(pg. 118)</sub>
-   Use k-folds cross validation instead of a train/test split if the dataset has less than 100,000 observations. <sub>(pg. 119)</sub>
-   When using mean squared error (MSE), increasing capacity lowers bias but raises variance. <sub>(pg. 126)</sub>
-   Bayesian methods typically generalize much better when limited training data is available, but they typically suffer from high computational cost when the number of training examples is large. <sub>(pg. 133)</sub>
-   The most common cost function is the negative log-likelihood. As a result, minimizing the cost function causes maximum likelihood estimation. <sub>(pg. 150)</sub>

## II) Deep Networks: Modern Practices 

### 6. Deep Feedforward Networks
-   Rectified linear units (ReLU) are the default activation function for feed forward neural networks. <sub>(note: not explicitly stated, but alluded to)</sub>
    -   They are based on the principle that models are easier to optimize if their behavior is closer to linear. <sub>(pg. 188)</sub>
    -   Sigmoidal activations should be used when ReLUs aren’t possible. Ex. RNNs, many probabilistic models, and some autoencoders. <sub>(pg. 190)</sub>
-   Cross entropy is preferred over mean squared error (MSE) or mean absolute error (MAE) in gradient based optimization because of vanishing gradients. **double check this one** <sub>(citation needed) </sub>
-   **ReLU advantages:** Reduced likelihood of vanishing gradients, sparsity, and reduced computation. <sub>(pg. 187)</sub>
    -   **ReLU disadvantages:** Dying ReLU (leaky or noisy ReLUs avoid this, but introduce additional parameters that need to be tuned). <sub>(pg. 187)</sub>
-   Large gradients help with learning faster, but arbitrarily large gradients result in instability. <sub>(citation needed)</sub>
-   Network architectures should ultimately be found via experimentation guided by monitoring the validation set error. <sub>(pg. 192)</sub>
-   Deeper models reduce the number of units required to represent the function and reduce generalization error. <sub>(pg. 193)</sub>
    -   Intuitively, models with deeper layers are preferred because we are learning a series of functions on the overall function. <sub>(pg. 195)</sub>

### 7. Regularization for Deep Learning
-   It’s optimal to use different regularization coefficients at each layer, but use the same weight decay for all layers. **check if this should be reworded** <sub>(citation needed)</sub>
-   Use early stopping. It’s an efficient hyperparameter to tune, and it will prevent unnecessary computations. <sub>(pg. 241)</sub>
    -   Two methods for early stopping:
        1.    Re-train the model again with new parameters on the entire dataset, and stop when hitting same number of training steps as the previous model at the early stopping point. <sub>(pg. 241)</sub>
        2. Keep the parameters obtained at the early stop, continue training with all the data, and stop when the average training error falls below the training error at the previous early stopping point. <sub>(pg. 242)</sub>
-   Model averaging (bagging, boosting, etc.) almost always increase predictive performance at the cost of computational power. <sub>(pg. 249)</sub>
    -   Averaging tends to work well because different models will usually not make all the same errors on the test set. <sub>(pg. 249)</sub>
    -   Dropout is effectively a way of bagging by creating sub-networks. <sub>(pg. 251)</sub>
        -   Dropout works better on wide layers because it reduces the chances of removing all of the paths from the input to the output. <sub>(pg. 252)</sub>
        -   Common dropout probabilities for keeping a node are 0.8 for the input layer and 0.5 for the hidden layers. <sub>(pg. 253)</sub>
        -   Models with dropout need to be larger and need to be trained with more iterations. <sub>(pg. 258)</sub>
        -   If a dataset is large enough, dropout likely won’t help too much. <sub>(pg. 258)</sub>
            -   Additionally, dropout is not very effective with a very limited number of training samples (ex. <5,000). <sub>(pg. 258)</sub>
        -   Batch normalization also adds noise which has a regularizing effect and may make dropout unnecessary. <sub>(pg. 261)</sub>

### 8. Optimization for Training Deep Models
-   **Mini-batch size (i.e. batch size):** Larger batch sizes offers better gradients, but are typically limited by memory. <sub>(pg. 272)</sub>
    -   I.e. make your batch size as high as your memory can handle. <sub>(pg. 272)</sub>
    -   With GPUs, increase the batch size by a power of 2, from 32 to 256. Try starting with a batch size of 16 for larger models. <sub>(pg. 272)</sub>
    -   Small batch sizes can have a regularizing effect due to noise, but at the cost of adding to the total run time. These cases require smaller learning rates for increase stability. <sub>(pg. 272)</sub>
-   Deep learning models have multiple local minima, but it’s ok because they all have the same cost. The main problem is if the cost at the local minima is much greater than the cost at the global minima. <sub>(pg. 277)</sub>
    -   You can test for problems from local minima by plotting the norm of the gradient and seeing if it shrinks to an extremely small value over time. <sub>(pg. 278)</sub>
    -   Saddle points are much more common than local minima in high dimensional non-convex functions. <sub>(pg. 278)</sub>
        -   Gradient descent is relatively robust to saddle points. <sub>(pg. 279)</sub>
-   Gradient clipping is used to stop exploding gradients. This is a common problem for recurrent neural networks (RNNs). <sub>(pg. 281)</sub>
-   Pick learning rate by monitoring learning curves that plot the objective function over time. <sub>(pg. 287)</sub>
-   Optimal learning rate is higher than the learning rate that yields the best performance after the first ~100 iterations. <sub>(pg. 287)</sub>
    -   Monitor the first few iterations and go higher than the best performing learning rate while avoiding instability. <sub>(pg. 287)</sub>
-   It doesn’t seem to matter much if initial variables are randomly selected from a Gaussian or uniform distribution. <sub>(pg. 293)</sub>
    -   However, the scale does. Larger initial weights help avoid redundant units, but initial weights that are too large have a detrimental effect. <sub>(pg. 294)</sub>
    -   Weight initialization can be treated as hyperparameter - specifically the initial scale and if they are sparse or dense. <sub>(pg. 296)</sub>
        -   Look at the range or standard deviation of activations or gradients on one minibatch for picking the scale. <sub>(pg. 296)</sub>
-   There is no one clear optimization algorithm that outperforms the others - it primarily depends on user familiarity of hyperparameter tuning. <sub>(pg. 302)</sub>
    -   Stochastic gradient descent (SGD), SGD with momentum, RMSProp, RMSprop with momentum, AdaDelta, and Adam are all popular choices. <sub>(pg. 302)</sub>
        -   RMSProp is an improved version of AdaGrad (pg. 299) and is currently one of the go-to optimization methods for deep learning practitioners. <sub>(pg. 301)</sub>
            -   *Note:* RMSProp may have high bias early in training. <sub>(pg. 302)</sub>
        -   Common values for the <img src="https://latex.codecogs.com/gif.latex?\alpha" title="\alpha" /> hyperparameter for momentum are 0.5, 0.9, and 0.99. <sub>(pg. 290)</sub>
            -   This hyperparameter can be adapted over time, starting with a small value and raising to larger values. <sub>(pg. 290)</sub>
        -   Adam is generally regarded as being fairly robust to the choice of hyperparameters. <sub>(pg. 302)</sub>
            -   However, the learning rate may need to be changed from the suggested default. <sub>(pg. 302)</sub>
-   Apply batch normalization to the transformed values rather than the input. Omit the bias term if including the <img src="https://latex.codecogs.com/gif.latex?\beta" title="\beta" /> learnable parameter. <sub>(pg. 312)</sub>
    -   Apply range normalization in and at every spatial location for convolutional neural networks (CNNs). <sub>(pg. 312)</sub>
-   Networks that are more thin and deep are more difficult to train, but they have better generalization error. <sub>(pg. 317)</sub>
-   It is more important to choose a model family that is easy to optimize than a powerful optimization algorithm. <sub>(pg. 317)</sub>

### 9. Convolutional Networks
-   Pooling is essential for handling inputs of varying size. <sub>(pg. 338)</sub>
-   Zero padding allows us to control the kernel width and output size independently to stop shrinkage, which would be a limiting factor. <sub>(pg. 338)</sub>
-   The optimal amount of zero padding for test set accuracy usually lies between:
    1.  “Valid convolutions” where no zero padding is used, the kernel is always entirely in the image, but output shrinks every layer. <sub>(pg. 338)</sub>
    2. “Same convolutions” where enough zero padding is used to keep the size of the output equal to the size of the input. <sub>(pg. 338)</sub>
-   One potential way to evaluate convolutional architectures is to use randomized weights and only train the last layer. <sub>(pg. 352)</sub>

### 10. Sequence Modeling: Recurrent and Recursive Nets
-   Bidirectional RNNs have been extremely successful in handwriting recognition, speech recognition, and bioinformatics. <sub>(pg. 383)</sub>
-   Compared to CNNs, RNNs applied to images are typically more expensive but allow for long-range lateral interactions between features in the same feature map. <sub>(pg. 384)</sub>
-    Whenever a RNN has to represent long-term dependencies, the long term interactions gradients have an exponentially smaller magnitude than the gradients of the short term interactions. <sub>(pg. 392)</sub>
-   The technologies used to set the weights in echo state networks could be used to initialize weights in a fully trainable RNN – an initial spectral radius of 1.2 and sparse initialization perform well. <sub>(pg. 395)</sub>
-   The most effective sequence models in practice are gated RNNs – including long short-term memory units (LSTMs) and gated recurrent units <sub>(pg. 397)</sub>
    -   LSTMs learn long-term dependencies more easily than simple RNNs. <sub>(pg. 400)</sub>
    -   Adding a bias of 1 to the forget gate makes the LSTM as strong as gated recurrent network (GRN) variants. <sub>(pg. 401)</sub>
    -   Using SGD on LSTMs typically takes care of using second-order optimization methods to prevent second derivatives vanishing with the first derivatives. <sub>(pg. 401)</sub>
-   It is often much easier to design a model that is easier to optimize than it is to design a more powerful algorithm. <sub>(pg. 401)</sub>
-   Regularization parameters encourage “information flow” and prevents vanishing gradients. However, gradient clipping is needed for RNNs to also prevent gradient explosion (which would prevent learning from succeeding). <sub>(pg. 404)</sub>
    -   However, this is not very effective for LSTMs with a lot of data, e.g. language modeling. <sub>(pg. 404)</sub>

### 11. Practical Methodology
-   It can be useful to have a model refuse to make a decision if it is not confident, but there is a tradeoff. <sub>(pg. 412)</sub>
    -   Coverage is the fraction of samples for which the machine learning sample is able to produce a response for, and it is a tradeoff with accuracy. <sub>(pg. 412)</sub>
-   ReLUs and their variants are the ideal activation functions for baseline models. <sub>(pg. 413)</sub>
-   A good choice for a baseline optimization algorithm is SGD with momentum and a decaying learning rate. <sub>(pg. 413)</sub>
    -   Decay schemes include:
        -   Linear decay until fixed minimum learning rate <sub>(pg. 413)</sub>
        -   Exponential decay <sub>(pg. 413)</sub>
        -   Decreasing by a factor of 2 to 10 each time the validation error plateaus <sub>(pg. 413)</sub>
-   Another good baseline optimization algorithm is ADAM. <sub>(pg. 413)</sub>
-   If considering batch normalization, introduce it ASAP if optimization appears problematic. <sub>(pg. 413)</sub>
-   If training set samples <10 million, include mild regularization from the start. <sub>(pg. 413)</sub>
    -   Almost always use early stopping. <sub>(pg. 413)</sub>
    -   Dropout is a good choice that works well with most architectures. Batch normalization is a possible alternative. <sub>(pg. 413)</sub>
-   If working on a problem similar to one that’s been extensively studied, it may be a good idea to copy that architecture, and maybe even copy the trained model. <sub>(pg. 414)</sub>
-   If unsupervised learning is known to be important to your application (e.g. word embedding in NLP) then include it in the baseline. <sub>(pg. 414)</sub>
-   **Determining when to gather more data:**
    -   If training performance is poor, try increasing the size of the model and adjusting the learning algorithm. <sub>(pg. 414)</sub>
        -   If it is still poor after this, it is a data quality problem. Start over and collect cleaner data or more rich features. <sub>(pg. 414)</sub>
    -   If training performance is good but testing performance is bad, gather more data if it is feasible and inexpensive. Else, try reducing the size of the model or improve regularization. If these don’t help, then you need more data. <sub>(pg. 415) </sub>
        -   If you can’t gather more data, the only remaining alternative is to try to improve the learning algorithm. <sub>(pg. 415) </sub>
        -   Use learning curves on a logarithmic scale to determine how much more data to gather. <sub>(pg. 415) </sub>
-   The learning rate is the most important hyperparameter because it controls the effective model capacity in a more complicated way than other hyperparameters. If bound by time, tune this one. <sub>(pg. 417)</sub>
    -   Tuning other hyperpameters requires monitoring both training and testing error for over/under fitting. <sub>(pg. 417)</sub>
        -   If training error is higher than the target error, increase capacity. If not using regularization and are confident that the optimization algorithm is working properly, add more layers/hidden units. <sub>(pg. 417)</sub>
        -   If testing error is higher than the target error, regularize. Best performance is usually found on large models that have been regularized well. <sub>(pg. 418)</sub>
-   As long as your training error is low, you can always decrease generalization error by collecting more training data. <sub>(pg. 418)</sub>
-   **Grid search:** Commonly selected when tuning less than four hyperpameters. <sub>(pg. 420)</sub>
    -   It’s typically best to select values on a logarithmic scale and use it repeatedly to narrow down values. <sub>(pg. 421)</sub>
    -   Computational cost is exponential with the number of hyperparameters, so even parallelization may not help out adequately. <sub>(pg. 422)</sub>
-   **Random search:** Simpler to use and converges to good hyperparameters much quicker than grid search. <sub>(pg. 422)</sub>
    -   Random search can be exponentially more efficient than grid search when there are several hyperparameters that don’t strongly affect the performance measure. <sub>(pg. 422)</sub>
    -   We may want to run repeated versions of it to refine the search based off of previous results. <sub>(pg. 422)</sub>
    -   Random search is faster than grid search because it doesn’t waste experimental runs. <sub>(pg. 422)</sub>
-   Model based hyperparameter tuning isn’t universally recommended because it rarely outperforms humans and can catastrophically fail. <sub>(pg. 423)</sub>
-   **Debugging strategies:**
    -   Visualize the model in action. I.e. look at samples of images and what the model detects. This helps determine if the quantitative performance numbers are reasonable. <sub>(pg. 425)</sub>
    -   Visualize the worst mistakes. This can reveal problems with pre-processing or labeling. <sub>(pg. 425)</sub>
    -   Fit a tiny dataset when the training error is high. This helps determine genuine underfitting vs. software defects. <sub>(pg. 426)</sub>
    -   Monitor a histogram of activations and gradients: Do this for about one epoch. This tells us if the units saturate and how often. <sub>(pg. 427)</sub>
        -   Compare magnitude of gradients to parameters. The gradient should be about ~1% of the parameter. <sub>(pg. 427)</sub>
            -   Sparse data (e.g. NLP) have some parameters that are rarely updated. Keep this in mind. <sub>(pg. 427)</sub>

### 12. Applications
-   When using a distributed system, use asynchronous SGD. The average improvement of each step is lower, but the increased rate of production of steps causes this to be faster overall <sub>(pg. 435)</sub>
-   Cascade classifiers is an efficient approach for object detection. One classifier with high recall <img src="https://latex.codecogs.com/gif.latex?\rightarrow" title="\rightarrow" /> another with high precision. Ex. locate street sign <img src="https://latex.codecogs.com/gif.latex?\rightarrow" title="\rightarrow" /> transcribe address <sub>(pg. 437)</sub>
-   One way to reduce inference time in an ensemble approach is to train a “gater” that selects which specialized network should make the inference <sub>(pg. 438)</sub>
-   Standardizing pixel ranges is the only strict preprocessing required for computer vision <sub>(pg. 441)</sub>
-   Contrast normalization is often a safe computer vision preprocessing step <sub>(pg. 442)</sub>
    -   Global contrast normalization (GCN) is one way to do this, but it can reduce edge detection within lower contrast areas (ex. within the dark section of an image) <sub>(pg. 442 & 444)</sub>
        -   Scaling parameters can either be set to 1 or chosen to make each individual pixel have a standard deviation across examples close to 1 <sub>(pg. 443)</sub>
        -   Datasets with closely cropped images can safely have <img src="https://latex.codecogs.com/gif.latex?\lambda&space;=&space;0" title="\lambda = 0" /> and <img src="https://latex.codecogs.com/gif.latex?\epsilon&space;=&space;10^-8" title="\epsilon = 10^-8" /> <sub>(pg. 443)</sub>
        -   Datasets with small randomly cropped images need higher regularization. Ex. <img src="https://latex.codecogs.com/gif.latex?\lambda&space;=&space;10" title="\lambda = 10" /> and <img src="https://latex.codecogs.com/gif.latex?\epsilon&space;=&space;0" title="\epsilon = 0" /> <sub>(pg. 443)</sub>
    -   Local contrast normalization can usually be implemented effectively by using separable convolution to compute feature maps of local means/standard deviations, then using element-wise subtraction/division on different feature maps <sub>(pg. 444)</sub>
        -   These typically highlight edges more than global contrast normalization <sub>(pg. 445)</sub>
-   In NLP, hierarchical softmax tends to give worse test results than sampling-based methods in practice <sub>(pg. 457)</sub>

## III) Deep Learning Research

### 13. Linear Factor Models
-   Linear factor models can be extended into autoencoders and deep probabilistic models that do the same tasks but are much more flexible and powerful <sub>(pg. 491)</sub>

### 14. Autoencoders
-   Sparse autoencoders are good for learning features for other tasks such as classification <sub>(pg. 496)</sub>
-   Autoencoders are useful for learning which latent variables explain the input. They can also learn useful features <sub>(pg. 498)</sub>
-   While many autoencoders have one encoder/decoder layer, they can take the same advantages of depth as feedforward networks <sub>(pg. 499)</sub>
    -   This especially applies when enforcing constraints such as sparsity <sub>(pg. 500)</sub>
    -   Depth exponentially reduces computational cost and the amount of training data for representing some functions <sub>(pg. 500)</sub>
    -   A common strategy for training a deep autoencoder is to greedily pre-train the deep architecture by training a stack of shallow autoencoders <sub>(pg. 500)</sub>

### 15. Representation Learning
-   In machine learning, a good representation is one that makes a subsequent learning task easier <sub>(pg. 517)</sub>
    -   Ex. supervised feed forward networks: each layer learns better representations for the classifier from the last layer <sub>(pg. 518)</sub>
-   Greedy layer-wise unsupervised training can help with classification test error, but not many other tasks <sub>(pg. 520)</sub>
    -   Is not useful with image classification, but very important in NLP (ex. word embedding) because of a poor initial representation <sub>(pg. 523)</sub>
    -   From a regularizer perspective, it’s most effective when the number of labeled examples are very small or the number of unlabeled examples are very large <sub>(pg. 523)</sub>
    -   Likely to be most useful when the function to be learned is extremely complicated <sub>(pg. 523)</sub>
    -   Use the validation error from the supervised phase to select the hyperparameters of the pretraining phase <sub>(pg. 526)</sub>
    -   Unsupervised pretraining has mostly been abandoned except for NLP <sub>(pg. 526)</sub>
-   Transfer learning, multitask learning, and domain adaptation can be achieved with representation learning when there are features useful for different tasks/settings corresponding to underlining factors appearing in multiple settings <sub>(pg. 527)</sub>
-   Distributed representations can have a statistical advantage to non-distributed representations when an already complicated structure can be compactly represented using a smaller number of parameters <sub>(pg. 540)</sub>
    -   Some traditional non-distributed algorithms generalize because of the smoothness assumption, but this suffers from the curse of dimensionality <sub>(pg. 540)</sub>

### 16. Structured Probabilistic Models for Deep Learning
-   Structured probabilistic models provide a framework for modeling only direct interactions between random intervals, which allow the models to have significantly fewer parameters thus being estimated reliably from less data and having a reduced computational cost for storing the model, performing inference, and drawing samples <sub>(pg. 553)</sub>
-   Many generative models in deep learning have no latent variables or only use one layer of latent variables, but use deep computational graphs to define the conditional distributions within a model <sub>(pg. 575)</sub>
    -   This is in contrast to most deep learning applications where there tend to be more latent variables than observed variables, of which are learned for nonlinear interactions <sub>(pg. 575)</sub>
-   Latent variables in deep learning are unconstrained but difficult to interpret outside of rough characterization via visualization <sub>(pg. 575)</sub>
-   Loopy belief propagation is almost never used in deep learning because most deep learning models are designed to make Gibbs sampling or variational inference algorithms efficient <sub>(pg. 576)</sub>

### 17. Monte Carlo Methods
-   Monte Carlo Markov Chains (MCMC) can be computationally expensive to use because of the time required to “burn in” the equilibrium distribution and to keep every n sample in order to assure your samples aren’t correlated <sub>(pg. 589)</sub>
-   When sampling from MCMC in deep learning, it is common to run a number of parallel markov chains that is equal to the number of samples in a minibatch, and then sample from these. 100 is a common number <sub>(pg. 589)</sub>
-   Markov chains will reach equilibrium, but we don’t know how long until it does or when it does. We can test if it has mixed with heuristics like manually inspecting samples or measuring correlation between successive samples <sub>(pg. 589)</sub>
-   While the Metropolis-Hastings algorithm is often used with Markov chains in other disciplines, Gibbs sampling is the de-facto method for deep learning <sub>(pg. 590)</sub>

### 19. Approximate Inference
-   Maximum a posteriori (MAP) inference is commonly used with a feature extractor and learning mechanism, primarily for sparse coding models <sub>(pg. 628)</sub>

### 20. Deep Generative Models
-   Variants of the Boltzmann machine surpassed the popularity of the original long ago <sub>(pg. 645)</sub>
    -   Boltzmann machines act as a linear estimator for observed variables, but are more powerful for unobserved variables <sub>(pg. 646)</sub>
-   When initializing a deep Boltzmann machine (DBM) from a stack of restricted Boltzmann machines (RBMs), it’s necessary to modify the parameters slightly <sub>(pg. 648)</sub>
    -   Some kinds of DBMs may be trained without first training a set of RBMs <sub>(pg. 648)</sub>
-   Deep belief networks are rarely used today due to being outperformed by other algorithms, but are studied for their historical significance <sub>(pg. 651)</sub>
    -    While deep belief networks are generative models, the weights from a trained DBN can be used to initialize the weights for a MLP for classification as an example of discriminative fine tuning <sub>(pg .653)</sub>
-   Deep Boltzmann machines have been applied to a variety of tasks, including document modeling <sub>(pg. 654)</sub>
-   Deep Boltzmann machines trained with a stochastic maximum likelihood often result in failure when initialized with random weights. The most popular way to overcome this is greedy layer-wise pre-training. Specifically, train each layer in isolation as an RBM, with the first layer on the input data and each subsequent layer as an RBM on samples from the previous layer’s RBM’s posterior distribution <sub>(pg. 660)</sub>
    -   The DBM is then trained with PCD, which typically only makes small changes in the parameters and model performance <sub>(pg. 661)</sub>
    -   Modifications for state of the art results for a DBM (write these b/c they were too long to write) <sub>(pg. 662)</sub>
    -   Two ways to get around this procedure <sub>(pg. 664)</sub>:
        1.  Centered Deep Boltzmann Machine: Reparametrizing the model to make the Hessian of the cost function better conditioned at the beginning of the learning process
            -  These yield great samples, but the classification performance is not as good as an appropriately regularized MLP
        2.  Multi Prediction Deep Boltzmann Machine: Uses an alternative training criterion that allows the use of the back-propagation algorithm which avoids problems with MCMC estimates of the gradient
            -  These have superior classification performance, but do not yield very good samples
-   In the context of generative models, undirected graph models (DBM, DBN, RBM, etc.) have overshadowed directed graph models (GANs, Sigmoid Belief Networks, Variational Autoencoders, etc.) until roughly 2013 when directed graph models started showing promise <sub>(pg. 682)</sub>
-   While variational autoencoders (VAEs) are simple to implement, they often obtain excellent results, and are among the state-of-the-art for generative modeling, samples from VAEs training on images tend to be somewhat blurry, and the cause for this is not yet known <sub>(pg. 688)</sub>
-   Unlike Boltzmann machines, VAEs are easy to extend, and thus have several variants <sub>(pg. 688)</sub>
-   Non-convergence is a problem that causes GANS to underfit (one network reaches a local minima and the other reaches a local maxima), but the extent o this problem is not yet known <sub>(pg. 691)</sub>
-   For GANs, best results are typically obtained by re-formulating the generator to aim to increase the log-probability that a generator makes a mistake rather than decrease the log-probability that the generator makes the right prediction (pg. 692)</sub>
-   While GANs have an issue with stabilization, they typically perform very well with carefully a selected model architecture and hyperparameters <sub>(pg. 693)</sub>
-   One powerful variation of GANs, LAPGAN, start with a low-resolution image and add details to it. The output of this often fools humans <sub>(pg. 693)</sub>
-   In order to make sure the generator in a GAN doesn’t apply a zero probability to any point, add Gaussian noise to all images in the last layer <sub>(pg. 693)</sub>
-   Always use dropout in the discriminator of a GAN. Not doing so tends to yield poor results <sub>(pg. 693)</sub>
-   Visual samples from generative moment matching networks are disappointing, but can be improved by combining them with autoencoders <sub>(pg. 694)</sub>
-   When generating images, using the “transpose” of the convolution operator often yields more realistic images while using fewer parameters than using fully connected layers without parameter sharing <sub>(pg. 695)</sub>
-   Even though the assumptions for the “un-pooling” operation in convolutional generative networks are unrealistic, the samples generated by the model as a whole tend to be visually pleasing <sub>(pg. 696)</sub>
-   While there are several approaches to generating samples with generative models, MCMC sampling, ancestral sampling, or a mixture of the two are the most popular <sub>(pg. 706)</sub>
-   When comparing generative models, changes in preprocessing, even small and subtle ones, are completely unacceptable because it can change the distribution and fundamentally alters the task <sub>(pg. 708)</sub>
-   If evaluating a generative model by viewing the sample images, it is best to have this done by an experimental subject that doesn’t know the source of the samples <sub>(pg. 708)</sub>
    -   Additionally, because a poor model can produce good looking samples, make sure the model isn’t just copying training images <sub>(pg. 708)</sub>
        -   Check this with the nearest neighbor to images in the training set according to the Euclidean distance for some samples <sub>(pg. 708)</sub>
-   A better way to evaluate samples from a generative model is to evaluate the log-likelihood that the model assigns to the test data if it is computationally feasible to do so <sub>(pg. 709)</sub>
    -   This method is still not perfect and has pitfalls. For example, constants in simple images (ex. blank backgrounds) will have a high likelihood <sub>(pg. 709)</sub>
-   There are many uses for generative models, so selecting the evaluation metric should depend on the intended use <sub>(pg. 709)</sub>
    -   Ex. some generative models are better at assigning a high probability to the most realistic points, and others are better at rarely assigning high probabilities to unrealistic points <sub>(pg. 709)</sub>
    -   Even when restricting metrics to the task it’s most suited for, all of the metrics currently in use have serious weaknesses <sub>(pg. 709)</sub>
