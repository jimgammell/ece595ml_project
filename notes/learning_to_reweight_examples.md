# Abstract
 - DNNs are susceptible to statistical issues in training dataset -- class imbalance, label noise
   - Highly-general function approximators which can overfit to these things
 - Previous approaches entail careful hyperparameter tuning
 - This approach: gradient descent on example weights to minimize loss on clean validation dataset
   - Automatic, easy to do, no hyperparameters
   - Appealing in cases where you have lots of data with questionable quality, and can ensure quality on a small subset of the data

# Introduction
 - DNNs are susceptible to training dataset biases
   - Joint distribution of training dataset differs from that of population dataset
   - Common example: class imbalances, i.e. different amount of representation for each class
   - Common example: label noise, i.e. labels are generally correct but sometimes wrong
 - Previous approach: dataset resampling
   - Manually choose correct proportion of each class to go into training dataset
   - Assign a weight to examples which corrects disproportionate representation, and minimize weighted loss
 - Contradiction in resampling
   - Noisy samples: want to heavily weight examples with low loss, since these are more-likely to be correct
   - Unbalanced classes: want to heavily weight examples with high loss, since these tend to be from minority class
   - In general not possible to correctly resample without knowing unbiased set properties
 - Necessary to have small, unbiased validation dataset to guide weighting decisions
   - Common in practice to have a small high-quality dataset and a large questionable-quality dataset
 - Proposed approach: dynamically reweight examples to minimize loss on unbiased validation set
   - Done at every training iteration

# Related work
 - Many methods reweight examples to emphasize harder ones
   - Importance sampling, AdaBoost, hard example mining, focal loss
 - Robust loss estimators emphasize easier examples
   - Self-paced learning, curriculum learning, others
 - Many papers exploring the effects of training set bias, label noise, class imbalance on DNNs
 - Similar to meta-learning insofar as it learns to learn better
   - Resembles other methods which optimize a meta-objective which may be validation loss
   - Differs because has no hyperparameters to optimize, and does not require offline training

# Method
 - Online approximation of meta-learning objective
   - Can fit into regular supervised training
   - Convergence rate the same as that of SGD ($$1/\epsilon^2$$)
 - Context
   - Have N training examples (x_i, y_i) including M<<N examples constituting clean validation dataset (x_i^v, y_i^v)
   - Neural network \Phi(x; \theta) with cost function C(y, \hat{y}) where \hat{y}=\Phi(x; \theta)
   - Denote f_i(\theta)=C(y_i, \hat{y_i})
 - Aim to learn reweighting of inputs so \theta^*(w)=argmin_\theta \sum w_if_i(\theta)
   - w^* = argmin_w>=0 \sum f_i^V(\theta^*(w))
   - i.e. seek to find w such that optimal theta under w minimizes loss on validation dataset
 - Naive computation of w^* requires nested optimization of w and theta -- very expensive
   - Instead use online approximation which requires only one loop
   - At each iteration: examine descent direction on batch, and reweight according to similarity with direction on validation loss
 - Can construct optimization problem for w: w^*=argmin_w \sum f_i^V(\theta^*(w)).
   - Too time-consuming to be practical, so instead consider single step in direction of gradient starting from w=0.
   - Rectify resulting rates, as negative weights can lead to instability.
   - Normalize so sum to 1 (unless all 0, in which case set to 0). Makes choice of learning rate arbitrary.
 - Easy to implement using auto-differentiation tools
 - Approximately triples time per epoch
   - Two forward, backward passes -- one on training, one on validation
   - One backward-on-backward pass, which is similar time to one forward pass
   - One additional backward pass
 - Can demonstrate formally that this converges
   - Assumes that validation loss is Lipschitz smooth with gradient bounded by sigma
   - Validation loss monotonically decreases, with stagnation in expectation only at a critical point
   - Gradient of validation loss less than epsilon in O(1/epsilon^2) timesteps (same as for SGD)

# Experiments
 - Explore effectiveness with both class imbalance, noisy labels, both
   - MNIST and CIFAR benchmarks with deep CNNs
 - MNIST class imbalance experiments
   - 5000 images from classes 4 and 9, with 9 dominating
   - Compare with several standard class imbalance methods:
     - Weight examples in inverse proportion to their overrepresentation relative to unbiased dataset
     - Resample examples to form class-balanced minibatches
     - Hard mining -- select highest-loss examples from majority class
     - Randomly weight examples
   - Significantly outperforms these baselines
 - CIFAR-10 noisy label experiments
   - Two label-noise settings
     - Uniformly flip a class label to any other class label
     - Flip labels to a single background class label (constitutes class imbalance since background class usually dominates distribution)
   - Compared to several baselines
     - REED -- bootstrapping technique where training target is convex combo of prediction and label
     - S-model -- add fully-connected softmax layer to model noise transition matrix
     - MentorNet -- RNN-based model which outputs example weights from sequence of losses
     - Randomly assign weights
     - Reweight training loss based on known proportion of clean images for class
   - Find that randomly reweighting examples outperforms all other baselines on uniform flip task
     - Hypothesize that this acts as a regularizer







