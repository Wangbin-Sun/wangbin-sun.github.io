---
layout: post
title: "Notes-Deep Learning"
description: "Notes of Deep Learning by Ian Goodfellow, Yoshua Bengio and Aaron Courville"
tags: [machine learning, deep learning, book]
image:
  path: /images/abstract-4.jpg
  feature: abstract-4.jpg
---

## Notes of "Deep Learning" by Ian Goodfellow, Yoshua Bengio and Aaron Courville


# Chap. 1 Introduction
- [Solution towards AI] It allows computers to learn from experience and understands the world in terms of **a hierarchy of concepts**, with each concept defined through its relation to simpler concepts
- [Crucial components in Machine Learning] the representation of the data they are given, or called "Feature"
    - [Representation Learning]  Use machine learning to discover not only the mapping from representation to output but also the representation itself.
- [Historical Wave] cybernetics -> connectionism + neural networks -> deep learning

# Chap. 4 Numerical Computation
- [Poor condition] For $A\in R^{n\times n}$, its condition number is $max_{i,j}(\| \frac{ \lambda _i}{\lambda _j} \|)$.When this number is large, matrix inversion is particularly sensitive to error in the input.
- [order of optimizaiton] Optimization algorithms that use only the gradient, such as gradient descent, are called first-order optimization algorithms. Optimization algorithms that also use the Hessian matrix, such as Newton’s method, are called second-order optimization algorithms 

## PROBLEM-4
- 牛顿法矩阵形式推导？
    - 牛顿法本质是求解一函数的零点，延伸到这里，求解一阶导数的零点相当于寻找最值。证明可利用“图解”或者二阶泰勒展开。 
- KKT条件的推导？
    - 强调互补松弛性 

# Chap. 6 Deep Feedforward Networks
- [Deep Feedforward Networks] also called feedforward neural networks or multilayer perceptrons (MLPs). It does not have feedback connections
- [Choose mapping $\phi$ options]
    - It should be generic, avoiding overfitting
    - Manually engineer in conventional way
    - As for deep learning, it is learned
- [Nonlinear transformation] usually followed by a fixed nonlinear function called an activation function
- [Learning component] like machine learning, including optimizaiton procedure, cost function and model familiy
    - Difference in optimization, loss function will be nonconvex due to nonlinear propertity of units, so iterative method required
- [Gradient-Based Learning] No global convengency assured, and initial parameters are crucial
    - Maximum cross entropy simplifies the choosing of cost function
    - Negative log-likelihood helps to solve vanishing gradient problem
    -  mean squared error and mean absolute error often lead to poor results when used with gradient-based optimization
- [Output Units] Linear, Sigmoid, Softmax, Other
- [Hidden Units] ReLU is widely used(without much consideration)
    - Although at 0 it is not differentiable, it doesn't matter as objective won't arrive it
    - Most only distinguish on activation function
    - [Maxout units] Divide the input into k groups and output the maximum.
    - Before ReLU, logistic sigmoid and hyperbolic tangent is used as activation function
- [Architecture] main consideration is on depth and width
    - [Universal Approximation Properties] a feedforward network introduced here can approximate any Borel measurable function
    - Two core process, learn and generalize
    - Hidden lay with activation function can be regared as "fold" the complex space
        - In general, deep model leads to high generalization
- [Backpropagation] Information flow back to help calculate the gradient
    - Not only vectors, all tensors can use this method
    - it performs on the order of one Jacobian product per node in the graph
    - Two categories: symbol-to-symbol and symbol-to-numeric, the former introduces the computational graph
    - It is a table-filling strategy, a.k.a., dynamic programming

## PROBLEM-6
- Cross-entropy的计算与其本质，与条件熵的区别？交叉熵比MSE好在哪里？
    - D(p\| \| q) = H(p) + 交叉熵H(p,q)。最小化DL Divergence等同于最小化交叉熵。
- 为什么最大似然估计能够学习条件概率？
    - 本身就以条件概率的形式表示 
- 线性单元输出为何能表示高斯分布？为什么最大化log似然与最小化MSE等价？
    - min -Elog(y\| x) 可推导成 min E(y - y^)^2 
- 混合密度网络是如何学习的？其中的高斯分布、协方差矩阵、精度矩阵有什么关系？
    - A: $N(x;\mu;\sigma^2) = \sqrt{\frac{1}{2\pi\sigma^2}}exp(-\frac{1}{2\sigma^2}(x-\mu)^2)$ 高斯分布需要求标准差平方的倒数，引入精度$\beta\in(0,\infty)$避免计算，原本的分布转变成$N(x;\mu;\beta^{-1}) = \sqrt{\frac{\beta}{2\pi}}exp(-\frac{1}{2}\beta(x-\mu)^2)$。多维高斯分布的协方差矩阵同样可转变成精度矩阵，更为高效，无需求逆。由$N(x;\mu;\Sigma) = \sqrt{\frac{1}{(2\pi)^ndet(\Sigma)}}exp(-\frac{1}{2}(x-\mu)^T\Sigma^{-1}(x-\mu))$转变成$N(x;\mu;\beta^{-1}) = \sqrt{\frac{det(\beta)}{(2\pi)^n}}exp(-\frac{1}{2}(x-\mu)^T\beta(x-\mu))$
- 矩阵表示的反向传播推导？
    - 存储中间计算的梯度，用于前一层梯度的传递 
- Hessian矩阵在深度学习的应用，另外Krylov方法是？
    - A: H矩阵可以应用于泰勒级数。 Krylov可以迭代近似求解矩阵的逆、特征值、特征向量（只利用矩阵乘法）
- 矩阵求导如何定义？
    - 基本上是逐元素进行，关键是考虑其排列的方式 
-  6.23公式-令y=1/0代入可推导 

# Chap. 7 Regularization for Deep Learning
- [Nature of Regularization] Regularization of an estimator works by trading increased bias for reduced variance.
- [Parameter penalties] Only the weights are imposed penalty, leaving the bias. The bias controls a single variable, the penalty on it may cause under-fitting.
    - [$L^2$ Regularization] a.k.a., weight decay. Other communities name it ridge regression or Tikhonov regularization.
        - Decay is along the eigenvector(OF H), the scaling factor is $\frac{\lambda_i}{\lambda_i+\alpha}$, the decay effect is less significant on larger $\lambda_i$.
        - L2 regularization causes the learning algorithm to “perceive” the input X as having higher variance
    - [$L^1$ Regularization] generating more sparse solution, so it can be used to select feature (eg. LASSO)
    - Too much weight decay will trap the network in a bad local minimum 
- [Dataset augmentation] Injecting noise in the input also works
    - It makes the model more robust.
    - Label smoothing is imposing on the output, functioning as regularization
- [Sparse Representation] Place a penalty on the activations of the units in a neural network, encouraging their activations to be sparse.
    - Mechanism is the same as parameter regularization, imposing penalty on objective function($\Omega(h)$) 
- [Ensemble techniques] combining several models to achieve low generalization error.
    - Bagging (short for bootstrap aggregating), helps regularization
    - Boosting, on the contrary, drives the capacity up.
- [Dropout] Similar to bagging,  it provides a computationally inexpensive but powerful method of regularizing a broad family of models 
    - take advantage of shared weight from parental structure.
    - [weight scaling inference rule] approximate $p_{ensemble}$ by evaluating $p(y\| x)$ in the model with all units, then multiply the weight with the probability of including the unit.
        - ensure that the expected total input to a unit at test time is roughly the same as the expected total input to that unit at train time 
- [Adversarial training]  explicitly introduce a local constancy prior into supervised neural nets. 

## PROBLEM-7
- Weight-decay中最优权重的推导与无正则条件下的对比 ;L1解析解的结论推导
    - 利用泰勒展开拟合最优点的二次曲线，讨论权重的表达式。对于L1，同样可利用泰勒公式，通过一定假设大致说明其稀疏性的结论
- reproject重投影与参数惩罚、显式约束的关系？
    - 重投影是和梯度下降算法相关的 
- 半监督学习中，生成模型、判别模型的关系？参数共享的含义是什么？
    - 生成模型x,y，判别模型y\| x；半监督通常是表示学习
- Early stopping 与 L2 regularization的关系推导？
    - Early stopping相当与在逼近无正则化的最优点过程中停下，停留的点为L2下的最优点 (参数选择满足一定条件)
- 流形的含义是什么，正切传播、双反向传播和对抗训练的关系是什么？
    - 将聚集的点想像成流形面？可以计算切线之类的……不使用欧式距离来定义
- 整理可以应用的正则表达工具包
    - 见目录 

# Chap. 8 Optimization for Training Deep Models
- [Difference]We reduce a different cost function $J(\theta)$ in the hope that doing so will improve P. It is indirect.
    - We hope to minimize the expectation on the data-generating distribution $p_{data}$: $J^*(\theta) = E_{(x,y)\sim p_{data}}L(f(x;\theta),y)$ 
    - However, the true distribution is unknown, so we can only minimize empirical risk instead, which may easily lead to overfitting
- [Challenge in Optimization] Optimization problem may be non-convex.
    - Ill-condition Hessian matrix
    - Model identifiability issues mean that a neural network cost function can have an extremely large or even uncountably infinite amount of local minima.
    - For many random functions, in low- dimensional spaces, local minima are common. In higher-dimensional spaces, local minima are rare, and saddle points are more common. 
        - Commonly, saddle points have higher cost than local minima
        - Newton method is not suitable, as it finds the point where the gradient is zero
    - Using gradient clipping to avoid cliffs with exploding gradients
    - Choose appropriate surrogate loss function to approximate true loss, and can be more accurately estimated
- [Basic algorithm] 
    - [SGD]Learning rate should be smaller as the training process goes
        - In practice, on iteration $\tau$,$\epsilon_k = (1-\alpha)\epsilon_0+\alpha\epsilon_\tau$,$\alpha = \frac{k}{\tau}$ 
    - [Momentum] SGD is kind of slow, this method is aimed to accelerate learning.
        - It solves two problems: poor conditioning of the Hessian matrix and variance in the stochastic gradient 
        - It accumulates the former gradient
        - Nesterov momentum method apply the current velocity before the gradient evaluated
    - For adapative learning rates algorithms, not much theoretical difference.  
- [Newton method] Approximate the point using 2-order Taloy formular. Calculating the inverse of H as the updating value
    - Utilizing conjugate directions to avoid calculating inversion
        - the seaching direction has to be vertical agaist the former one
    - BFGS takes advantage of newton method, while overcomte its shortage(inversion calculation), it performs better on time but memory is required more
- [block coordinate decent] Optimize the variable in term, when there's more than one optimization variable and the function has good properties like convex

## PROBLEM-8
- Hessian矩阵在优化中的应用，ill-condition的含义是什么？牛顿法的应用
    - Hessian就是二阶导数。病态条件使得梯度卡住，梯度不会显著缩小，但二次项增长会超过一个数量级。 
- 鞍点的梯度定义是什么？
    - 梯度为0，但Hessian矩阵有正负特征值 
- 理解Adam算法，矩阶数、偏置修正等
    - 引入指数加权、一二阶矩估计，并修正偏差
- 整理可以应用的优化方法工具包
    - 看目录 

# Chap. 9 Convolutional Networks
- [Convolution operation]x and w are the function of t, $s(t) = (x * w)(t) = \int{x(a)w(t-a)}da$
- [Motivation] sparse interactions, parameter sharing and equivariant representations are three import ideas by convoution
    - [sparse interactions] instead of full connection, node only inteacts with limited number of nodes in the next layer. This leads to the phenonmenan that deep node has a high receptive field, although the connection is indirect.
    - [parameter sharing] the weight matrix is shared between nodes and layers, which reduce the number of parameters to be estimated
    - [equivariant representations] the input changes, the output changes in the same way. A function f(x) is equivariant to a function g if f(g(x)) = g(f(x))
- [pooling] a typical layer consists of three stages: convolution stage, detector stage and pooling stage
    - pooling helps to make the representation invariant. Invariance to local translation identifies the significant feature while discard the location information
    - Pooling is able to deal with input with different dimension. Summarize the input and it can output the same dimension.
- [Downsampling] sample only every s(called "stride") pixels in each direction in the output
- [Weight sharing] locally connected layers: no sharing at all; tiled convolution: a sharing list cycling; tranditional convolution: all sharing

## PROBLEM-9
- 在进行convolution, ReLU和pooling后，特征维度的变化是怎么样的？如何选取卷积层的核？卷积层和仿射变换是什么关系？
    - https://zhuanlan.zhihu.com/p/29119239
- 卷积层的矩阵表示，及反向传播参数推导？
    - 反向传播，核心是一致的 

# Chap. 10 Sequence Modeling: Recurrent and Recursive Nets ***
- [Weight in RNN] Parameter sharing makes it possible to extend and apply the model to examples of different forms
- [Back-propagation through time(BPTT)] applied to the unrolled graph with O($\tau$) cost, with connections between hidden layers
- [Teacher Forcing] Models that have recurrent connections from their outputs leading back into the model may be trained with Teacher Forcing
- The price recurrent networks pay for their reduced number of parameters is that optimizing the parameters may be difficult.
- [Bidirectional RNNs] we want to output a prediction of y(t) that may depend on the whole input sequence.
    - Hidden layers are seperated as two direction 
- [Recursive Neural Networks] It is a tree structure, rather than simple chain structure like RNN. Actually, it extends the original RNN idea.
- [Main challenge] Long-term dependence, which means that gradients propagated over many stages tend to either vanish (most of te time) or explode (rarely, but with much damage to the optimization). It is more significant in the RNN structure.
- [Reservoir computing] e.g. "echo state networks". The recurrent hidden units do a good job of capturing the history of past inputs, and only learn the output weights.
    - As the input matrix and hidden layer matrix are difficult to learn
    - Spectral radius is the maximum absolute eigenvalue of Jacobians. It measure the scaling factor.
- [Gated recurrent unit] create paths through time that have derivatives that neither vanish nor explode. 
    - Leaky units manually choose constant or set them as parameters. Gated RNNs generalize this to connection weights that may change at each time step. 
    - [LSTM cell] LSTM recurrent networks have “LSTM cells” that have an internal recurrence (a self-loop)
        - Including forget gate, external input gate, output gate

## PROBLEM-10
- BPTT和Teacher Forcing的核心区别是什么？为什么计算时间会不同？如何将这两个训练方法结合起来？
- BPTT的矩阵梯度传播推导
- 为什么基于上下文时，模型会出现如此调整？context含义是什么？
- 已有架构的梳理，输入长度？输出长度？
- 谱半径的理解，反向传播中扰动的意义？扰动的缩放含义是？
- 回声状态网络中，动态系统如何应用？Jacobian矩阵的角色是什么？
- LSTM矩阵表示推导？结构整理
- RNN优化的核心，解决长期依赖问题的整理？包括梯度爆炸和梯度消失
- 基于外显记忆的NTM是如何实现的，如何评估记忆单元？

# Chap. 11 Practical Methogology
- [Recommended Procedure] Determine the goal -> Estabilish a pipeline -> Instrument the system -> Iteratively make changes
- [Performance Metrics] Choose the ideal level of performance; choose the metric used
    - PR curve is used when precision and recall are calculated
- [Parameter tuning]
    - [Manually] Effective capacity constrained by :the representational capacity of the model, the ability of the learning algorithm to successfully minimize the cost function used to train the model, and the degree to which the cost function and training procedure regularize the model
        - Learning rate is the most important hyper-parameter

## PROBLEM-11
- 总结实践的方法，注意细节点？

# Chap. 12 Application
- [Large Scale Learning]
    - [GPU] have a high degree of parallelism and high memory bandwidth, at the cost of having a lower clock speed and less branching capability relative to traditional CPUs.
    - [asynchronous stochastic gradient descent]  allow multiple machines to compute multiple gradient descent steps in parallel
    - [Dynamic Structure]  Data-processing systems can dynamically determine which subset of many neural networks should be run on a given input
        - use a cascade of classifiers
        - use a neural network called the gater
        - use a switch
- [Computor Vision]
    - [Preprocession]
        - [Contrast Normalization] includes Global contrast normalization (project the data point to a sphere) , sphering (or whitening) and Local contrast normalizaiton(focus on the edges)
- [Speech Recognition]
    - [Classic Models] HMM, GMM, RBMs
- [Natural Language Processing] 
    - [n-gram] requries smoothing
    - [word embedding] a kind of neural language model(NLM)
    - [High dimensional outputs]
        - Use of a Short List
        - Hierarchical Softmax
        - Importance Sampling
            - not only useful for speeding up models with large softmax outputs. More generally, it is useful for accelerating training with large sparse output layers 

## PROBLEM-12
- 分层SoftMax的机制是什么？如何展开训练？对于模型的影响有哪些？（联想到了huffman编码）
- 重要采样的公式理解与实现？重要采样对于稀疏向量处理的优势？
    - softmax更快采样 
    - n-gram的例子含义

# Chap. 13 Linear Factor Models
- [Linear Factor Model] the use of a stochastic linear decoder function that generates x by adding noise to a linear transformation of h.
    - Different models make differenet choices about the noise and the prior p(h)
- [Probabilistic PCA] Assume the conditional variances equal to each other, i.e., the nose is a multivariate Gaussian noise.
    - most variations in the data can be captured by the latent variables h, up to some small residual reconstruction error $\sigma^2$
    - probabilistic PCA becomes PCA as $\sigma \to 0$.
- [Independent Component Analysis] Model linear factors that seeks to separate an observed signal (fully independent) into many underlying signals that are scaled and added together to form the observed data.
- [Slow Feature Analysis] a linear factor model that uses information from time signals to learn invariant features
    - [Slowness Principle] the important characteristics of scenes change very slowly compared to the individual measurements that make up a description of a scene.
- [Sparse coding]  an unsupervised feature learning and feature extraction mechanism 
    - Sparse coding is not a parametric autoencoder. 

## PROBLEM-13
- 主成分分析的矩阵推导（第二章）？
- PCA中，x的高斯分布参数推导？PCA是去掉噪声的因子分析吗？需要对于线性变换有秩的要求吗？
- PCA、概率PCA的关系？
    - A：PCA是将x进行线性组合，构造分量x’；而概率PCA是基于因子分析的内容，因子分析是寻找潜分量h，使得h的线性组合加上噪声等于x，而概率PCA则是令噪声的方差等于0。
- 为什么在ICA中不用高斯分布，W会遇到什么问题？
    - 因为高斯的线性组合仍然是高斯，所以分解不唯一。ICA的目求出W
- 为什么稀疏编码不能使用最大似然法训练模型？
- 稀疏编码如何推导至L1范数的优化目标？
- 生成模型的含义？
    - 联合概率；判别模型是条件概率 

# Chap. 14 Autoencoders
- [autoencoder] a neural network that is trained to attempt to copy its input to its output
    - two parts: an encoder function h = f(x) and a decoder that produces a reconstruction r = g(h).
- [undercomplete autoencoder] code dimension is less than the input dimension
- [Regularized autoencoder] use a loss function that encourages the model to have other properties besides the ability to copy its input to its output
    - [Sparse autoencoder] impose a sparsity penalty, although it is different from weight decay
    - [Denoising autoencoder (DAE)] noise is added into input, and DAE must eliminate the noise itself 
        - By sampling, learn the distribution
- [contractive autoencoder] Regularize the derivative of f
- [Predictive sparse decomposition]  a hybrid of sparse coding and parametric autoencoders 
- Autoencoder can be applied in dimensionality reduction, which will benefit informaiton retrieval tasks, classification tasks

## PROBLEM-14
- MAP 近似贝叶斯推断是什么含义？为什么前馈网络的权重衰减属于这一类？正则自编码器的正则为何含义不同？
- 随机编码器、随机解码器的核心是什么？引入的概率分布与其他自编码器有什么区别？
- DAE的loss function为何是-log
    - 最大似然？ 
- 基于RBM的模型和去噪得分匹配的关系是什么？自编码器又有什么关系？为何重构函数是g(f(x))
- 自编码器与流形学习之间的关系
- 非参模型、分布式表示、深度学习捕获流形结构的区别
- CAE感觉有点类似泰勒展开，用线性拟合非线性？
- 正则自编码器的两股力量推动学习，不同自模型似乎不太一样？

# Chap. 15 Representation Learning

* Shared presentation benefits the learning process
    * The core idea of representation learning is that the same representation may be useful in both settings.  
* [Greedy layer-wise unsupervised pretraining] a representation learned for one task can sometimes be useful for another task
    * Every layer is optimized individually
    * In supervised learning, it can be viewed as a regularizer and a form of parameter initialization
    * Unsupervised pretraining contains massive hyper-parameters, which cannot show effect before supervised learning
    * Nowadays, apart form NLP, most algorithms do not contain this part
    * Unsupervised pretraining extends to supervised pretraining
* [Transfer learning] similar as multi-task learning, sharing input structure or output structure
* Many deep learning algorithms are motivated by the assumption that the hidden units can learn to represent the underlying causal factors that explain the data
* [Distributed representation] different from symbolic one as it can generalize due to shared attributes
    * nondistributed models generalize only locally via the smoothness prior, making it difficult to learn a complicated function with more peaks and troughs than the available number of examples 
    * information exists between labels, and they are not "atom"s but combinition of hidden features 

## Problem-15

* 为什么具有更多独立性的分布更容易建模？
* 表示学习中贝叶斯相关的概率理解？或者说核心？因果因子和解释因子的本质区别？
* 通用正则化策略，与前面章节的正则化有区别吗？
    * 似乎这里更偏向表示学习？
    


# Chap.16 Structured Probabilistic Models for Deep Learning

* [structured probabilistic model] describe a probability distribution, using a graph to describe which random variables in the probability distribution interact with each other directly.
* Structured probabilistic models provide a formal framework for modeling only direct interactions between random variables.
    * reduces the number of parameters and computational cost during model storage, model inference and model sampling
* [Directed models] As long as each variable has few parents in the graph, the distribution can be represented with very few parameters.
* [Undirected models] Edge exists when two variables affect each other without direction
    * Z is usually intractable in the context of deep learning
* We want to choose a graph that implies as many independences as possible, without implying any independences that do not actually exist.
* [Ancestral Sampling] directed graphical models has a a simple and efficient procedure to produce a sample from the joint distribution represented by the model
* The primary advantage of using structured probabilistic models is that they allow us to dramatically reduce the cost of representing probability distributions as well as learning and inference.
* Use latent variable to reduce complexity. Latent variables have advantages in efficiently capturing p(v). 
* Approximate inference is applied,, as the complexity in deep learning, even using probabilistic graph to represent it.
* Latent variables are given certain meaning in the traditional graph model, while in deep learning they have no pre-definition
    * Traditional approaches to graphical models typically aim to maintain the tractability of exact inference. 

## Problem-16

* 有向图采样时，为什么存在“其他变量”的影响？如何理解

# Chap.17 Monte Carlo Methods

* deterministic approximate algorithms or Monte Carlo approximations are used in machine learning
* The variance of importance sampling is sensitive to distribution q
* In deep learning, when p(x) is undirected model, it is hard to find a good importance sampling q(x)
    * [Markov Chain Monte Carlo] using energy-based model to sample
        * use markov chain to avoid "chicken-and-egg problem" in undirected graph.
        * Ancestral sampling is a topological way to implement
        * MCMC has large time cost, especially for burning-in and avoiding relevant samples
* MCMC easily stuck on one mode
    * using tempering to mix 

## Problem-17
* 比较（有/无偏）重要采样，MCMC，祖先采样以及（块）Gibbs采样。其中的q(x)如何得到？如何应用基于能量的模型？
* 如何理解MCMC是带有噪声的对能量函数进行梯度下降，能量函数的意义是什么？

# Chap. 18 Confronting the Partition Function

* Partition functions are integral of probability
* The main cost of the naive MCMC algorithm is the cost of burning in the Markov chains from a random initialization at each step
* Using pseudolikelihood to approximate log likelihood
* Score matching derivave models similar to certain metrics on true model, while denoising score match considers the unavalability of true model and use a noised model instead
* Noise contrastive estimation is based on the idea that a good generative model should be able to distinguish data from noise
* Annealed importance sampling (AIS) and bridge sampling addresses the shortcomings of importance sampling


## Problem-18
* 正相推高，负相压低数据分布概率的原理
* Naive MCMC，CD算法，PCD算法之间的比较
* 伪似然与负相降低之间的关系
* 得分匹配与平方误差的表达式较为相近，其内在是否存在关系？

# Chap. 19 Approximate Inference
* Interaction between hidden variables make it hard to inference (conditional probability) accurately and efficiently
* Evidence lower bound (ELBO) helps to transform the original inference problem into a optimization problem
* MAP inference means choose the most likely value as the variable

## Problem-19
* 变分推断的核心是什么？找到一个p来拟合吗？

# Chap. 20 Deep Generative Models
* Boltzman machines were an "connectionist" approach, and they are energy-based models
* Learning algorithms for Boltzmann macines are based on maximum likelihood

## Problem-20
* 生成模型在算法中的具体应用
