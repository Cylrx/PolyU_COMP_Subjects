#set page(
  paper: "a4",
  margin: (
    top: 0.17in,
    bottom: 0.17in,
    left: 0.24in,
    right: 0.24in,
  )
)
#set par(justify: true)
#set text(
  size: 6.5pt,
)

#let sps = {v(2pt)}
#let dd(y, x) = {$(diff #y) / (diff #x)$}
#let ddx(y) = {$(diff #y)/(diff x)$}
#let big(x) = {$upright(bold(#x))$}
#let code(c) = {highlight(raw(c), fill: silver)}
#let head(h) = {text(fill: maroon, weight: "black", style: "oblique")[#h]}

#columns(3, gutter: 6pt)[
  = #head("General Knowledge")
  - *Supervised Learning:*
    - Classification (Labeled Dataset)
  - *Unsupervised Learning:* 
    - Clustering (Unlabeled Dataset) 
    - Only given set of measurements, no classes
  - *Data Cleaning:*
    - Preprocess data to reduce noise
    - handle missing values
  - *Relevance Analysis:* Remove irrelevant or redundant attributes
  - *Data Transformation:* Generalize and/or normalize data

  #sps
  *Evaluating Classification Methods*
  - Prediction Accuracy
  - Speed (training speed), scalability (inference speed)
  - Robustness: ability to handle noise and missing values
  - Interpretability
  - Goodness of rules: compactness of classification rules (e.g. DT size)

  = #head("Decision Tree (DT)")
  *Two Stages* 
  1. Tree construction 
  2. Tree pruning (rm branch that reflects outliers and noise)

  == Tree Construction
  ID3/C4.5: *information gain*
  - Attributes assumed categorical
  - Can be modified for continuous-valued attributes
  IBM IntelligentMiner: *gini index*
  - Attributes assumed continuous
  - Need other tools to get possible split values
  - Can be modified for categorical attributes

  === Information Gain
  #sps
  *Definition: *
  - How much did information entropy *decrease by*?
  - Information Entropy $equiv$ impurity
  - $"Gain"(D, a)$: Information gain if we were to split samples $D$ by attribute $a$ into $V$ subsets ${D_1, D_2, dots, D_V}$.

  #sps
  *Equations*
  
  $"Information":I(x_i) = -log_2(p(x_i))$\
  #sps
  $"Information Entropy": "Ent"(X) &= E(I(X)) \
  &= sum_(i=1)^n p(x_i) I(x_i)\
  &= sum_(i=1)^n p(x_i)(-log_2(p(x_i))) \
  &= -sum_(i=1)^n p(x_i)(log_2(p(x_i)))
  $

  #sps #sps
  $"Information Gain": "Gain"(D, a) &= "Ent"(D) - sum_(v=1)^V (|D_v|)/(|D|) "Ent"(D_v)$
  $"Impurity After Split": -sum_(v=1)^V (|D_v|)/(|D|) sum_(x_i in D_v)p(x_i)log_2 p(x_i)$
  
  #sps
  Everything Together: 
  $
  &"Gain"(D, a) = \ 
  &= "Ent"(D) - sum_(v=1)^V (|D_v|)/(|D|) "Ent"(D_v)\
  &= -(sum_(i=1)^n p(x_i)log p(x_i) - sum_(v=1)^V (|D_v|)/(|D|) sum_(x_i in D_v)p(x_i)log p(x_i))
  $
  - For categorical values *$V$ is the number of distinct values of $a$*
  - $|D_v|$ is the size $v$'th split. $|D|$ is the size of the entire dataset

  == Tree Pruning & Enhancements
  - Pre-pruning: 
    - Halt tree construction early
    - Halt before _goodness measure_ falls below certain threshold. 
    - (hard to choose appropriate threshold)
  - Post-pruning: 
    - Removes branches from a "fully grown" tree. 
    - Progressively remove branches. 
    - Use testing set to decide the best pruned tree.

  #sps
  *Other Enhancements*
  
  Continuous-valued Attributes:
  - Treat each unique continuous attribute value in the dataset as discrete and identify all possible split points between them.
  - e.g. [21.6, 22.0, 23.0]. Try splits at: [21.8, 22.5]
  - Time Complexity: $O(N)$ after sort. $O(N log N)$ in total.

  Missing Values: 
  - Method 1: Assign most common value of that attribute
  - Method 2: Assign probability to each of the possible values

  Attribute Construction:
  - Create new attribute on existing ones (e.g., spare ones)
  - Purpose: Reduce fragmentation, repetition, replication

  Non-axis Parallel Split
  #image("figure-fig3.png", width: 91pt)

  #sps
  *Divide & Conquer*
  
  Internal Decision Nodes:\
  - Univariate: Use single attribute $a_i$
    - Numeric: Binary split
    - Discrete _n_-way split, for all _n_ possible values
  - Multivariate: use all attributes $a_i in A$
  Leaves:
  - Classification: class labels
  - Regression: Numeric - avg of leaf nodes / local fit (fit a more sophisticated model locally, e.g., linear regression)
  
  *Pseudocode*
  ```py
  def GenerateTree(𝒳): 
    if NodeEntropy(𝒳) < θ_I /* (See I_m on slide 29)
        Create leaf labelled by majority class in 𝒳
        return
    i ← SplitAttribute(𝒳)
    for each branch of x[i]: 
        Find 𝒳[i] falling in branch
        GenerateTree(𝒳[i]) 

  def SplitAttribute(𝒳):
      MinEnt ← MAX
      for all attributes i = 1, ..., d:
          if x[i] is discrete with n values:
              Split 𝒳 into 𝒳[1], ..., 𝒳[n] by x[i]
              e ← SplitEntropy(𝒳[1], ..., 𝒳[n])
              if e < MinEnt: 
                  MinEnt ← e
                  bestf ← i
          else: # x[i] is numeric 
              for all possible splits
                  Split 𝒳 into 𝒳[1], 𝒳[2] on x[i]
                  e ← SplitEntropy(𝒳[1], 𝒳[2])
                  if e < MinEnt:
                      MinEnt ← e
                      bestf ← i
      return bestf
  ```

  *Exam Tips (Rule Extraction Format)*
  ```
  R1:  IF (age > 38.5) AND (years-in-job > 2.5) THEN y = 0.8
  R2:  IF (age > 38.5) AND (years-in-job ≤ 2.5) THEN y = 0.6
  R3:  IF (age ≤ 38.5) AND (job-type = 'A') THEN y = 0.4
  R4:  IF (age ≤ 38.5) AND (job-type = 'B') THEN y = 0.3
  R5:  IF (age ≤ 38.5) AND (job-type = 'C') THEN y = 0.2
  ```

  #sps
  = #head("KNN (K-Nearest Neighbors)")

  == Instance-Based Methods _(a.k.a memory-based)_
  - Instance-based classification: #text(red)[Store training examples, delay processing until a new instance must be classified *("Lazy Evaluation")*]
  *Examples*: 
  - _k_-nearest neighbors: instances as points in Euclidean space.
  - Case-based reasoning: symbolic representation and knowledge-based inference

  == Procedure
  - Time Complexity: $O(n log k)$ via min-heap of size _k_
  1. Compute distance to all other training records.
  2. Sort the distance, and find _k_ nearest neighbors
  3. Use the _k_ nearest neighbors to determine the class label of the unknown sample (e.g., through majority voting)
  
  - Increasing _k_: Less sensitive to noise
  - Decreasing _k_: Capture finer structure

  == Voronoi Tessellation
  #columns(2, gutter: 6pt)[
    #image("dist-metric.png")
    $
    (a_(x_1) - b_x_1)^2 + (a_x_2 - b_x_2)^2
    $
    No weighting on either $x_1$ or $x_2$. They have equal effect on the decision boundaries.
    #colbreak()
    #image("dist-metric-stretch.png")
    $
    (a_x_1 - b_x_1)^2 + 3(a_x_2 - b_x_2)^2
    $
    Put more weighting on $x_2$, thus resulting decision boundary to be more sensitive towards $x_2$
  ]
  == Distance Metrics
  - Continuous Attributes: Euclidean Distance 
    - *normalize* each dimension by standard deviation
  - Discrete Data: use hamming distance

  == Curse of Dimensionality
  Definition: Prediction accuracy degrade quickly when number of attributes grow, because
  - When many irrelevant attributes involved, relevant attributes are shadowed, and distance measurements are unreliable
  *Solution*: 
  - Remove irrelevant attributes in pre-processing (e.g., dimensionality reduction such as PCA)

  #colbreak()
  = #head("Neural Network")

  == Perceptrons
  Step Activation: $y = cases(
    1 "if" z > 0,
    0 "otherwise"
  )
  $\
  #sps
  Update Rule: 
  $
  w <- w -alpha dd(L, w_i)\
  dd(L, big(w)) = dd(L, y) dot dd(y, z) dot dd(z, big(w))\ 
  $
  $
  &because dd(z, big(w)) = big(x), \
  &therefore dd(L, big(w)) = (dd(L, y) dot dd(y, z)) dot big(x) = delta dot big(x)
  $
  Where:
  - $delta$ is the error term
  - $L$ (Loss): Loss function
  - $alpha$: Learning rate
  - $dd(L, y)$: sensitivity of loss w.r.t. the activation output
  - $dd(y, z)$: derivative of the activation function
  - $dd(z, big(w))$: since $z = big(w)^T big(x) + b$, the derivative w.r.t. to $w$ is input vector $big(x)$

  
  #sps #sps #sps
  #text(maroon)[*Linear Activation: * _(w/ MSE Loss)_]
  $
  hat(y) &= z = big(w)^T big(x) + b;space L(y, hat(y)) = (y - hat(y))^2\
  delta &= dd(L, hat(y)) dot dd(hat(y), z) = dd(L, hat(y)) dot 1 = 2(y -hat(y)) 
  $


  
  #sps #sps #sps
  #text(maroon)[*Sigmoid Activation* _(w/ Binary Cross Entropy Loss)_ :]
  $
  hat(y) &= sigma(z) = 1/(1 + e^(-z))\
  L(y, hat(y)) &= -(y dot log hat(y) + (1 - y)log(1-hat(y)))
  $\
  Derivatives:
  $
  &"Loss": dd(L, hat(y)) = -(y/hat(y) - (1-y)/(1-hat(y))) = (hat(y) - y)/(hat(y)(1-hat(y)))\
  &"Activation": dd(hat(y), z) = hat(y)(1-hat(y)) big("or") sigma(z)(1 - sigma(z))\
  &"Error Term": delta = dd(L, hat(y)) dot dd(hat(y), z) = (hat(y) - y) / (hat(y)(1-hat(y))) dot hat(y)(1-hat(y)) \
  &= hat(y) - y big("or") sigma(z) - y\
  $
  #sps #sps
  Weight Update _(everything together)_:
  $
  big(w) <- big(w) - alpha(sigma(z) - y) dot big(x)
  $


  
  #sps #sps #sps
  #text(maroon)[*Softmax Activation* _(w/ Cross Entropy Loss)_ :]

  There are $C$ inputs (logits): ${z_1, z_2, dots, z_C}$\
  with $K$ output probability (activations): ${y_1, y_2, dots, y_C}$
  $
  &hat(y_i) = sigma(big(z))_i = e^(z_i) / (sum_(j=1)^C e^(z_j)), quad forall i in {1, 2, dots, C}\
  &L(hat(y_i), y_i) = cases(
    -log hat(y_i)   quad "if" y_i = "True",
    0 quad "        if" y_i = "False"
  ) 
  $
  
  Derivatives:
  $
  &"Loss": dd(L, hat(y_i)) = -y_i/hat(y_i) \
  &"Activation": dd(hat(y_i), z_j) = hat(y_i)(Delta_(i j) - hat(y_j)) \ 
  &"                where " Delta_(i j) = cases(1 quad "if" i = j, 0 quad "otherwise")\
  $
  $
  "Error Term": delta_j &= dd(L, z_j)\
      &= sum_(i=1)^C dd(L, hat(y_i)) dot dd(y_i, z_j)\
      &= sum_(i=1)^C (-y_i/hat(y_i)) (hat(y_i) (Delta_(i j) - hat(y_j)))\
      &= sum_(i=1)^C -y_i dot (Delta_(i j) - hat(y_j))\
      &= -sum_(i=1)^C y_i dot (Delta_(i j) - hat(y_j))\
      &= -sum_(i=1)^C y_i Delta_(i j) + sum_(i=1)^C y_i hat(y_j)\
  "1st Term":& sum_(i=1)^C y_i Delta_(i j) = y_j\
  "2nd Term":& sum_(i=1)^C y_i hat(y_j) = hat(y_j)sum_(i=1)^C y_i = hat(y_j), \
  &y "is one-hot so sum to 1"\ \ 
  big(therefore delta_j &=hat(y_j) - y_j)
  $

  #sps #sps
  Weight Update _(everything together)_: 
  $
  big(w) <- big(w) - alpha(hat(y_j) - y_j) dot big(x)
  $
]

#pagebreak()
#columns(3, gutter: 6pt)[
  == Linear Separability
  #image("linear-sep.png")

  == Training
  - Momentum: $Delta w^t = -alpha dd(L, w) + beta Delta w^(t-1)$, where $beta$ is momentum rate
  - Adaptive learning rate: Increasing $alpha$ if $L$ decreasing for a long time, else lower learning rate $alpha$

  #sps
  - Number of weights: $h(d+1) + (h+1)k$
  - Space Complexity: $O(h dot (d + k))$
  - Time complexity: $O(e dot h dot (d + k))$

  Where, 
  - $d$ is number of inputs
  - $h$ is number hidden units
  - $k$ is number of outputs
  - $e$ is number of training steps

  *Iteration Over Data*
  - Batch Gradient Descent: 
    - Entire dataset each step
    - avg error to determine gradient
  - Stochastic Gradient Descent (SGD)
    - Pick just one sample at every step
    - Update gradient only based on that single record
  - Mini-batch Gradient Descent
    - Pick batch size of $k << N$, 
    - where $N$ is the total size of the dataset

  = #head("CNN (Convolution Neural Network)")
  == Motivations
  Less learnable parameters. Better to learn small models. 
  - Sparse connection
  - Parameter sharing

  == Design
  #image("cnn.jpg")
  - Output: Binary, Multinomial, Continuous, Count
  - Input: fixed size, can use padding to make image same size
  - Architecture: choice requires experimentation
  - Optimization: Backprop
  #sps
  === Layers
  - Convolution Layer
  - Pooling Layer
  - Fully-connected (FC) Layer
  - Input and output
  === Activation
  ReLU (Rectified Linear Unit):
  $
  "ReLU"(x) = max(0, x)
  $
  Motivations: 
  - Outputs $0$ for negative values, introducing #text(red)[sparse representation]
  - Does not face vanishing gradient as with sigmoid or tanh
  - Fast: does not need $e^x$ computation
  
  === Conv Layer
  $
  n_"out" &= floor((n_"in" + 2p - k)/s) + 1\ &= ceil((n_"in" + 2p - k + 1) / s)
  $
  where,
  - $n_"in"$: number of input features
  - $n_"out"$: number of output features
  - $k$: convolution kernel size
  - $p$: padding size (on one side)
  - $s$: convolution stride size


  = Training
  == Supervised Pre-training
  - Use labeled data, work bottom up:
    - train hidden layer 1. Fix param
    - train hidden layer 2. Fix param
    - $dots$
  - Use labeled data, supervised finetune
    - Train entire network
    - Finetune to final task

  == Training w/ Max-pooling
  === Max-pooling
  $
  y_11 &= max(x_11, x_12, x_21, x_22)
  $
  #sps
  === Derivative
  For values of input feature map $x_i in {x_1, x_2, dots, x_n}$,\
  where $x_i$ is amongst the inputs of $y_j in {y_1, y_2, dots, y_m}$

  #text(red)[
    In simpler terms: 
    - $x$ is input before max-pooling, 
    - $y$ is output after max-pooling,
    - $m < n$ for obvious reasons
    - $x_i$ can be inputs of multiple $y$
  ]

  Then we have,
  $
  dd(L, x_i) = sum_(y_j space in "all max-pooling 
  windows convering" x_i) dd(L, y_j) dot dd(y_j, x_i)
  $
  The #text(red)[core idea] is that,
  $
  dd(L, x_i) = cases(
    dd(L, y) dot dd(y, x_i) quad "if" x_i = y,
    0 "           otherwise"
  )
  $

  = #head("Regularization")

  Generalization Error: $cal(L)(h) = E_((x, y)~P(x, y))[f(h(x), y)]$, where
  $h(x)$ is the learned model. It shows the expected error (using error function $f$) on the true distribution ($E_((x, y)~P(x, y))$)
  
  == Bias-Variance Tradeoff
  The following shows that the *expected* MSE error of our predictor $hat(f)$ on a single data point $x$ can be explained by bias, variance, and irreducible terms.
  $
  E[(Y - hat(f)(x))^2] 
  &= E[(f(x) + epsilon - hat(f)(x))^2]\
  &= E[(f(x) - hat(f)(x))^2] + 2E[(f(x) - hat(f)(x))epsilon] + E[epsilon^2]\
  &= underbrace("Bias"[hat(f)(x)]^2, "Bias"^2) + underbrace("Var"[hat(f)(x)], "Variance") + underbrace(sigma^2, "Irreducible Error 
  (Inherent Noise in data)")
  $

  Detailed breakdown of each term: 
  
  $2E[(f(x) - hat(f)(x))epsilon] = 2(f(x) - E[hat(f)(x)])E[epsilon] = 0$\
  
  $E[epsilon^2] &= sigma^2$  (*Note*: $E(epsilon) = 0, "Var"(epsilon) = sigma^2$, where $sigma$ is SD of noise)
  $
  E[(f(x) - hat(f)(x))^2] 
  = &E[(f(x) - E[hat(f)(x)] + E[hat(f)(x)] - hat(f)(x))^2]\
  = &E[(f(x) - E[hat(f)(x)])^2] + \
  &2E[(f(x) - E[hat(f)(x)])(E[hat(f)(x)] - hat(f)(x))] + \ 
  &E[(E[hat(f)(x)] - hat(f)(x))^2]\
  $
  The *1st* term can be turn into *Bias*. 
  Since $f(x)$ is not a random variable, but a fixed deterministic function of $x$, hence $E[f(x)] = f(x)$.
  $
  E[(f(x) - E[hat(f)(x)])^2] &= E[f(x)^2] - 2E[f(x) E[hat(f)(x)]] + E[E[hat(f)(x)]^2]\ 
  &= f(x)^2 - 2f(x)E[hat(f)(x)] + E[hat(f)(x)]^2\
  &= (f(x) - E[hat(f)(x)])^2\
  &= "Bias"[hat(f)(x)]^2
  $

  The *2nd* term can be reduced to 0. Remember that $f(x)$ is a constant and can be taken out of $E[dots]$
  $
  &2E[(f(x) - E[hat(f)(x)])(E[hat(f)(x)] - hat(f)(x))]\ 
  = & E[f(x)E[hat(f)(x)] - f(x)hat(f)(x) - E[hat(f)(x)]^2 + E[hat(f)(x)]hat(f)(x)]\ 
  = & f(x)E[hat(f)(x)] - f(x)E[hat(f)(x)] - E[hat(f)(x)]^2 + E[hat(f)(x)]^2 = 0
  $
  
  And the *3rd* term is directly *Variance*: 
  $
  E[(E[hat(f)(x)] - hat(f)(x))^2] = "Var"[hat(f)(x)]
  $

  

  #sps
  - Underfitting: high bias, low variance. High train & test error
  - Overfitting: low bias, high variance. Low train error. High test error

  == Regularization

  === Norm
  Add regularization term to the original loss function $lambda sum ||w||^2$. This encourages smaller model weights. 
  - $L_1$ norm: sum of absolute weights $||w||^1 = sum_i abs(w_i)$
  
  - $L_2$ norm: sum of squared weights $||w||^2 = sqrt(sum_i abs(w_i)^2)$ 
  - $L_p$ norm: $||w||^p = root(p, sum_i abs(w_i)^p)$

  #sps
  - Squared weight penalize large values more
  - Absolute weight penalize smaller values more
  - In general, smaller values of p (< 2) encourages sparser vectors. Larger values of p discourage large weights.

  === Dropout
  Has a $p$ value. (Suggested: 0.5 for hidden, and 0.8 for input). The number of ways to drop out for $n$ nodes: $vec(n, 1) + vec(n, 2) + dots + vec(n, n)$
  
  Conceptually: 
  1. Seen as *bagged ensemble* of exponentially many neural networks. 
   - The NN is the entire ensemble. 
   - Each sub-network formed by dropout is a base model
  2. Simulates *sparse activation*, encouraging spare representation

  === Early Stop 
  Stop training before overfitting occurs
  
  === Data Augmentation
  Artificially create more training data (e.g., rotation, trim, masking)

  === Cross Validation
  #image("crossval.png", width: 80%)
  
  === Notes
  Three kinds of error
  - Inherent: unavoidable
  - Bias: due to over-simplifications
  - Var: inability to perfectly estimate parameters from limited data
  How to reduce variance?
  - Choose a simpler classifier (like regularizer)
  - Get more training data (like data augmentation)
  - Cross-validate the parameters

  = #head("Ensemble Methods")

  *Advantages*: Improved accuracy; other types of classifiers can be directly included; Easy to implement; Not too much parameter tuning
  *Disadvantage*: Black box; Not compact representation.
  
  #columns(2, gutter: 0pt)[
    *Who Minimize Variance*:
    - Bagging
    - Random Forest
    #colbreak()
    *Who Minimize Bias*: 
    - Boosting
    - Ensemble Selection
  ]

  == Proof
  Consider binary classification problem $y in {-1, +1}$, with ground truth function $f$, base classifier $h_i$, and base classifier error rate $epsilon$: 
  
  - Base Classifier Error Rate: $P(h_i(x)!=f(x)) = epsilon$
  - Ensemble Classifier: $H(x) = "sign"(sum_i^T h_i(x))$

  Assume independence of $h_i$ error rates. From _Hoeffding Inequality_: 

  $
  P(H(x)!=f(x)) &= sum_(k=0)^(floor(T slash 2)) vec(T, k) (1-epsilon)^k epsilon^(T-k)\
  &<= exp(-1/2 T (1-2epsilon)^2)
  $
  Therefore, as the number of base classifiers $T$ increase, the probability of misclassification decrease exponentially

  == Bagging
  Bagging = *B*\ootstrap *Agg*\regation

  === Traditional Bagging
  + Takes original dataset $D$ with $N$ training examples
  + Creates $M$ *different* copies ${D_m}_(m=1)^M$ via
    - Sampling from $D$ with replacement
    - Each $D_m$ has same number of examples as $D$
  + Train base classifiers $h_1, dots, h_M$ using $D_1, dots, D_m$
  + Average / Vote model as the final ensemble $h = 1/M sum_(m=1)^M h_m$

  === Random Forest
  + Takes original dataset $D$ with $N$ training examples
  + Creates $M$ *different* copies ${D_m}_(m=1)^M$ via
    - Sampling from $D$ with replacement
    - Each $D_m$ has same number of examples as $D$
  + Train #text(fill: red, weight: "bold")[decision trees] $t_1, dots, t_M$ using $D_1, dots, D_m$
    - suppose there are $k$ features
    - #text(fill: red, weight: "bold")[randomly sample $k'$ ($<=k$) features] at each split
    - when $k' = k$ its indifferent to traditional DTs
    - usually choose $k' = log_2 k$ (or $sqrt(k)$ according to Korris)
  + Average / Vote model as the final ensemble $h = 1/M sum_(m=1)^M h_m$

  #sps
  Intuition: By randomizing not just samples (step 2), but also features (step 3), we introduce even greater variety compared to _Bagging_, reducing correlation. (The entire premise of ensemble method is independence between classifiers)

  == Boosting
  === Basic Idea
  $
  H(x) = sum_(t=1)^T alpha_t h_t (x)
  $
  - Train $T$ weak learners (through $T$ iterations)
  - Each weak learner:
    - *only* have to be slightly better than random
    - Focuses on difficult data points

  === Procedure 
  - Initialize *weights* for the $N$ samples. 
    - $D_1 = {w_(11), dots w_(1N)}$ is uniform distribution ($forall w_(1i) = 1 slash N$)
    - $D_t$ means the weighted dataset at $t$ iteration
  - *For $t = 1, 2, dots, T$ do: *
    + Train $h_t$ on weighted dataset $D_t$, and compute error of $epsilon_t$
      $
      epsilon_t = sum_(i=1)^N P_(x~D_t)(h_t (x_i) != y_i) = sum_(h_t(x_i) != y_i) w_(t i)
      $

    + Use $h_t$'s error rate $epsilon_t$ to determine its weight $alpha_t$ in the ensemble
      #columns(2)[
        $
        alpha_t = 1/2 log (1-epsilon_t) / epsilon_t
        $
        - if $epsilon_t > 0.5$, then $alpha_t < 0$ and exponentially decrease.
        - if $epsilon_t < 0.5$, then $alpha_t > 0$ and exponentially increases.
        #colbreak()
        #image("alpha_t.png")
      ]

    + Use $h_t$'s weight $alpha_t$ to determine the *next* weighted dataset $D_(t+1)$
      $
      D_(t+1) = {w_(t+1,1), dots, w_(t+1, N)}\
      w_(t+1, i) = cases(
        w_(t i) times exp(alpha_t) space "  if" h_t(x_i) != y_i "(incorrect prediction)",
        w_(t i) times exp(-alpha_t) space "if" h_t(x_i) = y_i "(correct prediction)",
      )
      $
      
      - Wrong → $w_(t+1,i)$ ↑; correct → $w_(t+1, i)$ ↓
      - Change magnitude $prop exp(alpha_t)$
      - because high $h_t$ importance $alpha_t$ requires:
        - ... much higher weights for misclassified samples
        - ... much lower weights for correct samples
        - to balance error impact

    + Normalize $D_(t+1)$ by so that it sums to 1
      $ w_(t+1, i) = w_(t+1, i) / (sum_(j=1)^N w_(t+1, j)) $
      
  - Output "boosted" ensemble $H(x) = "sign"(sum_(t=1)^T alpha_t h_t (x))$

  === XGBoost
  - All of the advantages of gradient boosting
  - CPU parallelism by default
  - Low runtime, high model performance

  == Boosting vs Bagging
  - Bagging computationally efficient
  - Both reduce variance
  - Bagging can't reduce bias, but boosting can
  - Bagging is better when we don't have high bias, and only want to reduce variance. (e.g., when overfitting)
    
  = #head("Natural Language Processing (NLP)")
  == Text Similarity
  - Stop List: _irrelevant_ words, despite high frequency (a, the, of, for...)
  - Word Stem: syntactic variants of words (drug, drugs, drugged)
  - Term freq table: T[i,j] = num. occurence of word $t_i$ in document $d_j$
  - Cosine distance: 
    - $S(v_1, v_2) = (v_1 dot v_2) / (||v_1|| dot ||v_2||)$
    - sum of element wise multiplication / product of their lengths

  == Bag of Words Model
  - Bag of Words (BoW): unordered set of words. no positional info
  - Term Frequency (tf): \# of occurence of term $t$ in document $d$
  - Document Frequency (df): \# of *documents* that contains term $t$
  - Collection Frequency (cf): \# of occur. of term $t$ across *all* documents
  - Inverse Document Frequency (idf): *Significance* of a term. 
    - $"idf"_t = log N/"df"_t$; If term $t$ only appears in small fraction of documents, $"idf"_t arrow.t$. If a lot of documents have term $t$, $"idf"_t arrow.b$
  - $"tf-idf"_(t,d) = "tf"_(t,d) times "idf"_t$. Greatest when term $t$ *occurs many times* within a *small fraction* of documents, v.v. ($therefore$ low for stop words)

  == Vector-Space Model
  - Each doc is a vector. Each dimension is *tf-idf* of a term in dictionary.
  - Issue: too many terms, too many dimensions of tf-idf
  === Dimensionality Reduction
  Latent Semantic Analysis (LSA): 
  - Use SVD, decompose *term-document matrix* $arrow$ *term-topic matrix* $times$ *topic-document matrix*
  - Low rank approximation of document-term matrix that is *loseless*
  - As 3-layer FC-NN: [Layer 1 Terms] [Layer 2 Topics] [Layer 3 Docs]

  == Word2Vec
  - Word Similarity = Word Vector (Embedding) Similarity
  - Continuos BoW: Predict mid word from surrounding (fixed window)
  - Skip-gram: Predict surrounding from mid word. (Difficult but scales)

  = #head("Clustering")
  === Distance Metrics
  - Continuous Variable: $L_1, L_2, dots, L_p$ norm
  - Binary Variable: $abs(0 - 1) + abs(1 - 0) + abs(1 - 1) dots$
  - Categorical: 1. One-hot encoding, 2. Simple matching: $("total features" - "matched features") slash "total features"$
  
  === K-Means
  Strength: 1. efficient $O(t k n)$, $t$ iterations, $k$ clusters, $n$ samples\
  Weakness: local optimum. Not good for non-convex shapes, categorical features, and noisy data (outliers).
  
  $
  "Loss" =  sum_(i=1)^k sum_(x_j in C_i) (x_j - m_i)^2\
  dd("Loss", m_i) = sum_(i=1)^k sum_(x_j in C_i) dd("", m_i)(x_j - m_i)^2 = sum_(x_j in c_i) = -2 (x_j - m_i)\
  "Let" dd("Loss", m_i) = 0, space "then" m_i = 1/abs(C_i) sum_(x_j in C_i) x_j
  " (the mean operation)"
  $

  === Hierarchical Clustering _(Agglomerative - ANGES)_
  + Initialize: 1 sample per cluster $arrow$ $N$ clusters
  + *REPEAT*: 
    - Merge a pair of clusters with *least distance*
    - Decrement \# of clusters by one 
    - *UNTIL* only $1$ cluster left.
  #sps
  _Sinle linkeage clustering_
    - aka. nearest neighbor (1NN) technique
    - Distance measured by *closest pair* of sample from each group

  _Complete Linkage Clustering_
  - aka. furthest neighbor technique
  - Distance measured by *furthest pair* of samples from each group

  _Centroid Linkage Clustering_
  - Distance measured by *mean* (centroids) of each cluster $1/n sum_(i=1)^n x_i$

  = #head("Dimensionality Reduction (DR)")
  === Principal Component Analysis (PCA)
  + Compute covariance matrix $Sigma$
  + Calculate eigenvalues of eigenvectors of $Sigma$
    - Eigenvectors w/ largest eigenvalue $lambda_1$ is $1^("st")$ principal component
    - Eigenvectors with $k^("th")$ largest eigenvalue $lambda_k$ is $k^("th")$ PC
    - Proportion of variance captured by $k^("th")$ PC = $lambda_k slash (sum_i lambda_i)$

  NOTES:
  - PCA can be seen as noise reduction. 
  - Fail when data consists of multiple clusters
  - Direction of max variance may not be most informative

  === Non-linear DR
  - ISOMAP: Isometric Feature Mapping
  - t-SNE: t-Sochastic Neighbor Embedding
  
  
  = #head("Reinforcement Learning (RL)")
  == Value Iteration
  $V ^(pi )\(s_0\)= bb(E )_(pi )\[tau|s_0]$: expected reward if start at state $s_0$, and following policy $pi$, forming a trajectory $tau$.
  $
    V^pi (s_0) & = R(s_0) + EE [ gamma R(s_1) + gamma^2 R(s_2) + dots.h|s_0 = s, pi ] \
    & = R(s_0) + EE [ gamma V^pi (s_1)|s_0 = s, pi ] \
    & = R(s_0) + gamma sum_(a_0 in A) sum_(s' in S) pi(a|s) p(s'|s_0, a_0) V^pi (s') \
    & = R(s_0) + gamma sum_(s' in S) p_(a_0 tilde.op pi(s_0))(s'|s_0, a_0) V^pi (s') 
  $

  $V^*(s)$: Optimal value function; 
  $pi^* (s)$: Optimal policy function
  $
  V^* (s) & = max_pi V^pi (s) = R(s) + max_(a in A) gamma sum_(s' in S) P_(a tilde.op pi^* (s))(s'|s, a) V^* (s') \
  pi^*(s) &= arg max_a [R(s) + gamma dot p(s'|s, a) V^*(s')]
  $
  #figure(
    image("val-it.png", width: 70%)
  )

  == Q-Function
  $Q ^(pi )\(s_0,a) =  bb(E)[tau|s_0,a]$: Similar to $V^pi (s_0)$, but stricter -- It not only starts from $s_0$, but also select action $a$ immediately thereafter
  $
    because   Q ^(pi )\(s,a) &= sum _(s 'in  S )thick  p \(s'|s, a)V ^(pi)\(s')\
    
    therefore V ^(pi )\(s \)
    &= R \(s) +  gamma sum _(a in  A )pi \(a|s)sum _(s 'in  S )thick  p \(s'|s ,a)V ^(pi )\(s '\)\ 
    
    &= R \(s \) +  gamma sum _(a in  A )pi \(a|s)thick  Q ^(pi) (s,a)\ 
 $

 === Exploration-Exploitation
 $
  pi_(k+1) (a|s) = cases(
    epsilon / abs(A) + 1 - epsilon "   "a = arg max Q(s, a), 
    epsilon / abs(A) "             otherwise."
  )
 $
 - $epsilon$ decays $1 -> 0$. $therefore pi$ would start w/ uniform distribution
 - as $epsilon -> 0$, the $epsilon/abs(A)$ across all actions no longer sum to 1.
 - We give the remaining part ($1 - epsilon$) to best $a$ (exploit)

_Korris_: $pi(a|s) prop e^(Q(s,a) slash T)$. prob. of exploit exponentially $arrow.t$ as $T arrow.b$

= #head("Semi-supervised Learning")
- *small set* of labelled training data
- *large set* of unlabeled training data
 
== Cluster-based Methods
=== _The Yarowsky Algorithm_
+ Train classifier on labeled data (e.g., DT / probabilistic model)
+ Unlabeled data with *high confidence* by classifier is *labeled*
+ Repeat step 1-2 on expanded labeled set.
+ Stop when converged 

=== _Seeded K-Mean Clustering_
- labeled data used for initialization (choosing cluster centers)
- labeled data not used in subsequent steps
- init: $C_h = 1/abs(S_h) sum_(x in S_h) x, "for" h = 1 dots k "(i.e. # of clusters)"$ 

=== _Constrained K-Mean Clustering_
- labeled data used for initialization (choosing cluster centers)
- Original labeled data unchanged. Only unlabeled are changed.

=== _COP-Kmeans_
- Weaker than partial labeling
- More compatible with data exploration
+ Given constraints: *must-link* and *cannot-link* data points
+ Cluster centers chosen randomly *without* violating constraints
+ Each step, data points reassigned to nearest cluster without violating constraints.
#image("c-k-mean.png", width: 85%)

== Data-based Methods

_Manifold Assumption_
- Geometry affects the answer
- Assumption: Data is distributed on low-dimension manifold
- *Unlabeled Data* is used to *estimate the manifold geometry*!

_Smoothness Assumption_
- Decision boundary pass through regions of low data density

  
  
= #head("Appendix")

== Derivatives
Basic Rules
$
dd(c, x) = 0
$
_where c is a constant_

$
dd(x^n, x) &= n x^(n - 1)\
dd(k * f(x), x) &= k  dd(f(x), x)\
dd(f(x) + g(x), x) &= dd(f(x), x) + dd(g(x), x)\
dd(f(x) - g(x), x) &= dd(f(x), x) - dd(g(x), x)\
dd(f(x) g(x), x) &= f(x)dd(g(x), x) + g(x)dd(f(x), x)\
dd(f(x) / g(x), x) &= (g(x)dd(f(x), x) - f(x)dd(g(x), x)) / (g(x))^2\
dd(f(g(x)), x) &= dd(f(u), u)dot dd(g(x), x) \ 
$
_where u = g(x)_

=== Derivatives of Elementary Functions

Exponential Functions
$
dd(e^x, x) = e^x\
dd(a^x, x) = a^x dot ln(a)
$

Logarithmic Functions
$
dd(ln(x), x) = 1 / x\
dd(log_a(x), x) = 1 / (x dot ln(a))
$

Hyperbolic Functions
$
dd(sinh(x), x) = cosh(x)\
dd(cosh(x), x) = sinh(x)\
dd(tanh(x), x) = sech^2(x)\
dd(coth(x), x) = -csch^2(x)\
dd(sech(x), x) = -sech(x) * tanh(x)\
dd(csch(x), x) = -csch(x) * coth(x)\
$


Special Rules
$
dd(f(g(h(x))), x) = dd(f(u), u) * dd(g(v), v) * dd(h(x), x)
$
_where u = g(v), v = h(x)_

$
"if" y = f^(-1)(x), "then" dd(y, x) = 1 / dd(f(y), y)
$

== Derivatives of #head("Loss Function") ($dd(L, hat(y))$)
- No need to consider their batched version. For example, only need to consider $(y - hat(y))^2$ for MSE. No need for $1/n sum_(i=1)^n (y - hat(y))^2$ 
- Because $n$ samples $->$ $n$ different $hat(y)$ $->$ $n$ different gradients for each weight down the chain rule.
- For any $L$ in the form of $1/n sum f(hat(y_i), y_i)$, the derivative yield by each sample is $1/n f'(hat(y_i), y_i)$. Try 
- So as the gradients accumulate for each weight $w$, they are automatically batch averaged ($1 slash n)$.




=== Mean Absolute Error (MAE)
$
L_"MAE" (y, hat(y)) = abs(y - hat(y))\ 
dd(L, hat(y)) = dd(, hat(y)) abs(y - hat(y)) = cases(
  1 "             if" y - hat(y) > 0, 
  -1 "           if" y - hat(y) < 0, 
  "undefined" "  if" y - hat(y) = 0
)
$

=== Mean Squared Error (MSE)
$
L_"MSE" (y, hat(y)) = (y - hat(y))^2 "   " dd(L, hat(y)) = 2(hat(y) - y)
$

=== Root Mean Squared Error (RMSE)
$
"RMSE" &= sqrt(1/n sum_(i=1)^n (y - hat(y)) ^2)\ 

dd("RMSE", hat(y_j)) &= 1/2(1/n sum_(i=1)^n (y - hat(y))^2)^(-1 slash 2) dot dd(, hat(y_j)) (1/n sum_(i=1)^n (y - hat(y))^2)\

&= 1/2(1/n sum_(i=1)^n (y - hat(y))^2)^(-1 slash 2) dot (2(y_j - hat(y_j))) / n\

&= 1/2 1/sqrt("MSE") dot (2(y_j - hat(y_j))) / n\

&= (hat(y_j) - y_j) / (n dot "RMSE")
$

=== Huber Loss (HL)
Let $e = y - hat(y)$: 

$
L_"HL" (y, hat(y)) = cases(
  1/2 e^2 "             if" abs(e) <= delta, 
  delta dot (abs(e) - 1/2 delta) " otherwise"
)
$

- When $abs(e) <= delta: dd(L, hat(y)) = e dot dd(e, hat(y)) = e dot (-1) = -e$

- When $abs(e) > delta: dd(L, hat(y)) = delta dot dd(, hat(y)) abs(e)$

#sps
From MAE: 

$
dd(abs(e), hat(y)) = "sign"(e) =  cases(
  1 "             if" e > 0, 
  -1 "           if" e < 0, 
  "undefined" "  if" e = 0
)
$

Finally: 

$
dd(L_"HL", hat(y)) =  cases(
  y - hat(y) "           if" abs(y - hat(y)) <= delta,
  delta dot "sign"(y - hat(y)) "if" abs(y - hat(y)) > delta
)
$

=== Log Cosh Loss
$
L(y, hat(y)) = log(cosh(hat(y) - y))\ 
$

Let $z = hat(y) - y$, then $L = log(cosh(z))$
$
dd(L, hat(y)) = dd(L, z) dot dd(z, hat(y))\

dd(L, z) = dd(, z) log(cosh(z)) = 1/ cosh(z) dot sinh(z) = tanh(z)\ 

dd(z, hat(y)) = dd(, hat(y)) (hat(y) - y) = 1\ 

therefore dd(L, hat(y)) = tanh(hat(y) - y)
$

=== Binary Cross Entropy (BCE)
$
L(y, hat(y)) = -[y log hat(y) + (1-y) log (1-hat(y))]
$
Derivative:
$
dd(L, hat(y)) &= dd(, hat(y)) [-y log hat(y)] + dd(, hat(y)) [-(1 - y) log(1 - hat(y))]\
&= -y / hat(y) - (1 - y) dot (-1)/(1-hat(y)) = (1-y) / (1-hat(y))\

therefore dd(L, hat(y)) &= - y / hat(y) + (1 - y) / (1 - hat(y))
$

=== Cross Entropy
$
L(y, hat(y)) &= - sum_(k = 1) ^K y_k log hat(y_k)\
L(y_k, hat(y_k)) &= cases(
  -log hat(y_k) "  if" y_k "is ground truth label",
  0 "          otherwise"
  
)
$

Derivatives: $ dd(L, hat(y_i)) = - y_i/hat(y_i) $

=== Hinge Loss
$
L(y, hat(y)) = max(0, 1-y dot hat(y))\
dd(L, hat(y)) = cases(
  -y " if" 1 - y hat(y)  > 0,
  0 "   otherwise"
)
$

=== Kullback-Leibler (KL) Divergence

$
L(P || Q)     = sum_(k = 1)^K P(k) log(P(k)/Q(k))
$
- $P$ is true distribution
- $Q$ is approximated distribution
- $K$ is the number of classes


#sps
== Derivative of #head("Activation Function") ($dd(hat(y), z)$)
#table(
  columns: (6em, auto, 9em, auto),
  inset: 8pt,
  align: horizon,
  table.header(
    [*Name*], [*Plot*], [*Function*], [*Derivative*],

    [identify], image("Activation_identity.svg.png"), $ x $, $ 1 $, 

    [Binary Step],
    image("Activation_binary_step.svg.png"),
    $ cases(0 "if" x < 0, 1 "if" x >= 0) $, $ 0 $,

    [Sigmoid], image("Activation_logistic.svg.png"),
    $ sigma(x) = 1/(1 + e^(-x)) $,
    $ sigma(x)(1-sigma(x)) $,
    
    [Tanh], image("Activation_tanh.svg.png"),
    $ (e^x - e^(-x)) / (e^x + e^(-x)) $,
    $ 1 - tanh^2(x) $,

    [ReLU], image("Activation_rectified_linear.svg.png"), 
    $ cases(0 "if" x <= 0, x "if" x > 0)\ = max(0, x) $, $ cases(0 "if" x < 0, 1 "if" x > 0) $,

    [GeLU], image("Activation_gelu.png"),
    $ (x(1 + "erf"(x/sqrt(2))))/2\ = x Phi(x) $, $ Phi(x) + x phi.alt (x) $,

    [Softplus], image("Activation_softplus.svg.png"), 
    $ ln(1 + e^x) $, $ 1 / (1 + e^(-x)) $,

    [ELU], image("Activation_elu.svg.png"), 
    $ cases(alpha(e^x - 1) "if" x < 0, x "if" x >= 0) $,
    $ cases(alpha e^x "if" x < 0, 1 "if" x >= 0) $,

    [SELU], image("Activation_selu.png"), 
    $ lambda dot "ELU",\ lambda = 1.0507,\ alpha = 1.67326 $, 
    $ lambda cases(alpha e^x "if" x < 0, 1 "if" x >= 0) $, 

    [PReLU], image("Activation_prelu.svg.png"), $ cases(alpha x "if" x < 0, x "if" x >= 0) $, $ cases(alpha "if" x < 0, 1 "if" >= 0) $,

    [ELiSH], image("Elish_activation_function.png"), 
    $ cases(
      (e^x - 1) / (1 + e^(-x)) "if" x < 0,
      (x) / (1 + e^(-x)) "if" x >= 0
    ) $,
    $
      cases(
        (2e^(2x) + e^(3x) - e^x) / (e^(2x) + 2e^x + 1), 
        (x e^x + e^(2x) + e^x) / (e^(2x) + 2e^x + 1)
      )
    $,

    [Gaussian], image("Activation_gaussian.svg.png"), 
    $ e^(-x^2) $, $ -2x e^(-x^2) $,

    [Softmax], [], $ sigma(arrow(x)) = e^(x_i) / (sum_(j = 1)^C e^(x_j) ) \ "for" i = 1, dots, J$,
    $ sigma(arrow(x))_i dot \ (Delta_(i j) - sigma (arrow(x))_j) $
  )
)
  === Softmax Derivative
  $
    sigma(z)_i = e^(z_i) / sum_(j = 1)^C e^(z_j)
  $
  subsititute, $S = sum_(j=1)^C e^(z_j)$ 
  $
    sigma(z)_i = e^(z_i) / S
  $
  $
  because ddx(,)(u / v) &= (v u' - u v' ) / v^2\ 
  $
  Find $u'$ and $v'$ for $sigma(z)_j$ where $u &= e^(z_i), v &= S$: 
  \
  $
  u' &= dd(e^(z_i), z_j) = cases( e^(z_i) "if" i = j, 0 "  if" i != j)\ 
  v' &= dd(S, z_j) = dd(, z_j) (sum_(j = 1)^C e^(z_j)) = e^(z_j)
  $
  Differentiate  ( _note_: $Delta_(i j) = cases(1 "if" i = j, 0 "if" i != j) space$ ): 
  $
  therefore dd(sigma_i, z_j) &= (S dot [e^(z_i) dot Delta_(i j)] - e^(z_i) dot e^(z_j)) / S^2\
  &= e_(z_i) dot (S dot Delta_(i j) - e^(z_j)) / S^2\ 
  &= e^(z_i) / S dot (S dot Delta_(i j) - e^(z_j)) / S\ 
  &= sigma(z)_i dot (S dot Delta_(i j) - e^(z_j)) / S\
  &= sigma(z)_i dot (Delta_(i j) - (e^(z_j)) / S)\
  &= sigma(z)_i dot (Delta_(i j) - sigma(z)_j)\
  $

=== _Tanh_ Derivative
$
  tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))
$
Let $u = e^x - e^(-x)$ and $v = e^x + e^(-x)$, find $u'$ and $v'$: 
$
  u' &= e^x + e^(-x)\
  v' &= e^x - e^(-x)
$
Differentiate:  
$
ddx(,) tanh(x) &= ((e^x + e^(-x)) (e^x + e^(-x)) - (e^x - e^(-x))(e^x - e^(-x)))/(e^x + e^(-x))^2\

&= ((e^x + e^(-x))^2 - (e^x - e^(-x))^2)/(e^x + e^(-x))^2\
&= 1 - ((e^x - e^(-x))^2)/(e^x + e^(-x))^2\
&= 1 - tanh^2(x)
$

Equivalently: 
$
ddx(,)  tanh(x) &= ddx(,)(sinh x / cosh x)\
&= (cosh x cosh x - sinh x sinh x) / (cosh^2 x)\
&= (cosh^2 x - sinh^2 x) / (cosh^2 x)\
&= 1 / (cosh^2 x) = sech^2 x = 1 - tanh^2(x)
$

=== Sigmoid Derivative 
$
  sigma(x) = 1/ (1 + e^(-x))
$
$
  because dd(,u) (1/u) = (-1/u^2)
$
Let $u = 1 + e^(-x)$: 
$
  dd(sigma, x) &= dd(sigma, u) dot dd(u, x) \
  &= dd(, u) (1/ u) dot dd(, x) (1 + e^(-x))\
  &= -1/u^2 dot - e^(-x)\
  &= e^(-x) / u^2 = e^(-x) / (1 + e^(-x))^2\
  &= 1 / (1 + e^(-x)) dot e^(-x) / (1 + e^(-x))\
  &= 1 / (1 + e^(-x)) dot ((1 + e^(-x)) - 1) / (1 + e^(-x))\
  &= 1 / (1 + e^(-x)) dot ((1 + e^(-x)) / (1 + e^(-x)) - 1 / (1 + e^(-x))) \
  &= 1 / (1 + e^(-x)) dot (1 - 1 / (1 + e^(-x))) \
  &= sigma(x) dot (1 - sigma(x))
$
#place(right, image("yann-lecun.jpg", width: 30%))
]






