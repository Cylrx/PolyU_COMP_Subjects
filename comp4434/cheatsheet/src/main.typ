#set page(
  paper: "a4",
  margin: (
    top: 0in,
    bottom: 0in,
    left: 0in,
    right: 0in,
  )
)
#set par(justify: true)
#set text(
  size: 6.0pt,
)

#let sps = {v(2pt)}
#let dd(y, x) = {$(diff #y) / (diff #x)$}
#let ddx(y) = {$(diff #y)/(diff x)$}
#let big(x) = {$upright(bold(#x))$}
#let code(c) = {highlight(raw(c), fill: silver)}
#let head(h) = {text(fill: maroon, weight: "black", style: "oblique")[#h]}

#columns(3, gutter: 6pt)[
  = #head("Regression Metrics")
  - *MSE*: Mean Square Error $1/n sum_(i=1)^n (hat(y)_t - y_t)^2$
  - *MAE*: Mean Absolute Error $1/n sum_(i=1)^n |hat(y)_t - y_t|$
  - *MAPE*: Mean Absolute Percentage Error $(100%)/n sum_(i=1)^n |(hat(y)_t - y_t)/y_t|$
  = #head("Classification Metrics")
  *Accuracy* 
  #place(right, image("accuracy-table.png", width: 23%), dy: -20pt)
  _Limitation_: Imbalanced data
  $ "Acc" = ("TP" + "TN") / ("TP" + "TN" + "FP" + "FN") $,
  *Precision and Recall*
  #place(left, image("precision-table.png", width: 23%), dy: -5pt)
  #place(right, image("recall-table.png", width: 23%), dy: -10pt)
  $ "Precision" = "TP" / ("TP" + "FP"), \
  "Recall" = "TP" / ("TP" + "FN") $
  - _Precision_: how many predicted positives actually positive? 
  - _Recall_: how many positives are correctly found? 
  *F1 Score*
  $ "F1" = (2 times "Precision" times "Recall") / ("Precision" + "Recall") $
  - _F1 Score_: harmonic mean of precision and recall. 
  - _Harmonic Mean_: closer to the smaller value

  = #head("Linear Regression")
  *Univariate Normation Equation*
  $
    overline(x) = 1/m sum_(i=1)^m x^((i)), 
    overline(y) = 1/m sum_(i=1)^m y^((i)), 
    overline(x y) = 1/m sum_(i=1)^m x^((i)) y^((i)), 
    overline(x^2) = 1/m sum_(i=1)^m (x^((i)))^2
$
  $
    theta_0 "(intercept)" = (overline(y) dot  overline(x^2) - overline(x) dot overline(x y)) / (overline(x^2) - (overline(x))^2), 
    theta_1 "(slope)" = (overline(x y) - overline(x) dot overline(y)) / (overline(x^2) - (overline(x))^2) 
  $
  *Multivariate Normation Equation*
  $
    X = mat(
      1, x_1^((1)), dots, x_j^((1)), dots, x_n^((1));
      1, x_1^((2)), dots, x_j^((2)), dots, x_n^((2));
      dots.v, dots.v, , , dots.down , dots.v;
      1, x_1^((m)), dots, x_j^((m)), dots, x_n^((m))
    ),
    theta = vec(theta_0, theta_1, dots.v, theta_n),
    y = vec(y^((1)), y^((2)), dots.v, y^((m)))
  $
  $
    (X^T X) theta = X^T y\
    theta = (X^T X)^(-1) X^T y
  $

  = #head("Logistic Regression")
  *Basic Definition*:  $f_theta(x) = sigma(theta^T x) = 1/(1 + exp(-theta^T x)) in (0, 1)$\
  *Decision Bound*: $theta^T x = 0$,  predicts $y=1$ when $theta^T x > 0$, vice versa\
  *Logistic Loss*: $L(y, f_theta) = -(y log f_theta (x) + (1-y) log (1-f_theta (x)))$
  #grid(columns: 3, image("neg-log.png", width: 30%), image("neg-log-reverse.png", width: 36%), 
  [
  #v(2em)
  $
    L(y, f_theta) = cases(
      -log f_theta (x) &"if" y = 1,
      -log (1-f_theta (x)) &"if" y = 0
    )
  $],
  gutter: -9em
  )
  *Multi-class Classification*\
  _One-vs-One_ (OvA): Train $C_n^2$ classifers, use majority voting\
  _One-vs-Rest_ (OvR): Train $C$ classifers; each treat all other $C-1$ classes as negative. Choose class with largest probability as output.
  #image("multiclass-classification.png")

  = #head("Support Vector Machine (SVM)")
  Distance from point to a line $big(w)^T big(x) + b = 0$: 
  $ "Distance" = (big(w)^T big(x) + b) /  (||big(w)||) $
  Let training set ${(big(x)^((i)), y_i)}_(i=1dots n), y_i in {-1, 1}$ be separated by a hyperplane with margin $gamma$. Then for each training sample $(big(x)^((i)), y_i)$: 
  $
    y_i (big(w)^T big(x)^((i)) + b) >= gamma slash 2 =>
    cases(
      big(w)^T big(x)^((i)) + b <= -gamma slash 2 quad "if" y_i = -1,
      big(w)^T big(x)^((i)) + b >= gamma slash 2 quad "if" y_i = 1
    )
  $
  If we rescale $big(w)$ and $b$ by $gamma slash 2$ in the above equality, we obtain distance between each *support vector* sample $big(x)^((s))$ with the hyperplane: 
  $
    (y_s (big(w)^T big(x)^((s)) + b)) / (||big(w)||) = 1 / (||big(w)||), 
    quad
    "normalized margin" gamma' = 2 / (||big(w)||)
  $
  *Optimization Problem*\
  1. *Maximize Formulation*: Find $big(w)$ and $b$ s.t. $gamma' = 2 / (||big(w)||)$ is maximized, while for all samples $(big(x)^((i)), y_i)$ satisfies : $y_i (big(w)^T big(x)^((i)) + b) >= 1$
  2. *Minimize Formulation*: Find $big(w)$ and $b$ s.t. $||big(w)||^2=big(w)^T big(w)$ is minimized, while for all samples $y_i (big(w)^T big(x)^((i)) + b) >= 1$

  *Soft Margin SVM*\
  Allow some samples to be misclassified, and instead minimize: 
  $ 
    1/2||big(w)||^2 + C sum_(i=1)^n xi_i 
    quad"subject to"
    cases(
      big(w)^T big(x)^((i)) + b <= -1 + xi_i & "if" y_i = -1,
      big(w)^T big(x)^((i)) + b >= 1 - xi_i & "if" y_i = 1,
      xi_i >= 0 & "for all" i 
    )
  $
  $xi_i$ is the slack variable, and $C$ is the regularization parameter.\
  - $xi_i$ is *not* hyperparameter. It is calculated for each training sample. 
    - $xi_i = 0.3$: point "penetrated" $30%$ way through marign boundary
    - $xi_i = 1.0$: point sitting exactly on hyperplane (decision boundary)
    - $xi_i = 1.5$: point crossed hyperplane, and 50% way on the other side
  - $C$ is hyperparameter. controls the trade-off

  = #head("Neural Network")

  == Perceptrons
  Step Activation: $y = cases(
    1 "if" z > 0,
    0 "otherwise"
  )
  $\
  #grid(columns: 2, gutter: 1em,
  $
  w <- w -alpha dd(L, w_i)\
  dd(L, big(w)) = dd(L, y) dot dd(y, z) dot dd(z, big(w))\ 
  $,
  $
  &because dd(z, big(w)) = big(x), \
  &therefore dd(L, big(w)) = (dd(L, y) dot dd(y, z)) dot big(x) = delta dot big(x)
  $)
  Where: $delta$ is the error term, $L$ (Loss): Loss function, $alpha$: Learning rate, $dd(L, y)$: sensitivity of loss w.r.t. the activation output, $dd(y, z)$: derivative of the activation function
  - $dd(z, big(w))$: since $z = big(w)^T big(x) + b$, the derivative w.r.t. to $w$ is input vector $big(x)$

  #text(maroon)[*Linear Activation: * _(w/ MSE Loss)_]
  $
  hat(y) &= z = big(w)^T big(x) + b;space L(y, hat(y)) = (y - hat(y))^2\
  delta &= dd(L, hat(y)) dot dd(hat(y), z) = dd(L, hat(y)) dot 1 = 2(y -hat(y)) 
  $

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
  Weight Update _(everything together)_: 
  $
  big(w) <- big(w) - alpha(hat(y_j) - y_j) dot big(x)
  $
  #text(maroon)[*Regularized Gradient Descent*]
  #columns(2, gutter: 1em)[
  *L2 Regularization*
  $
    J(theta) &= 1/m [sum_(i=1)^m L(y_i, hat(y_i)) + lambda sum_(j=1)^n theta_j^2]\
    &= 1/m sum_(i=1)^m L(y_i, hat(y_i)) + lambda/m sum_(j=1)^n theta_j^2\
    therefore dd(J, theta_j) &= 1/m [ sum_(i=1)^m dd(L, theta_j) + 2 lambda theta_j ]
  $
  #colbreak()
  *L1 Regularization*
    $
    J&= 1/m [sum_(i=1)^m L(y_i, hat(y_i)) + lambda sum_(j=1)^n abs(theta_j)]\
    &= 1/m sum_(i=1)^m L(y_i, hat(y_i)) + lambda/m sum_(j=1)^n abs(theta_j)\
    therefore dd(J, theta_j) &= 1/m [ sum_(i=1)^m dd(L, theta_j) + lambda "sign"(theta_j) ]
  $
  ]
  where $"sign"(x)$ is the sign function:
  $"sign"(x) = cases(
    1 & "if " x > 0,
    -1 & "if " x < 0,
    0 & "if " x = 0
  )$
  = #head("CNN (Convolution Neural Network)")
  === Activation
  ReLU (Rectified Linear Unit): $"ReLU"(x) = max(0, x)$
  Motivations: 
  - Outputs $0$ for negative values, introducing #text(red)[sparse representation]
  - Does not face vanishing gradient as with sigmoid or tanh
  - Fast: does not need $e^x$ computation
  
  === Conv Layer
  $
  n_"out" &= floor((n_"in" + 2p - k)/s) + 1 &= ceil((n_"in" + 2p - k + 1) / s)
  $
  where,
  - $n_"in"$: number of input features, $n_"out"$: number of output features, $k$: convolution kernel size, $p$: padding size (on one side), $s$: convolution stride size

  == Training w/ Max-pooling
  $y_11 &= max(x_11, x_12, x_21, x_22)$
  === Derivative
  For values of input feature map $x_i in {x_1, x_2, dots, x_n}$,\
  where $x_i$ is amongst the inputs of $y_j in {y_1, y_2, dots, y_m}$

  #text(red)[
    In simpler terms: $x$ is input before max-pooling, $y$ is output after max-pooling, $m < n$ for obvious reasons, $x_i$ can be inputs of multiple $y$
  ]

  #grid(columns: 2, gutter: 3em,
  [
  Then we have,
  $
  dd(L, x_i) = sum_(y_j space in "all max-pooling 
  windows convering" x_i) dd(L, y_j) dot dd(y_j, x_i)
  $
  ],
  [
  The #text(red)[core idea] is that,
  $
  dd(L, x_i) = cases(
    dd(L, y) dot dd(y, x_i)  "if" x_i = y,
    0 "           otherwise"
  )
  $
  ])

  = #head("RNN (Recurrent Neural Network)")
  *Vanilla RNN:* $h_t = f_theta (h_(t-1), x_t)$\
  where $h_t$ is new state, $h_(t-1)$ is previous state, $x_t$ is input
    - *Pros*: Process variable-length sequence
    - *Cons*: vanishing or exploding gradient.
  Let's assume that the weight $big(w)$ is scalr, so is $h_t$. Loss at time $T$ is $L_T$
  $
    dd(L_T, h_k) = dd(L_T, H_T) dd(H_T, h_(T-1)) dd(h_(T-1), h_(T-2)) dots dd(h_(k+1), h_k) = dd(L_T, H_T) (w)^(T-k)
  $
  - Then any weight value $w != 1$ will cause unstable gradient.

  #v(1em)
  *LSTM (Long Short-Term Memory):*
  - `new_state = forget(old_state) + select(new_state)`
  - `output = select(new_state)`
  #grid(columns: 2, gutter: 0em,
  [
    Input Processing: 
    $
      f_t &= sigma(W_f [h_(t-1), x_t] + b_f) in (0, 1)\
      i_t &= sigma(W_i [h_(t-1), x_t] + b_i) in (0, 1)\
      tilde(C)_t &= tanh(W_C [h_(t-1), x_t] + b_C)\
      C_t &= f_t * C_(t-1) + i_t * tilde(C)_t
    $
    Output: 
    $
      o_t &= sigma(W_o [h_(t-1), x_t] + b_o) in (0, 1)\
      h_t &= o_t * tanh(C_t)
    $
    - $o_t$ select part of $C_t$ to output based on $h_(t-1)$ and $x_t$
    - $h_t$ is the actual output
  ],
  [
    #image("lstm-cell.png")
    Intuition: 
    - $C$ is long-term memory.
    - $f_t$ forgets some old mem in $C_(t-1)$
    - $i_t$ select some new mem in $tilde(C)_t$
    - merge 'forgotten' and 'selected' mem to form new $C_t$
  ]
  )

  #v(1em)
  *GRU (Gated Recurrent Unit):*
  #grid(columns: 2, gutter: 1em,
  [
    $
      z_t &= sigma(W_z [h_(t-1), x_t]) in (0, 1)\
      r_t &= sigma(W_r [h_(t-1), x_t]) in (0, 1)\
      tilde(h)_t &= tanh(W_h [r_t * h_(t-1), x_t])\
      h_t &= (1-z_t) * h_(t-1) + z_t * tilde(h)_t
    $
  ],
  [
    - $r_t$ forgets some old mem from $h_(t-1)$
    - $tilde(h)_t$ candidate state made from 'forgotten' old mem $r_t * h_(t-1)$ and $x_t$
    - $z_t$ creates coefficient for mixing candidate $tilde(h)_t$ and old mem $h_(t-1)$
  ]
  )
  *Sequence Learning w/ One RNN Layer*
  #image("sequence-learning.png")
  + Standard NN - image classification
  + Sequence output - image captioning
  + Sequence input - Sentiment analysis
  + Sequence input output - machine translation
  + Sync seq input and output - video classification, label each frame

  = #head("MapReduce")
  - _Problem_: Process massive data beyond single machine capacity.
  - _Solution_: Parallel processing on commodity clusters.
  - _MapReduce_: Programming model (like Von-Neumann)
  - _User_: Defines `Map` & `Reduce` functions.
  - _System_: Handles parallelism, data distribution, fault tolerance, communication.

  *Workflow*: `Input` $->$ `Map` $->$ `Shuffle & Sort` $->$ `Reduce` $->$ `Output`
  + *`Input`*: split input data into $M$ splits, 
    - each split processed by 1 machine
    - each split contains $N$ records as key-value pairs
  + *`Map`* transforms input k-v to new set of k'-v' pairs
  + *`Shuffle & Sort`* the k'-v' pairs
  + All k'-v' with the same k' grouped together, sent to *same reduce*
  + *`Reduce`*  processes all k'-v' into new k''-v'' pairs
  + *`Output`*: write resulting pairs to file

  *Example*: Set difference $A - B$ (i.e., what $A$ has but $B$ doesn't)
  #grid(columns: 2, gutter: 1em,
  [
    ```python
    Map(key, value) {
      # key: split A or B
      # value: elements
      for e in value: 
        emit(e, key);
    }
    ```
  ],
  [
    ```python
    Reduce(key, values) {
      # key: element
      # values: list of names of splits
      if 'A' in values: 
        if 'B' not in values: 
          emit(key);
    }
    ```
  ]
  )

  *Dealing w/ Failures*
  - *`Map`* failures: 
    - Completed & in-progress tasks on failed worker are reset to idle & rescheduled.
    - Completed `Map` outputs (on local disk of failed worker) are lost; *re-executed on another worker*
  - *`Reduce`* failures: 
    - In-progress tasks reset & restarted.
    - Completed `Reduce` output to global FS, so *NOT re-executed*

  = #head("Recommender Systems")
  == Content Based (CB)
  _Assumption_: if past user liked a set of items with certain features, they will continue to like other items with similar features.
  - *Pros*: 
    - No need for data on other users
    - Able to recommend to unique user taste
    - Able to recommend new & unpopular items 
    - Interpretability: provide explanation
  - *Cons*: 
    - Finding appropriate features is hard
    - Building user profile is slow (e.g., for new users)
    - Overspecialization (don't show diverse content to user)
  
  *Approach 1: CB based on Linear Regression*\
  - _Input_: feature vector of item (e.g., movie's style feature vector)
  - _Output_: user preference (e.g., scalar user rating value)
  - _Model_: each user have their own model $theta^((j))$
    - predicted rating $R_(j,i) = theta_j^T x^((i))$ 
      - where $x^((i))$ is feature vector of item $i$

  == Collaborative Filtering (CF)
  _Assumption_: user with similar history likely have similar preference
  - *Pros*: Works on any kind of item (no feature selection needed)
  - *Cons*: 
    - Cold start, need enough user in system to find match
    - Sparsity: user / rating matrix very sparse
    - First rater: cannot recommend item has not been rated
    - Popularity bias: cannot recommend to someone with unique taste
  
  *Approach 1: CF based on Linear Regression*
  - _*Simutaenously*_ learn user preference $theta^((j))$ and item feature $x^((i))$
  - Intuition: factorize $N times M$ user-item rating matrix into $N times F$ user-preference matrix and $F times M$ feature-item matrix

  $
  J(x, theta) = &underbrace((1/(2m_"ratings")) sum_((i,j) "where" r(i,j)=1) ((theta^((j)))^T x^((i)) - y^((i,j)))^2,"Sum of Squared Errors for known ratings") \
  &- underbrace((lambda/(2m_"ratings")) sum_(i=1)^(N_m) sum_(k=1)^n (x_k^((i)))^2,"Regularization for Movie Features") \
  &- underbrace((lambda/(2m_"ratings")) sum_(j=1)^(N_u) sum_(k=1)^n (theta_k^((j)))^2,"Regularization for User Preferences")
  $
  - $r(i, j) = 1$ if user $i$ has rated item $j$, $0$ otherwise
  - $theta^((j)) in RR^F$ latent preference vector of user $j$ *being optimized*
  - $x^((i)) in RR^F$ latent feature vector of item $i$ *being optimized*
  - $y^((i,j)) in RR$ is the scalar rating of user $i$ for item $j$
  - $m_"ratings"$ is the number of known ratings
  - $N_m$ is the number of movies, $N_u$ is the number of users
  - $n$ is the number of features
  Gradient Update for CF Objective: 
  $
    dd(J(x^((i)), theta^((j))), x_k^((i))) &= 1/m (sum_(j:r(i,j)=1) ((theta^((j)))^T x^((i)) - y^((i,j))) theta_k^((j)) + lambda x_k^((i)))\

    dd(J(x^((i)), theta^((j))), theta_k^((j))) &= 1/m (sum_(i:r(i,j)=1) ((theta^((j)))^T x^((i)) - y^((i,j))) x_k^((i)) + lambda theta_k^((j)))
  $
  $
    x_k^((i)) = x_k^((i)) - alpha dot (dd(J(x^((i)), theta^((j))), x_k^((i)))), 
    theta_k^((j)) = theta_k^((j)) - alpha dot (dd(J(x^((i)), theta^((j))), theta_k^((j))))
  $

  *Approach 2: Finding 'Similar' Users* 
  - _Jaccard Similarity_: $"sim"(x, y) = |x sect y| slash | x union y|$ 
    - counting number of *items rated by both users*
    - _Issue_: ignores the value of the rating
  - _Cosine Similarity_: $ "sim"(x, y) = (sum_(S_(x y)) r_(x i) dot r_(y i)) / (sqrt(sum_(S_(x)) r_(x i)^2) dot sqrt(sum_(S_(y)) r_(y i)^2)) $
    - _Issue_: implicitly zero-fills missing ratings. 

  - _*Pearson Correlation Coefficient*_\
    #text(fill: red)[_Intuition_: the formula is equivalent to normalized cosine similarity. Each term mean-centered by subtracting average. Therefore, missing ratings (implicitly 0) is $r_(x s) - overline(r_x) = 0$. Not bad!]

    *Therefore, EXAM TIP*: make matrix mean-centered by subtracting each $r_(x s)$ by $overline(r_x)$ first. Then do it like normal cosine similarity.
  $
    "sim"(x, y) = (sum_(s in S_(x y)) (r_(x s) - overline(r_x)) (r_(y s) - overline(r_y)))
    
    / (sqrt(sum_(s in S_(x)) (r_(x s) - overline(r_x))^2) sqrt(sum_(s in S_(y)) (r_(y s) - overline(r_y))^2))
  $
  - $S_(x y)$ = set of items rated by both users $x$ and $y$
  - $S_x$, $S_y$ = set of items rated by $x$, $y$ respectively
  - $overline(r_x)$, $overline(r_y)$ = avg. rating of $x$, $y$ respectively
  - $r_(x s)$ = rating of $x$ for item $s$

  After computing a similarity matrix with the above
  - Let $r_x$ be the vector of user $x$'s ratings
  *User-User CF*: Predicted item $i$ rating by user $x$ from similar users:
  - Let $K_(s i)$ be set of $k$ users most similar to $x$ who have rated item $i$
  $
    hat(r)_(x i) = 1/k sum_(j in K_(s i)) r_(j i ), 
    quad "OR" quad
    r_(x i) = (sum_(j in K_(s i)) "sim"(x, j) dot r_(j i)) / (sum_(j in K_(s i)) "sim"(x, j))
  $
  *Item-Item CF*: Predicted item $i$ rating by user $x$ from similar items:
  - Let $K_(s i)$ be set of $k$ items most similar to $i$ that user $x$ has rated
  $
    hat(r)_(x i) = 1/k sum_(j in K_(s i)) r_(x j ), 
    quad "OR" quad
    r_(x i) = (sum_(j in K_(s i)) "sim"(i, j) dot r_(x j)) / (sum_(j in K_(s i)) "sim"(i, j))
  $
  #text(fill: red)[*item-item generally works better*, since items are simpler, while user have complex and nuanced tastes]

  *Common Practice*:\
  For weighted average methods, add baseline to correct for user biases
  $
    r_(x i) = b_(x i) + (sum_(j in K_(s i)) "sim"(i, j) dot (r_(x j) - b_(x j))) / (sum_(j in K_(s i)) "sim"(i, j))
  $
  - $b_(x i) = mu + b_x + b_i$: baseline rating for user $x$ and item $i$
  - $b_x = ("avg. rating of" x) - mu$ rating deviation of user $x$
  - $b_i = ("avg. rating of" i) - mu$ rating deviation of item $i$
  - $mu$ = global mean movie rating

 = #head("PageRank")

 _*Intuition*_: each score is like a *web surfer*
  - *Locally*: each surfer randomly (evenly) go to one neighbor.
  - *Globally*: after sufficient iterations, popular websites will accumulate more surfer than smaller ones (e.g., Google have more visitors than smaller websites at any given time)

  *Term Definitions*: 
  - Let page $1 <= i <= N$ has $d_i$ out-links. \
  - Then define _*Column Stochastic Matrix*_ $M in RR^(N times N)$: 
    - $M_(i,j)$ = probability that a surfer at page $i$ will next go to page $j$
    $
      M_(i,j) = cases(
        1/d_i quad & "if exist path" i -> j, 
        0 & "otherwise"
      )
    $
  - as well as _*Rank Vector*_ $r in RR^N$: 
      - Each $r_i$ is the current importance score of page $i$. 
      - Scores sum to 1: $sum_i r_i = 1$
  
  *Flow Equation*: $r^((t+1)) = M dot r^((t))$
  $
    therefore r^((t+1)) = M r^((t)) = M(M r^((t-1))) = M^t r^((0))
  $
  _Intuition_: 
  - assume $r^((t))$ converge and stop changing, then $M r^((t)) = 1 dot r^((t))$. 
  - Using the eigenvector definition $M x = lambda x$, $r^((t))$  is the first (or principal) eigenvector of $M$, with eigenvalue $lambda = 1$

  _Proof_ 
  + Assume $M$ has $n$ linear independent eigenvectors $x_1, x_2, dots x_n$ with eigenvalues $lambda_1, lambda_2, dots lambda_n$, where $lambda_1 > lambda_2 > dots > lambda_n$
  + Vectors $x_1, x_2, dots x_n$ form a basis, thus we can write 
    - $r^((0)) = c_1 x_1 + c_2 x_2 + dots + c_n x_n$
  + Then: 
  $
    M r^((0)) &= M(c_1 x_1 + c_2 x_2 + dots + c_n x_n)\
    &= c_1 (M x_1) + c_2 (M x_2) + dots + c_n (M x_n)\
    &= c_1 (lambda_1 x_1) + c_2 (lambda_2 x_2) + dots + c_n (lambda_n x_n)\
    M^k r^((0)) &= c_1 (lambda_1^k x_1) + c_2 (lambda_2^k x_2) + dots + c_n (lambda_n^k x_n)\
    &= lambda_1^k [c_1 x_1 + c_2 (lambda_2 / lambda_1)^k x_2 + dots + c_n (lambda_n / lambda_1)^k x_n]
  $
  + Since $lambda_1 > lambda_2$, then $lambda_2/ lambda_1, lambda_3 / lambda_1, dots < 1$
  + Therefore, $(lambda_i / lambda_1)^k = 0$ as $k -> oo$ for all $i >= 2$
  + Thus $M^k r^((0)) -> c_1 (lambda_1^k x_1)$, *the largest eigenvalue always 1*

  #place(right, image("pagerank-problems.png", width: 25%))
  *Power Iteration Method*
  + Suppose there are $N$ web pages
  + Uniformaly initialize $r^((0)) = [1/N, 1/N, dots, 1/N]^T$
  + Iterate $r^((t+1)) = M dot r^((t))$ 
  + Stop when $|r^((t+1)) - r^((t))|_1 < epsilon$ 
    - $|x|_1$ is L1-norm. But other norms work too.
  
  *Problems*: 
  - _Dead Ends_: scores leak out of network
  - _Spider Traps_: scores stuck indefinitely in small set of pages. Eventually the _Spider Trap_ will absorb all importance, draining other pages.

  *Solutions*: Random Teleports
  - _*What?*_
    - With probability $beta$, follow a link at random
    - With probability $(1-beta)$, jump to some random page
    - In practice, $beta = 0.8 ~ 0.9$ (make 5 steps on avg. before jump)
    $
      r_j = sum_(i -> j) beta r_i / d_i + (1 - beta) 1/N
    $
    - Alternatively, reformulate as *Power Iteration Method*:
    $
      A = beta M + (1 - beta) [1/N]_(N times N)
    $
    - Using the new matrix $A$, we get the familiar form: $r^((t+1)) = A r^((t))$

  - _*Why?*_
    - _Spider Traps_: Teleport out of spider traps in finite steps
    - _Dead Ends_: teleport to a random page when nowhere to go
  
  - _*How?*_ MapReduce program for PageRank: 
      #grid(columns: 2, gutter: 1em,
      ```python
      Map(key, value) {
        # key: a current page id
        # value: page rank of the current page
        for adj_page in Adj[key]: 
          emit(adj_page, value / count(Adj[key]))
      }
      ```,
      ```python
      Reduce(key, values) {
        # key: a current page id
        # values: a list of incoming scores from other pages
        scores[key] = 1 - beta
        for score in values: 
          scores[key] += beta * score
        emit(key, scores[key]);
      }
      ```)
 
  = #head("Clustering")
  *Distance Metrics (for Vectors)*
  - Euclidean Distance: $d(big(A), big(B)) = sqrt(sum_(i=1)^n (A_i - B_i)^2)$
  - Cosine Similarity: $ cos(theta) = (big(A) dot big(B)) / (||big(A)|| ||big(B)||) = (sum_(i=1)^n A_i B_i) / (sqrt(sum_(i=1)^n A_i^2) sqrt(sum_(i=1)^n B_i^2)) $
  - Cosine Distance = $1 - cos(theta)$

  *Distance Metrics (for Sets)*
  - Jaccard Similarity: $J(S_A, S_B) = |S_A sect S_B| slash  |S_A union S_B|$
  - Jaccard Distance = $1 - J(S_A, S_B)$

  *K-Means Clustering*
  + Initialize: randomly create $k$ cluster centroid points
  + Assign: assign each point to the nearest cluster centroid
  + Update each cluster centroid to the mean of all itis points
  + Repeat 2-3 until convergence
  - *pro*: simple, user provide $k$
  - *cons*: sphere, hard to guess $k$
  - *complexity*: $O(n times k times I times d)$
    - $n$: num of points, $k$ num of clusters,\ $I$ num iterations, $d$ num of attributes

  #place(right, image("centroid-clustroid.png", width: 40%), dy: -50pt, dx: -10pt)
  *Concepts*
  - _Centroid_: is the avg. of all points in the cluster (artificial point)
  - _Clustroid_ is *existing point* closest to all other points
  #place(right, image("singular-value-decomposition.png", width: 50%))
    - smallest avg distance to other points? 
    - smallest sum of squares to other points? 
    - smallest max distance to other points?

  === Distance Metrics
  - Continuous Variable: $L_1, L_2, dots, L_p$ norm
  - Binary Variable: $abs(0 - 1) + abs(1 - 0) + abs(1 - 1) dots$
  - Categorical: 1. One-hot encoding, 2. Simple matching: $("total features" - "matched features") slash "total features"$

  = #head("Dimensionality Reduction (DR)")

  == Singular Vector Decomposition (SVD)

  *Easy Method* (w/ Calculator): 
  + Given matrix $A in RR^(m times n)$ to factorize
  + $U = A A^T in RR^(m times m)$. Find its eigenvectors ${arrow(u)_1, arrow(u)_2, dots, arrow(u)_m}$
  + $V = A^T A in RR^(n times n)$. Find its eigenvectors ${arrow(v)_1, arrow(v)_2, dots, arrow(v)_n}$
  + Determine $U$ and $V$'s *common eigenvalues* ${lambda_1, lambda_2, dots, lambda_r}$

  #image("singular-value-decomposition-calculation.jpg")

  *Best Low Rank Approx.*
  - Let $B = U S V^T$, where $S in RR^(k times k), k <= r$
  - Objective: $min_B ||A-B||_F = min_B sqrt(sum_(i j) (A_(i j) - B_(i j))^2)$
  - Intuitively, select *top $k$ best ranks* to approximate $A$


  == Eigenvector & Eigenvalue
  *$M x = lambda x$*, where $M in RR^(n times n)$ (square matrix), $x in RR^n$, $lambda in RR$
  $
    M x &= lambda x => M x - lambda x &= 0
  $
  To factor our $lambda$ introduce *identity matrix* $I in RR^(n times n)$:
  $
    cases(
      M x - lambda I x &= 0,
      (M - lambda I) x &= 0
    ) 
    quad => quad "solve: " 
    det(M - lambda I) &= 0
  $
  For $2 times 2$ matrix: $det mat(a, b; c, d) = a d - b c$\
  Therefore, solve for characteristic equation: $(a - lambda)(d - lambda) - b c = 0$

  == Principal Component Analysis (PCA)
  + Compute covariance matrix $Sigma$
  + Calculate eigenvalues of eigenvectors of $Sigma$
    - Eigenvectors w/ largest eigenvalue $lambda_1$ is $1^("st")$ principal component
    - Eigenvectors with $k^("th")$ largest eigenvalue $lambda_k$ is $k^("th")$ PC
    - Proportion of variance captured by $k^("th")$ PC = $lambda_k slash (sum_i lambda_i)$

  NOTES:
  - PCA can be seen as noise reduction. 
  - Fail when data consists of multiple clusters
  - Direction of max variance may not be most informative

  == Non-linear DR
  - ISOMAP: Isometric Feature Mapping
  - t-SNE: t-Sochastic Neighbor Embedding
  - Autoencoders: map $RR^n -> RR^k -> RR^n$, minimize reconstruction loss
  
= #head("Appendix")
#grid(columns: 2, gutter: 0em,
[
== Derivatives

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
],
[
=== Elementary Functions

Hyperbolic Functions
$
dd(sinh(x), x) = cosh(x)
dd(cosh(x), x) = sinh(x)\
dd(tanh(x), x) = sech^2(x)
dd(coth(x), x) = -csch^2(x)\
dd(sech(x), x) = -sech(x) * tanh(x)\
dd(csch(x), x) = -csch(x) * coth(x)\
$
#grid(columns: 2, gutter: 3em, 
[
Exponential Functions
$
dd(e^x, x) = e^x\
dd(a^x, x) = a^x dot ln(a)
$
],
[
Logarithmic Functions
$
dd(ln(x), x) = 1 / x\
dd(log_a(x), x) = 1 / (x dot ln(a))
$
]
)
])

== Derivatives of #head("Loss Function") ($dd(L, hat(y))$)
- No need to consider their batched version. For example, only need to consider $(y - hat(y))^2$ for MSE. No need for $1/n sum_(i=1)^n (y - hat(y))^2$ 
- Because $n$ samples $->$ $n$ different $hat(y)$ $->$ $n$ different gradients for each weight down the chain rule.
- For any $L$ in the form of $1/n sum f(hat(y_i), y_i)$, the derivative yield by each sample is $1/n f'(hat(y_i), y_i)$. Try 
- So as the gradients accumulate for each weight $w$, they are automatically batch averaged ($1 slash n)$.
  === Softmax Derivative
  $sigma(z)_i = e^(z_i) / sum_(j = 1)^C e^(z_j)$
  subsititute, $S = sum_(j=1)^C e^(z_j)$ 
  then, $sigma(z)_i = e^(z_i) / S$
  $because ddx(,)(u / v) &= (v u' - u v' ) / v^2\ $
  Find $u'$ and $v'$ for $sigma(z)_j$ where $u &= e^(z_i), v &= S$: \
  $u' &= dd(e^(z_i), z_j) = cases( e^(z_i) "if" i = j, 0 "  if" i != j), quad 
  v' &= dd(S, z_j) = dd(, z_j) (sum_(j = 1)^C e^(z_j)) = e^(z_j)$\
  Differentiate  ( _note_: $Delta_(i j) = cases(1 "if" i = j, 0 "if" i != j) space$ ): \
  $therefore dd(sigma_i, z_j) = (S dot [e^(z_i) dot Delta_(i j)] - e^(z_i) dot e^(z_j)) / S^2 = e_(z_i) dot (S dot Delta_(i j) - e^(z_j)) / S^2 = e^(z_i) / S dot (S dot Delta_(i j) - e^(z_j)) / S = sigma(z)_i dot (S dot Delta_(i j) - e^(z_j)) / S = sigma(z)_i dot (Delta_(i j) - (e^(z_j)) / S) = sigma(z)_i dot (Delta_(i j) - sigma(z)_j)\
  $

=== _Tanh_ Derivative
$tanh(x) = (e^x - e^(-x)) / (e^x + e^(-x))$
Let $u = e^x - e^(-x)$ and $v = e^x + e^(-x)$, find $u'$ and $v'$: 
$u' = e^x + e^(-x) quad v' = e^x - e^(-x)$\
Differentiate: $ddx(,) tanh(x) = ((e^x + e^(-x)) (e^x + e^(-x)) - (e^x - e^(-x))(e^x - e^(-x)))/(e^x + e^(-x))^2= ((e^x + e^(-x))^2 - (e^x - e^(-x))^2)/(e^x + e^(-x))^2\
= 1 - ((e^x - e^(-x))^2)/(e^x + e^(-x))^2= 1 - tanh^2(x)
$\ *OR*: 
$ddx(,) tanh(x) = ddx(,)(sinh x / cosh x)= (cosh x cosh x - sinh x sinh x) / (cosh^2 x)= (cosh^2 x - sinh^2 x) / (cosh^2 x)= 1 / (cosh^2 x) = sech^2 x = 1 - tanh^2(x)$

=== Sigmoid Derivative 
$sigma(x) = 1/ (1 + e^(-x))$
, $because dd(,u) (1/u) = (-1/u^2)$
Let $u = 1 + e^(-x)$: 
#place(bottom, float: true,
[$
  dd(sigma, x) = dd(sigma, u) dot dd(u, x)= dd(, u) (1/ u) dot dd(, x) (1 + e^(-x))\
  = -1/u^2 dot - e^(-x) = e^(-x) / u^2 = e^(-x) / (1 + e^(-x))^2 = 1 / (1 + e^(-x)) dot e^(-x) / (1 + e^(-x))\ = 1 / (1 + e^(-x)) dot ((1 + e^(-x)) - 1) / (1 + e^(-x)) = 1 / (1 + e^(-x)) dot ((1 + e^(-x)) / (1 + e^(-x)) - 1 / (1 + e^(-x)))\ = 1 / (1 + e^(-x)) dot (1 - 1 / (1 + e^(-x))) = sigma(x) dot (1 - sigma(x))
$], dx: -10pt, dy: -10pt

)

]




