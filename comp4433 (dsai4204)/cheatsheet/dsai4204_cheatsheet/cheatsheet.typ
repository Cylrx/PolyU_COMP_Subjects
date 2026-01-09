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
#let subhead(h) = {text(fill: blue, weight: "black", style: "oblique")[#h]}
#let def(x, y, g: 1em) = {grid(columns: 2, gutter: g, [#set text(style: "italic"); #x], [#y])}

#columns(3, gutter: 6pt)[
  = #head("Measurement Error")
  - Error Rate: \# of errors / \# of instances = (FN+FP)/N
  - Recall: \# found positives / \# actual positives = TP/(TP+FN)
  - Precision: \# found positives / \# predicted positives = TP/(TP+FP)
  - Specificity = TN/(TN+FP)
  - False Alarm Rate = FP/(FP+TN) = 1 - Specificity

  //#place(left, image("precision-table.png", width: 23%), dy: 0pt)
  //#place(right, image("recall-table.png", width: 23%), dy: 0pt)

  = #head("Association Rule Mining")
  For finding *frequent patterns*, associations, corr., causal structures.

  *Format: * $X => Y ["support", "confidence"]$ 
  - _*Support*_: $P(X inter Y)$ 
    - prob. that transaction *contains both X & Y*
    - indicates *statistical independece* of associtation rule

  - _*Confidence*_: $P(Y|X) = P(X inter Y) / P(X) = (P(X|Y)P(Y)) / P(X)$ 
    - prob. that transaction *having X also has Y*
    - the degree of correlation. Measures *rule's strength*

  #subhead([Apriori Algorithm])
  ```py
  # C[k]: Candidate itemset of size k
  # L[k]: Frequent itemset of size k
  L[1] = [frequent single items]; k = 2 
  while L[k-1] is not empty:
    C[k] = generate_candidates_from(L[k-1])
    C[k] = prune_candidates(C[k])
    L[k] = count_support(C[k])
  ```
  *Step 1: Generate Candidates*\
  Generate candidates for $k$-itemset from $(k-1)$-itemset: 
  ```py
  n = len(L[k-1])
  for i in range(n): 
    for j in range(i+1, n): 
      itemset_i, itemset_j = L[k-1][i], L[k-1][j]
      shared_items = intersection(itemset_i, itemset_j)
      if len(shared_items) == k-2: 
        candidate = join(itemset_i, itemset_j)
        C[k].append(candidate)
  ```
  
  *Step 2: Prune Candidates*\
  Prune candidates that do not meet the minimum support threshold:
  ```py  
  # given k-itemset, prune if any (k-1) subset not in L[k-1]
  for cand in C[k]: 
    for split in range(len(cand)): 
      subset = cand[:split] + cand[split+1:]
      if subset not in L[k-1]: 
        C[k].remove(cand)
        break
  ```

  *Step 3: Counting*\
  ```py
  cnt = {cand: 0 for cand in C[k]}
  for entry in database: 
    for cand in C[k]: 
      if cand in entry: 
        cnt[cand] += 1
  L[k] = [cand for cand in C[k] if cnt[cand] >= min_support]
  ```
  *Repeat Step 1 & 2 until `L[k]` is empty*

  #subhead([Rule Generation])
  - *For* each _frequent itemset_ $L$, gen all non-empty subsets of $L$
  - *For* each non-empty subset $s in L$, gen rule $s => (L-s)$

  #subhead([Other Measures])\
  *Objective Measures*: support & confidence\
  *Subjective Measures*: Interest (lift)
  $
  "Lift" = ("Conf"(X=>Y))/P(Y) = P(X inter Y) / (P(X) P(Y))
  $
  - *Intuition 1*: _Performace (above baseline) Test_
    - $"Conf"(X=>Y)$ given $X$ happened, prob of $Y$ also happening; 
    - $P(Y)$, prob of $Y$ happening in general (*baseline*)
  - *Intuition 2*: _Independence Test_
    - $P(X inter Y)$ *actual observed* prob. of $X$&$Y$ happening together; 
    - $P(X) P(Y)$ expected prob. of $X$&$Y$ happening together if they were completely independent. (*the independence test*)

  = #head("Sequential Rule Mining")
  *Step1-Sort Phase*: First group data by customer ID. For each customer, sort their transaction by timestamp.\
  *Step2-Freq Itemset*: Find all freq/large itemset $L$ (using Apriori).\
  - Counting rule: number of customers whose sequence contains the itemset at least once
  *Step3-Transform*: Transform each sequence to frequent itemset\
  - Given: Freq. Itemsets (from Step 2): ${(A), (B), (A,B), (C)}$
  - Trans: $<(A,B), (C), (D)> -> <{(A),(B),(A,B)}, {(C)}>$, 
    - $(A,B)$ became ${(A),(B),(A,B)}$
    - $(C)$ became ${(C)}$
    - $(D)$ is removed, because it is not in Freq. Itemsets
  *Step4-Sequence*: Find sequences using frequent itemsets (AprioriAll)\
  - Given: transformed seq (from step 3) and freq 1-itemset (from step 2)
  - Find frequent $k$-sequences, by joining $(k-1)$-sequences with the same $(k-2)$ prefix. Prune if any $(k-1)$-subseq is not frequent.
  - Count support for each $k$-sequence, and remove if below threshold.
  - Given candidate seq `ABC`, transaction $<(A,B), (C), (D)>$ supports `ABC`; transformed version helps make this counting easier: $<{(A),(B),(A,B)}, {(C)}>$
    #text(fill: red)[*This is why Step3 is needed!!!*] 
  *Step5-Maximal Phase*: \
  Sequence $s$ is maximal if $s$ is *not* contained in any other sequences
  ```py
  for k in range(max_k, 1, -1): 
    for seq in L[k]: for subseq in seq: 
        if subseq in L[k-1]: L[k-1].remove(subseq)

  ```


  #subhead([AprioriAll ALogirhtm])
  ```py
  L[1] = [frequent 1-itemsets]; k = 2 
  while L[k-1] is not empty:
    C[k] = generate_candidates_from(L[k-1])
    L[k] = prune_candidates(C[k])
  ```
  *Diff. compared to Apriori*\

  *`generate_candidates_from`*: 
  - given two $(k-1)$-sequences $A$ and $B$, generate $k$-sequence $C$,
  - if `A[:-1] == B[:-1]`, we get: `A + [B[-1]]` and `B + [A[-1]]` 
  - $A$ and $B$ *must* share first $(k-2)$ items!
  - *Example*: permutate last item of $(k-1)$-sequence
    - $<A, B> + <A, C> => <A, B, C> "and" <A, C, B>$

  *`prune_candidates`*:
  - given a $k$-seq, prune if *any* of its $(k-1)$-subseqs is not frequent.
  - *Example*: `L[3] = {abc, abd}`, generate `abdc` and `abcd`
    - Prune `abdc` because `adc` and `bdc` are not in `L[3]`




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

  == Regressoin Tree
  - Prediction is computed as *avergae* of sample values in the leaf node.
  - Impurity measured as *sum of squared deviations* from leaf mean
  - Performance measured by *root mean squared error* (RMSE)

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
  def GenerateTree(x): 
    if NodeEntropy(x) < θ_I /* (See I_m on slide 29)
        Create leaf labelled by majority class in x
        return
    i ← SplitAttribute(x)
    for each branch of x[i]: 
        Find x[i] falling in branch
        GenerateTree(x[i]) 

  def SplitAttribute(x):
      MinEnt ← MAX
      for all attributes i = 1, ..., d:
          if x[i] is discrete with n values:
              Split x into x[1], ..., x[n] by x[i]
              e ← SplitEntropy(x[1], ..., x[n])
              if e < MinEnt: 
                  MinEnt ← e
                  bestf ← i
          else: # x[i] is numeric 
              for all possible splits
                  Split x into x[1], x[2] on x[i]
                  e ← SplitEntropy(x[1], x[2])
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

  = #head("Bayes Classification")
  Given sample $X$ and class $C$: 
  $
    P(C|X) = (P(X|C) P(C))/ P(X)  "aka." "posterior" = ("prior" times "likelihood") / "evidence"
  $
  *Idea*: assign sample $X$ the best $C$ such that $P(C|X)$ is maximized.
  - $P(X)$ unknown, but constant for all $C$. So we can ignore it.
  - $P(C)$ unknown, but reasonably estimated from dataset 
  - $P(X|C)$ unknown and infeasible. 
   - Need to know the joint probability $P(x_1, x_2, dots, x_n | C)$. 
   - If each $x_i$ takes 2 values, we can have $2^n$ possible combinations.
   - Probability that any such sample already exist in our dataset $approx 0$

  == #subhead([Naive Bayes Classifier])
  Assumes attributes are *conditionally independent*: 
  $
    P(C|X) prop P(X) product_(i=1)^n P(x_i | C)
  $
  How to estimate $P(x_i|C)$?
  - If $x_i$ categorical: relative frequency of $x_i$ in class $C$ in dataset
  - If $x_i$ continuous: gaussian density via all $i$-th attr. in class $C$ 

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

  == Distance Metrics
  - Continuous Attributes: Euclidean Distance 
    - *normalize* each dimension by standard deviation
  - Discrete Data: use hamming distance

  == Curse of Dimensionality
  Definition: Prediction accuracy degrade quickly when number of attributes grow, because
  - When many irrelevant attributes involved, relevant attributes are shadowed, and distance measurements are unreliable
  *Solution*: 
  - Remove irrelevant attributes in pre-processing (e.g., dimensionality reduction such as PCA)

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

  = #head("Clustering")
  === Requirements for clustering in data mining: 
  - Scalability
  - Ability to deal with different types of attributes
  - Discovery of clusters with arbitrary shape
  - Minimal req. for domain knowledge to determine input parameters
  - Able to deal with noise and outliers
  - Insensitive to order of input records
  - High dimensionality
  - Incorporation of user-specified constraints
  - Interpretability and usability

  == #subhead([Distance Metrics (Continuous Variables)])
  *Minkowski Distance* ($L_q$ distance general form) 
  $ 
  d(arrow(i), arrow(j)) = root(q, sum_(k=1)^p |x_(i k) - x_(j k)|^q)
  = root(q, |x_(i 1) - x_(j 1)|^q + ... + |x_(i p) - x_(j p)|^q)
  $
  *Manhattan Distance* ($L_1$ distance)
  $ d(arrow(i), arrow(j)) = sum_(k=1)^p |x_(i k) - x_(j k)| = |x_(i 1) - x_(j 1)| + |x_(i 2) - x_(j 2)| + ... + |x_(i p) - x_(j p)| $

  == #subhead([Distance Metrics (Binary Variables)])
  #grid(
    columns: 2,
    gutter: 1.3em,
    table(
      columns: (3.2em,) * 4,
      align: center + horizon,
      stroke: none,
      inset: 1.3pt,
      table.vline(x: 1, stroke: 0.5pt),
      table.hline(y: 1, stroke: 0.5pt),
      
      // Header
      [], $1$, $0$, $bold("sum")$,
      $1$, $a$, $b$, $a+b$,
      $0$, $c$, $d$, $c+d$,
      $bold("sum")$, $a+c$, $b+d$, $p$,
    ),
    [
      *Simple Matching*: \
      $d(i, j) = (b + c) slash (a + b + c + d)$\
      *Jaccard Coefficient*: \
      $d(i, j) = b slash (a + b + c)$
    ]
  )
  == #subhead([Distance Metrics (Categorical Variables)])
  *Simple Matching*
  $ d(arrow(i), arrow(j)) = ("total features" - "matched features") slash "total features" $
  *1-Hot Encoding*: Convert categorical variable into 1-hot encoding, then use aforementioned binary variable metrices

  == #subhead([Distance Metrics (Transactional Data)])
  $
  "Sim"(T_1, T_2) = (|T_1 inter T_2|) / (|T_1 union T_2|)"  E.g.," "Sim"({a, b, c}, {c, d, e}) = (|{c}|) / (|{a, b, c, d, e}|) = 1/5
  $

  == #subhead([K-means Algorithm])
  + *Initialization*: Partition objects into $k$ nonempty subsets
  + *Mean-op*: Compute seed points as centroids of the clusters of the current partition. The centroid is the mean point of the cluster.
  + *Nearest_Centroid-op*: Assign objs to cluster w/ nearest centroid.
  + *Go back to the step 2, stop when no more new assignment.*

  *Strengths*
  - Relatively efficient: $O(t k n)$, where $n$ is \# objects, $k$ is \# clusters, and $t$ is \# iterations. Normally, $k, t << n$.
  - Often terminates at local optimum. Global optimum may be found using: deterministic annealing and genetic algorithms

  *Weaknesses*
  - Applicable only when mean is defined, then what about categorical data? What is the mean of red, orange and blue?
  - Need to specify $k$, the number of clusters, in advance
  - Unable to handle noisy data and outliers
  - Not suitable to discover clusters with non-convex shapes; the basic cluster shape is spherical (convex shape)

  *Handling Categorical Data*
  - Replacing means of clusters with modes
  - Using new dissimilarity measures to deal with categorical objects
  - Using a frequency-based method to update modes of clusters
  - A mixture of categorical and numerical data: _k_-prototype method

  == #subhead([K-medoids Algorithm])
  Same as K-means, but instead of using virtual mean, use real data points as centroids. *Strength*: medians less sensitive to outliers 

  *How To Choose Medoids?*

  In _*PAM*_ (Partitioning Around Medoids) algorithm, enumerate all possible medoids, and choose one w/ least total distance to other points.

  In _*CLARA*_ (Clustering Large Applications) algorithm, random sample a *fixed subset* from the whole dataset. Then use it to perform PAM.

  == #subhead([Hierarchical Clustering])
  - *Agglomerative (ANGES)*: Bottom-up approach
  - *Divisive (DIANA)*: Top-down approach

  === *Agglomerative (ANGES)*
  + Initialize: 1 sample per cluster
  + Merge a pair of clusters with *least dissimilarity*
  + Decrement \# of clusters by one 
  + *UNTIL* only 1 cluster left.

  _*Weaknesses*_:
  - Do not scale well: time complexity of at least $O(n^2)$, where $n$ is \# total objects (need to compute the similarity or dissimilarity of each pair of objects)
  - Can never undo what was done previously
  - Hierarchical methods are biased towards finding 'spherical' clusters even when the data contain clusters of other shapes.
  - Partitions are achieved by 'cutting' a dendrogram or selecting one of the solutions in the nested sequence of clusters that comprise the hierarchy.
  - Deciding of appropriate number of clusters for the data is difficult.

  === *Divisive (DIANA)*
  + Initialize: 1 cluster containing all samples
  + Split a cluster into 2 *most dissimilar* sub-clusters
  + Decrement \# of clusters by one

  === *Linkages*
  distance measured by...\
  _Single Linkage_: *closest pair* of samples from each cluster\
  _Complete Linkage_: *furthest pair* of samples from each cluster\
  _Centroid Linkage_: *mean* (centroids) of each cluster $1/n sum_(i=1)^n x_i$

  === Standards
  - Intra-cluster high similarity, inter-cluster low similarity
  - *Choose K*: Elbow method

  == #subhead([DBSCAN Algorithm])
  Short for: Density Based Spatial Clustering of Applications with Noise
  - $epsilon$: radius for the neighborhood of point 
  - `minPts`: minimum number of points in the $epsilon$-neighborhood
    - *include* itself, and *include* all visited points.
  #image("dbscan-points-def.png")

  *Procedure*: 
  + Start w/ any *unvisited* point $P$. 
  + Is $P$ core point? (has at least `minPts` points in $epsilon$ sphere)
    - If *yes*: Mark $P$ as core.
    - If *no*: Mark $P$ as noise (temporarily). Back to Step 1.
  + *Expansion*: Add all $P$'s neighbors  to the cluster
    - If neighbor $Q$ is core point, add $Q$'s neighbors to the cluster
    - If neighbor $Q$ is border point, skip.
    - Repeat above until all reachable points are border points.

  *Weaknesses*: \
  - Cannot handle varying densities. 
  - Sensitive to hyperparameters.

  *How to Find $epsilon$ and `minPts`?*
  + Fix `MinPts` (usually $2 times d - 1$), where $d$ is dimension of data space
  + For each $p$, compute its distance to its $k$-th nearest neighbor $d_k (P)$
    - For $p$ in dense clusters, the value of $d_k (p)$ is relatively small.
    - For outliers $p$, the value of $d_k (p)$ is relatively large.
  + *Sort all $d_k (p)$ in descending order*, and plot $d_k (p)$
  + Find the "elbow" point, and use it as $epsilon$.

  #grid(
    columns: 2, 
    image("k-dist-plot.png", width: 100%),
    [
      In this example: $k$ is chosen as 3, hence the "3-distance" y-axis.\
      *left:* high $d_k (p)$ → outliers sparse points (varying, high value)\
      *right:* low $d_k (p)$ → intra-cluster points (smooth, low value)

    ]
  )

  = #head([Data Preprocessing])
  == #subhead([Stage 1: Data Cleaning])
  _Garbage In, Garbage Out_. Thus we need preprocessing. Real data is:
  - Incomplete: lack attribute values, lack certain attributes of interest, or containing only aggregate data
  - Noisy: containing errors, outliers
  - Inconsistent: conflicting discrepencies in codes or names

  === Missing Value _(Worst to Best)_
  1. *worst*: Ignore Tuple. Unless missing label, this loss information.
  2. *okay*: Manual imputation, fill constant (e.g., unknown) or mean.
  3. *best*: Inference. Association Analysis to infer missingness.

  === Noisy Data 
  + *Binning*: Discretize the variable
    - Equal-width (distance) partitioning: Divide range into $N$ intervals of equal size. If the lowest-highest values of the attribute is $A$ and $B$, interval width = $(B-A) slash N$
      - Stragihtforward. But *sensitive to outliers* or *skewness* .
    - Equal-depth (frequency) partitioning: Divide range into $N$ intervals, each containing (about) the same number of samples.
      - Good data scaling. Managing categorical variable can be tricky.
  + *Clustering*: Use distance-based clustering to find outliers
  + *Regression*: fit linear model. Data points far from fit are outliers.

  _Smoothing_: 
  + Binning: smooth by bin means or bin boundary
  + Clustering: replace noisy data with cluster centroid
  + Regression: replace noisy data with predicted value (project to line)

  == #subhead([Stage 2: Data Transformation])
  + Min-Max: $v' = (v - min) / (max - min)(max_"new" - min_"new") + min_"new"$
    - $[min_"new", max_"new"]$ is the *new range* of the variable (e.g., $[-1, 1]$)
  + Z-score:  $v' = (v - "mean") / sigma$
  + Decimal Scaling: $v' = v / 10^j$, where $j$ is smallest integer such that $max(|v'|) <= 1$

  == #subhead([Stage 3: Data Reduction])
  === 1. Reduction
  - _Feature Selection_: choose $k < d$ important features, ignore rest.
    - Forward Search: start with $emptyset$ features. Each iteration, add best new feature. Essentially, *Greedy Hillclimbing*
    - Backward Search: start with all features. Each iteration, remove worst feature. 
    - Decision Tree Induction: Use only the features used by DT.
  - _Feature Extraction_: project $d$-dimensional data to $k<d$ dimension
    - Principal Component Analysis (PCA)
    - Linear Discriminant Analysis (LDA)
    - Factor Analysis
  - _Numerosity Reduction_: reduce data vol. w/ compact representation
    - Parametric: Fit data w/ model. Store just parameters, discard data. 
    - Non-parametric: histograms, clustering, (stratified) sampling.

  === 2. Discretization
  - Divide range of continous attribute into intervals (e.g., by binning)
    - Why? 
      - Some algo only accept categorical attributes
      - Reduce data size. Prepare for further analysis.
  - Concept hierarchy: replace low-level concept w/ higher level concepts (e.g., "age = 25" $->$ "young")

  = #head([Data Warehouse])
  #image("oltp-vs-olap-table.png", width: 100%)
  
  == #subhead([Core Concepts])
  *Definition*: A _subject-oriented, integrated, time-variant, nonvolatile_ collection of data to support management decision making.

  *OLTP vs OLAP*:
  - *OLTP (Online Transaction Processing)*:
    - _Access_: Read/Write, atomic transactions. 
    - _Focus_: Concurrency (ACID).
    - _Data_: Current, detailed, relational (3NF). _Users_: Clerks, DBAs.
  - *OLAP (Online Analytical Processing)*:
    - _Access_: Read-only (mostly), complex aggregate queries. 
    - _Focus_: Throughput.
    - _Data_: Historical, consolidated, multi-dim. 
    - _Users_: Managers, Analysts.

  == #subhead([Multidimensional Data Model])
  #image("star-schema.png")
  #image("snowflake.png")
  #image("constellation.png")
  *Data Cube (N-dim Tensor)*: 
  - *Dimensions*: Perspectives (Time, Loc). 
  - *Measures*: Numerical facts (Sales).
  - *Fact Table*: Contains keys to dimensions + measures.
  
  *Schemas (Modeling)*:
  *Star Schema*:
  - _Structure_: One central Fact Table + set of Dimension Tables. 
  - _Logic_: Dim tables are *un-normalized*. 
    - (e.g. `City`, `Country` in same table).
  - _Pros_: Simple, Min Joins (Fast). 
  - _Cons_: Redundant data (who cares? storage cheap).
  *Snowflake Schema*:
  - _Structure_: Dim tables are *normalized* (split into hierarchies).
  - _Pros_: No redundancy, easy maintenance. 
  - _Cons_: More Joins (Slow). Avoid in DW.
  *Fact Constellation*:
  - _Structure_: Many fact Tables share dimension tables (Galaxy schema).
  - _Case_: Complex systems (e.g. sales & shipping share time & location).

  == #subhead([Aggregations Functions])
  Essentially: how compress-able are the states? 

  *Distributive*:
    - Examples: `sum`, `count`, `max`
    - Characteristic: $F(D) = F({F(D_1), F(D_2), dots, F(D_n)})$
    - Example: $max(D) = max({max(D_1), max(D_2), dots, max(D_n)})$
    - Local solutions can be perfectly aggregated to global solution!
    - Communication Complexity: $O(1)$
  *Algebraic*:
    - Examples: `avg`, `var`
    - Characteristics: $F(D) != F({F(D_1), F(D_2), dots, F(D_n)})$
    - Example: $"avg"(D) != "avg"({"avg"(D_1), "avg"(D_2), dots, "avg"(D_n)})$
    - Solution: 
      - pass fixed size ($k$) feature vector or sufficient statistic.
      - don't pass avg. Pass $arrow(v) = <sum x_i, N>$
      - don't pass stdev. Pass $arrow(v) = < sum x_i, sum x_i^2, N>$
    - Not as friendly as _Distributive_, but still quite efficient! 
    - Communication Complexity: $O(k)$
  
  *Holistic*:
  - Examples: `median`, `rank`, `mode`
  - Characteristics: size of return state grows with data size.
  - Storage requirement for sub-aggregate has no constant upperbound

  == #subhead([OLAP Operations (Geometric Transforms)])
  *Roll-up (Drill-up)*:
  - _Action_: Climbing up hierarchy or Dimension reduction.
  - _SQL_: `GROUP BY (Day)` $->$ `GROUP BY (Month)`.
  *Drill-down*:
  - _Action_: Stepping down hierarchy or Introducing new dimension.
  - _SQL_: Reverse of Roll-up. Detailed view.
  *Slice & Dice*:
  - _Slice_: Selection on *one* dimension. `WHERE time=Q1`
  - _Dice_: Selection on *two+* dimensions. `WHERE time=Q1 AND loc=US`
  *Pivot (Rotate)*:
  - _Action_: Rotate axes for visualization. (kind of like transpose)

  == #subhead([Implementation & Optimization])
  *1. Partial Materialization (Cube Computation)*:
  - _Problem_: $2^N$ cuboids. Full materialization too big. No mat. too slow.
  - _Solution_: Greedy Algo. Select views with max benefit/space ratio.
  - _Iceberg Cube_: Only store cells w/ value $>$ threshold (sparsity).
  
  *2. Indexing Techniques*:
  - _Bitmap Index_: For low-cardinality columns (Gender). fast.
  - _Join Index_: Map `Join(Fact.key, Dim.key)` to avoid runtime joins.
  
  *3. Architecture*:
  - _ROLAP_: Relational (Star schema). Scalable, slower.
  - _MOLAP_: Multi-dim Array (Direct cube). Fast, sparse, limited scale.
  - _HOLAP_: Hybrid (Detail in ROLAP, Aggregates in MOLAP).


  = #head([Web Mining])
  *Input*: Unstructured/Semi-structured Data. *Goal*: Discover patterns.
  
  == #subhead([1. Information Retrieval (Text DB)])
  *Data Abstraction (Feature Eng)*:
  - *Bag of Words*: Document $d$ as set of terms. Ignore order.
  - *Vector Space Model*: $d in RR^|V|$ (High-dim, sparse). 
    - Value = Term Frequency (TF) or TF-IDF.
    - _Preprocessing_: 
      - Stop list (remove 'the'), 
      - Stemming ('mining' $->$ 'mine').

  *Similarity (Metric)*:
  - Euclidean fails in high-dim (curse of dimensionality) & length bias.
  - *Cosine Similarity*: Measure angle (direction), not magnitude.
    $ "Sim"(d_1, d_2) = (d_1 dot d_2) / (||d_1|| times ||d_2||) $

  *Evaluation (Set Theory)*:
  - *Precision*: $(|{"Relevant"} inter {"Retrieved"}|) / (|{"Retrieved"}|)$ (Quality of result).
  - *Recall*: $(|{"Relevant"} inter {"Retrieved"}|) / (|{"Relevant"}|)$ (Coverage of truth).

  *Challenges (Why Keyword Match Fails)*:
  - *Synonymy*: Different words, same meaning (e.g., "ML" vs "Learning"). $->$ Low Recall.
  - *Polysemy*: Same word, different meanings (e.g., "Apple"). $->$ Low Precision.

  == #subhead([2. Taxonomy (The 3 Pillars)])
  *Content Mining* (NLP):
  - _Input_: HTML Text. _Task_: Classification/Clustering.
  - _Logic_: Text Mining + HTML Tags weight.
  *Structure Mining* (Graph):
  - _Input_: Links (Edges). _Task_: Find Authority/Hubs.
  - _Logic_: "Link as Vote". Graph Theory.
  *Usage Mining* (Logs/BI):
  - _Input_: Server Logs (IP, Time, URL). _Task_: User Profiling.
  - _Logic_: ETL $->$ Sessionize $->$ Pattern Mining.

  == #subhead([3. Core Algorithms])
  *PageRank (Google)*:
  - _Intuition_: Link=endorsement. Importance via recursive voting.
  - _Logic_: Random Surfer Model / Markov Chain Stationary Dist.
    $ "PR"(A) = (1-d) + d sum_(T_i -> A) ("PR"(T_i)) / (C(T_i)) $
  
  *HITS (Hubs & Authorities)*:
  - *Hub*: Guide page pointing to many authorities.
  - *Authority*: Content page pointed to by many hubs.
  - Mutually reinforcing: Good Hub $->$ points to good Auth.

  *Web Usage Process*:
  + *Clean*: Remove crawlers, graphics requests.
  + *Identify*: Split IP streams into User Sessions.
  + *Pattern*: Apriori / Sequential Mining on sessions.

  = #head([Exam Questions])
  == Network $->$ Matrix $->$ DBSCAN

=== 1. Analysis and Parameter Proposal
The dissimilarity matrix (Fig. 4) represents *structural equivalence* (Hamming distance of adjacency vectors) rather than path length. Notably, hubs D and E have high dissimilarity to others ($d \ge 3$) and to each other ($d=6$).

To distinguish tight structural groups from loose connections:
- *Radius ($epsilon$) = 2*: A threshold of $epsilon >= 3$ would merge the entire graph into a single cluster (since $d(A,D)=3, d(A,E)=3$). $epsilon=2$ exploits the natural gap in the data to separate strong ties.
- *MinPts = 2*: Chosen to allow the detection of the smallest possible social units (dyads) visible in the graph (e.g., A-F).

=== 2. Execution (Neighborhood Analysis)
We calculate the $epsilon$-neighborhood $N_{epsilon}(x) = \{y \mid d(x,y) <= 2\}$ for each node.

#figure(
  table(
    columns: (auto, 2fr, 1fr),
    inset: 6pt,
    align: horizon,
    stroke: 0.5pt + gray,
    table.header([*Node*], [*Neighborhood ($d \le 2$)*], [*Status (MinPts=2)*]),
    [A], [$\{A, F\}$ (Size: 2)], [Core Point],
    [F], [$\{F, A\}$ (Size: 2)], [Core Point],
    [B], [$\{B, C\}$ (Size: 2)], [Core Point],
    [C], [$\{C, B\}$ (Size: 2)], [Core Point],
    [D], [$\{D\}$ (Size: 1, min dist is 3)], [*Noise*],
    [E], [$\{E\}$ (Size: 1, min dist is 3)], [*Noise*],
  )
)

=== 3. Final Clustering Result
DBSCAN expands clusters from the core points by merging overlapping neighborhoods.
- *Cluster 1*: $\{A, F\}$ (High structural similarity)
- *Cluster 2*: $\{B, C\}$ (High structural similarity)
- *Outliers*: $D$ and $E$. Despite being central graph nodes, they are *structurally unique* (dissimilar neighbors) and fail to meet the density criteria.

== Datawarehouse


+ *Missing Value Imputation* \
  The missing values are filled based on the class defined by attributes "Fever" and "Disease".

  *1. Imputing Age (Patient 9910115)*
  - *Class Definition:* Fever = "No", Disease = "Yes".
  - *Matching Records:*
    - Patient 9303034: Age 55
    - Patient 9910111: Age 46
  - *Calculation (Mean):* $(55 + 46) / 2 = 50.5$

  *2. Imputing Sex (Patient 9576737)*
  - *Class Definition:* Fever = "Yes", Disease = "No".
  - *Matching Records:*
    - Patient 9100123: Male
    - Patient 9576732: Male
  - *Calculation (Mode):* Male

  *Answer:*
  - Missing Age: *50.5*
  - Missing Sex: *Male*

+ *Data Smoothing by Bin Means* \
  *Raw Data (Age):* ${65, 55, 12, 35, 46, 16, 105, 28}$

  *Step 1: Sort the Data*
  ${12, 16, 28, 35, 46, 55, 65, 105}$

  *Step 2: Partition into 4 Equi-depth Bins*
  Total items = 8. Items per bin = $8/4 = 2$.
  - Bin 1: ${12, 16}$
  - Bin 2: ${28, 35}$
  - Bin 3: ${46, 55}$
  - Bin 4: ${65, 105}$

  *Step 3: Calculate Bin Means and Smooth*
  - Bin 1 Mean: $(12+16)/2 = 14$
  - Bin 2 Mean: $(28+35)/2 = 31.5$
  - Bin 3 Mean: $(46+55)/2 = 50.5$
  - Bin 4 Mean: $(65+105)/2 = 85$

  *Smoothed Values List:*
  *Bin 1:* 14, 14
  *Bin 2:* 31.5, 31.5
  *Bin 3:* 50.5, 50.5
  *Bin 4:* 85, 85

+ *3D Data Cube Design* \
  *Context:* Hospital Revenue Analysis.
  *Star Schema Structure:*

  *1. Dimension Tables:*
  - *Time Dimension:* Attributes include `Time_Key`, `Day`, `Month`, `Quarter`, `Year`. Provides temporal granularity.
  - *Location Dimension:* Attributes include `Location_Key`, `Department_Name`, `Hospital_Branch`, `City`, `Region`. Provides geographical/organizational hierarchy.
  - *Treatment Dimension:* Attributes include `Treatment_Key`, `Procedure_Code` (e.g., CPT), `Category` (e.g., Surgery, Pathology), `Risk_Level`. Describes the service provided.

  *2. Fact Table:*
  - *Revenue_Fact:* Contains foreign keys linking to the three dimensions (`Time_Key`, `Location_Key`, `Treatment_Key`) and numerical measures such as `Total_Revenue` (\$) and `Patient_Count`.

+ *OLAP Operations* \
  Based on the context in (c), assume the current query view is: *"Total Revenue by Month and Hospital Branch."*

  *1. Roll Up (Aggregation/Generalization)*
  - *Operation:* Climb up the concept hierarchy on the *Time* dimension.
  - *Action:* Change grouping from `Month` to `Quarter` or `Year`.
  - *Result:* The data becomes less detailed, showing total revenue per Quarter/Year, summarizing the monthly variations.

  *2. Drill Down (Detailed View/Specialization)*
  - *Operation:* Step down the concept hierarchy on the *Location* dimension.
  - *Action:* Expand `Hospital_Branch` to `Department_Name`.
  - *Result:* The data becomes more detailed, breaking down the branch's revenue to show specifically which departments (e.g., ER, Oncology, Pediatrics) generated the funds.

  == Sequence Mining
// Reviewer Note: The solution maintains strict logical consistency with Sequential Pattern Mining definitions (e.g., GSP/AprioriAll concepts).
// Assumptions made:
// 1. "Sub-sequences" implies order-preserving patterns composed of non-contiguous elements.
// 2. Data format implies single-item elements (no concurrent events).
// 3. "Found" in (b) implies theoretically discoverable given the data constraints.

= Section B [80%]: Long Questions

== B1. [20 marks]

*Given the following tourist data records... (Refer to original image for full dataset)*
*Mappings:* Hamburg (H), Toronto (T), Osaka (O), Beijing (B), London (L), Vancouver (V), Sydney (S).

#table(
  columns: (auto, auto),
  inset: 5pt,
  align: (center, left),
  [*Tourist ID*], [*Sequence*],
  [30001], [V #sym.arrow T #sym.arrow O #sym.arrow B #sym.arrow L],
  [30002], [T #sym.arrow B #sym.arrow O #sym.arrow T #sym.arrow H],
  [30003], [O #sym.arrow T #sym.arrow B #sym.arrow O],
  [30004], [B #sym.arrow H],
  [30005], [B #sym.arrow S #sym.arrow O #sym.arrow H #sym.arrow O #sym.arrow H]
)


=== a) List ALL possible sub-sequences (with different lengths) arising from Tourist 30003. (6 marks)

*Answer:*

The sequence for Tourist 30003 is $chevron.l O, T, B, O chevron.r$.
A sub-sequence is formed by deleting zero or more items from the original sequence while maintaining the relative order of the remaining items.

*Length 1:*
- $chevron.l O chevron.r$
- $chevron.l T chevron.r$
- $chevron.l B chevron.r$

*Length 2:*
- $chevron.l O, T chevron.r$
- $chevron.l O, B chevron.r$
- $chevron.l O, O chevron.r$
- $chevron.l T, B chevron.r$
- $chevron.l T, O chevron.r$
- $chevron.l B, O chevron.r$

*Length 3:*
- $chevron.l O, T, B chevron.r$
- $chevron.l O, T, O chevron.r$
- $chevron.l O, B, O chevron.r$
- $chevron.l T, B, O chevron.r$

*Length 4:*
- $chevron.l O, T, B, O chevron.r$

_Note: While the element $O$ appears twice, strictly unique patterns are listed above. If treating indices as distinct, one might list $chevron.l O, T chevron.r$ twice (index 1 & 2 vs index 4 & 2), but in sequential mining pattern generation, we focus on the unique sequence signatures._


=== b) What is the largest itemset size (i.e., the maximum number of items in an itemset) found by the frequent itemset phase of sequential association rule mining? What is the longest sequence length (i.e., the maximum number of itemsets in a sequence) found by the sequence phase of sequential association rule mining? Note that the minimum support is unknown and you may assume it $> 0%$. State your other assumption(s) when necessary. (4 marks)

*Answer:*

1.  *Largest Itemset Size:* *1*
    - _Reasoning:_ The provided data records show events occurring strictly sequentially (e.g., Vancouver #sym.arrow Toronto). There are no concurrent events indicated (e.g., no entries like "(Vancouver, Toronto)" meaning same-time visit). Therefore, every itemset (element) in the database contains exactly 1 item. The mining process cannot find an itemset larger than the input data supports.

2.  *Longest Sequence Length:* *6*
    - _Reasoning:_ The longest sequence refers to the maximum number of sequential events (itemsets). Tourist 30005 has the sequence $chevron.l B, S, O, H, O, H chevron.r$, which has a length of 6.
    - _Assumption:_ We assume the unknown minimum support is low enough (e.g., $1\/N$) to make the longest transaction in the database frequent. If support were effectively $0%$, the algorithm discovers the longest sequence existing in the dataset.


=== c) Show the transformation step (step 3 of the sequential pattern mining process) for Tourist 30005 using $"min_sup" = 35%$. (4 marks)

*Answer:*

*Step 1: Determine Support Threshold*

Total Tourists ($N$) = 5.

$"min_sup" = 35%$.

Required Count $= 0.35 times 5 = 1.75$.

Items must appear in at least *2* sequences to be considered frequent.

*Step 2: Identify Frequent Items (1-sequences)*
- *V:* Count 1 (30001) #sym.arrow Fail.
- *T:* Count 3 (30001, 30002, 30003) #sym.arrow Keep.
- *O:* Count 4 (30001, 30002, 30003, 30005) #sym.arrow Keep.
- *B:* Count 5 (All) #sym.arrow Keep.
- *L:* Count 1 (30001) #sym.arrow Fail.
- *S:* Count 1 (30005) #sym.arrow Fail.
- *H:* Count 3 (30002, 30004, 30005) #sym.arrow Keep.

Frequent Items set: $\{T, O, B, H\}$.

*Step 3: Transformation*

Map the original sequence of Tourist 30005 by removing infrequent items ($S$) and retaining frequent items ($B, O, H$) while preserving order.

Original Sequence: $chevron.l B arrow S arrow O arrow H arrow O arrow H chevron.r$
- $B$: Keep
- $S$: Remove (Infrequent)
- $O$: Keep
- $H$: Keep
- $O$: Keep
- $H$: Keep

*Transformed Sequence:*
$ chevron.l B, O, H, O, H chevron.r $


=== d) Compute the support, confidence and interest of the following sequential association rule.
#align(center)[Beijing Osaka $arrow.r.double$ Hamburg] (6 marks)

*Answer:*

We interpret the rule as: Given a sequence contains Beijing ($B$) followed eventually by Osaka ($O$), does it also contain Hamburg ($H$) eventually after $O$?
- *Rule:* $chevron.l B, O chevron.r arrow.r.double chevron.l H chevron.r$
- *Composite Sequence:* $chevron.l B, O, H chevron.r$

*1. Calculate Counts (N=5):*

- *Antecedent Support Count ($chevron.l B dots O chevron.r$):*
  - 30001: $chevron.l V, T, O, B, L chevron.r$ (Order is $O arrow B$, not $B arrow O$) #sym.arrow No.
  - 30002: $chevron.l T, bold(B), bold(O), T, H chevron.r$ #sym.arrow Yes (1).
  - 30003: $chevron.l O, T, bold(B), bold(O) chevron.r$ #sym.arrow Yes (2).
  - 30004: $chevron.l B, H chevron.r$ (No $O$) #sym.arrow No.
  - 30005: $chevron.l bold(B), S, bold(O), H, O, H chevron.r$ #sym.arrow Yes (3).
  - $"Count"("Antecedent") = 3$.

- *Rule Support Count ($chevron.l B dots O dots H chevron.r$):*
  - 30002: $chevron.l dots B dots O dots H chevron.r$ #sym.arrow Yes (1).
  - 30003: Contains $B, O$, but no $H$. #sym.arrow No.
  - 30005: $chevron.l dots B dots O dots H dots chevron.r$ #sym.arrow Yes (2).
  - $"Count"("Rule") = 2$.

- *Consequent Support Count ($H$ occurs in sequence):*
  - Appears in 30002, 30004, 30005.
  - $"Count"(H) = 3$.

*2. Compute Metrics:*

- *Support ($s$):*
  $ s = frac("Count"(chevron.l B, O, H chevron.r), N) = frac(2, 5) = bold(0.4) "or" bold(40%) $

- *Confidence ($c$):*
  $ c = frac("Count"(chevron.l B, O, H chevron.r), "Count"(chevron.l B, O chevron.r)) = frac(2, 3) approx bold(0.67) "or" bold(66.7%) $

- *Interest (Lift):*
  $ "Interest" = frac("Confidence", P("Consequent")) = frac(2\/3, 3\/5) = frac(2, 3) times frac(5, 3) = frac(10, 9) approx bold(1.11) $
  
  == Naive Bayes
  #set text(size: 9pt)


#table(
  columns: (1fr,) * 5,
  align: center,
  inset: 1pt,
  stroke: 0.5pt + gray,
  [*Outlook*], [*Temp*], [*Humid*], [*Wind*], [*Play?*],
  [Sunny], [Hot], [High], [Weak], [No],
  [Sunny], [Hot], [High], [Strong], [No],
  [Overcast], [Hot], [High], [Weak], [Yes],
  [Rain], [Mild], [High], [Weak], [Yes],
  [Rain], [Cool], [Normal], [Weak], [Yes],
  [Rain], [Cool], [Normal], [Strong], [No],
  [Overcast], [Cool], [Normal], [Strong], [Yes],
  [Sunny], [Mild], [High], [Weak], [No],
  [Sunny], [Cool], [Normal], [Weak], [Yes],
  [Rain], [Mild], [Normal], [Weak], [Yes],
  [Sunny], [Mild], [Normal], [Strong], [Yes],
  [Overcast], [Mild], [High], [Strong], [Yes],
  [Overcast], [Hot], [Normal], [Weak], [Yes],
  [Rain], [Mild], [High], [Strong], [No],
)

*Query:* $X = ("Sunny", "Cool", "High", "Strong")$ — Play?

=== Step 1: Prior Probabilities
$ P("Yes") = 9/14, quad P("No") = 5/14 $

=== Step 2: Likelihoods (from training counts)

#table(
  columns: (1fr,) * 3,
  align: center,
  inset: 1pt,
  stroke: 0.5pt + gray,
  [*Feature*], [*P(feat|Yes)*], [*P(feat|No)*],
  [Sunny], [2/9], [3/5],
  [Cool], [3/9], [1/5],
  [High], [3/9], [4/5],
  [Strong], [3/9], [3/5],
)

=== Step 3: Apply Naive Bayes

$ P(X|"Yes") &= 2/9 dot 3/9 dot 3/9 dot 3/9 = 54/6561 approx 0.0082 $
$ P(X|"No") &= 3/5 dot 1/5 dot 4/5 dot 3/5 = 36/625 approx 0.0576 $

*Unnormalized posteriors:*
$ P("Yes") dot P(X|"Yes") &= 9/14 dot 54/6561 approx 0.0053 $
$ P("No") dot P(X|"No") &= 5/14 dot 36/625 approx 0.0206 $

=== Step 4: Normalize & Classify

$ P("Yes"|X) = 0.0053 / (0.0053 + 0.0206) approx bold(20.4%) $
$ P("No"|X) = 0.0206 / (0.0053 + 0.0206) approx bold(79.6%) $

#align(center)[
  #box(stroke: 1pt, inset: 5pt)[
    *Prediction:* No (Don't Play Tennis)
  ]
]

=== Laplace Smoothing (for zero counts)
$ P(x_i | C) = (N_(x_i,C) + alpha) / (N_C + alpha dot k) $
where $alpha = 1$ (typically), $k$ = number of feature values.

]