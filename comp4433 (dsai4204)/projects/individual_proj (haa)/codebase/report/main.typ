#import "@preview/bloated-neurips:0.7.0": botrule, midrule, neurips2025, paragraph, toprule, url
#import "@preview/xarrow:0.4.0": xarrow, xarrowSquiggly, xarrowTwoHead

#let affls = (
  polyu: (
    department: "Department of Computing",
    institution: "PolyU",
  )
)

#let authors = (
  (
    name: "WANG Yuqi",
    affl: "polyu",
    email: "[redacted]@connect.polyu.hk",
    equal: true
  ),
)

#show: neurips2025.with(
  title: [Beyond Binary Prediction:\ Calibrated Risk Stratification for Pattern Discovery],
  authors: (authors, affls),
  keywords: ("Machine Learning", "Kaggle", "DSAI4204", "House Price Prediction"),
  abstract: [#align(center)[
       Most heart attack analysis seen on Kaggle treats the task as binary classification, optimizing for accuracy at the expense of calibration and interpretability. To address this, I developed a unified framework beyond binary predictions, via three interconnected stages: (1) *calibrated classification* with vectorized Monte Carlo perturbation and calibration error minimizing ensemble, (2) *manifold clustering* of Tabular Foundation Model (TFM) embeddings residualized against calibrated logits, and (3) *pattern discovery* using association rule mining and subgroup discovery. Experimenting on the UCI Cleveland heart disease dataset demonstrates superior pattern discovery capacity of my cohesive analytical pipeline when compared against treating classification, clustering, and mining as isolated tasks. The core insight is that calibrated estimates should serve not as terminal outputs, but as structural priors that guide downstream manifold analysis and pattern discovery.
  ]],
  bibliography: bibliography("4204.bib"),
  bibliography-opts: (title: none, full: true),  // Only for example paper.
  appendix: [],
  accepted: true,
)
#show table.cell: set text(size: 8.6pt)
#set figure.caption(position: bottom)
#show link: l => [
    #set text(font: "PT Serif",size: 11pt)
    #box(stroke: 1pt + rgb(0, 255, 255))[#l]
]

#let sref(label) = {
  link(label)[§#ref(label, supplement: none)]
}

#let answer-block(content) = {
  text(
    [
      #rect(
        [
          #content
        ],
        fill: gray.lighten(95%),
        width: 100%,
        stroke: gray.lighten(40%),
        inset: 1em
      )
    ]
  )
}
#let lk(content) = text()[#underline[(_*#content*_)]]
#import "@preview/wrap-indent:0.1.0": wrap-in, allow-wrapping
#show terms.item: allow-wrapping

= Introduction<introduction>
/ #wrap-in(answer-block):
  *Project Highlights*\
  #grid(columns: 2,
    [
      + Unified three-stage pipeline with principled information flow
      + Vectorized Monte Carlo perturbation (100$times$100 in 30 min)
      + Calibration-optimized ensemble minimizing $"ECE"^"KDE"$
      + TFM embedding residualization via RBF kernel regression
      + Risk-stratified mining outperforms binary-label baselines
    ],
    [
      #align(right)[
      #lk([#link(<roadmap>)[@roadmap]])\
      #lk([#link(<calibrated-classification>)[@calibrated-classification]])\
      #lk([#link(<calibrated-classification>)[@calibrated-classification]])\
      #lk([#link(<manifold-clustering>)[@manifold-clustering]])\
      #lk([#link(<pattern-discovery>)[@pattern-discovery]])\
      ]
    ]
  )
  *Notably:* I astutely identified that the original dataset labels were _incorrect_, revealing data quality issues in the Kaggle dataset. See #lk([#link(<appendix>)[@appendix]]) for more details.
#v(1em)


Machine Learning (ML) methods have been extensively applied to the Cleveland Heart Disease dataset on Kaggle. However, these methods approach healthcare analysis as a binary classification task, optimizing for accuracy metrics that obscures the continous nature of disease risk, and thus, neglecting the calibration necessary for clinical decision-making. This methodological gap is even more noticeable when practitioners seek insights beyond binary predictions, but understanding of the underlying risk patterns for subgroup identification and actionable early interventions.


For this project, a naive approach is to treat classification, clustering, and mining as orthogonal tasks. A typical workflow might start from training a classifier, then separately a clustering algorithm for performance comparisons, and finally Association Rule Mining (ARM) for pattern discovery and interpretability. In such approach, each stage operates in isolation with minimal to no information flow between modules. While convenient, such disjoint approach suffers three limitations: 

+ Binary predictions discard continuous information inherent in disease risk.
+ Raw feature space failed to capture non-linear relationships and nuanced topological structure.
+ Discovered patterns are not grounded in calibrated risk assessments.

#paragraph[Project Highlight] To overcome the aforementioned challenges, I designed a unified framework through _principled information propagation_ across the three interconnected stages. Instead of treating each as isolated tasks, I architect a schematic pipeline where each stage enriches the next. 

+ *Calibrated Classification (#sref(<calibrated-classification>))* A vectorized Monte Carlo perturbation protocol is developed to stress-test model stability under 100 perturbation levels, each sampled 100 times with full vectorization in a single-pass. During which, the Expected Calibration Error via Kernel Density Estimation ($"ECE"^"KDE"$) and brier score are calculated. This helps identify candidate models with high robustness and calibration. Next, grid search is performed over ensemble weights of these candidate models to produce $sum_i w_i dot hat(p)(y_i=1|x)$ that minimizes $"ECE"^"KDE"$. This yields a risk proxy more nuanced and informative than simple binary labels. 

+ *Manifold Clustering (#sref(<manifold-clustering>))* Recognizing the rich geometric structure encoded by Tabular Foundation Model (TFM) embeddings, we extract TabPFN embeddings and orthogonalize it against calibrated logits using Radial Basis Function (RBF) Kernel Ridge Regression. Clustering on these residualized embeddings reveals subpopulations that prompts further subgroup analysis. By doing so, I was able to shift an otherwise uninformative benchmark comparisons between classification and clustering models into a hypothesis generation process for pattern discovery.

+ *Pattern Discovery (#sref(<pattern-discovery>))* Calibrated probabilities are first converted into logits, then discretized into risk strata. Next, subgroup discovery algorithms alongside association rule mining are employed. Through anchoring pattern discovery to calibrated risk rather than binary labels, I was able to uncover more nuanced interpretable rules. 

The key insight of my approach is that information must flow _progressively_: calibrated probabilities from #sref(<calibrated-classification>) guide analysis in #sref(<manifold-clustering>), which in turn generates hypotheses for pattern discovery in #sref(<pattern-discovery>). Experiments on UCI Cleveland dataset ($n = 303$, $k = 13$) reveals that my framework was able to identify rules and patterns that would otherwise be missed by disjoint approach that directly run association rule mining. Experiments confirm that my former approach results in statistically richer discoveries that aligns with medical priors. Though the experiments were confined to a single small sized dataset (an inevitable constraint for a course project) the methodological insights might offer some transferrable insights. 

The remainder of this report is organized as follows: #sref(<preliminaries>) review preliminaries for this project, #sref(<roadmap>) details my way to solution and #sref(<appendix>) details some initial Exploratory Data Analysis (EDA) and transformation made to the dataset.


= Preliminaries<preliminaries>

== Tabular Foundation Models

Recent works in deep learning challenge classical machine learning methods through large-scale pretraining. TabPFN @hollmann_accurate_2025, short for Tabular Prior-Data Fitted Network, address the previous limitations of deep learning based approaches by pretraining on large synthetic distributions of tabular tasks. They are training-free, meta-learned and perform inference via in-context learning. Later work, such as TabICL @qu_tabicl_2025 and Mitra @zhang_mitra_2025, extends and scales up this method through retrieval-augmented and metric-learning approaches. Most recently, LimiX @zhang_limix_2025 introduces a unified family of Large Structured-Data Models (LDMs) that treats tabular data as a joint distribution over variables and missingness.

#paragraph[TabPFN] TabPFN amortizes the often intractable Bayesian prediction for tabular datasets by learning an explicit one-step transformation from labelled training data to test data class probabilities. Given data $D = {(x_i, y_i)}_(i=1)^n$, a query $x'$, the Bayesian target is the posterior predictive
$
p(y'|x', D) = integral p(y'|x' phi)p(D|phi)p(phi) dif f dif theta
$
Here, $phi$ ranges over *all data-generating mechanisms*. This integral essentially averages prediction over all plausible sets of underlying rules and mechanisms, weighted by how well each explains the observed data $D$. In practice, TabPFN approximates this intractable integral with a permutation-invariant Transformer trained across many synthetic tabular datasets, too, generated from such mechanisms, so that its one-shot an output approximates the integral above @hollmann_accurate_2025. 
#paragraph[LimiX] Contrary to TabPFN which is architectually designed to optimize for directrional supervised mapping $p(y'|x', D)$, LimiX disregards column nature and treat the entire dataset as a joint distribution over all variables, including output label and missingness @zhang_limix_2025. Instead of amortizing a specific posterior predictive as in TabPFN, LimiX approximates joint density of *the entire table*, enabling querying any subset of features, which is prefect for rule mining Formally, given a dataset partitioned into an in-context subset $D_c$ and query subset $D_q$, and a mask $pi subset.eq [d]$, then the masked and observed subvectors of the query samples are $D_q^pi$ and $D_q^(-pi)$, respectively. Then, LimiX minimizes the NLL: 

$
  cal(L)_k (theta) = - EE_((D_c, D_q)~p, pi~"Unif"(Pi_k))[-log q_theta (D_q^pi|D_q^(-pi),D_c)]
$

== Calibration Error
Calibration ensures that the model's estimated probabilities are faithful and match real-world likelihoods, which is cruicial for sensitive applications like heart attack analysis. A model is considered confidence-calibrated if, for all confidence levels $alpha in [0, 1]$, the model is correct on average at that confidence level ($alpha$ proportion of the times) @pavlovic_understanding_2025. The $L_p$ calibration error of $f$ can be defined as: 
$
  L_p (f) = lr((EE 
    lr([ 
      lr(norm(EE[y mid(|) f(x)] - f(x)), size: #1.5em)_p^p
    ], size: #2em)
  ), size: #2em)^(1/p)
$

#paragraph[Expected Calibration Error] Expected Calibration Error (ECE) is a widely used measure of calibration. It computes the average calibration error across binned confidence levels by taking the absolute difference between average accuracy (acc) and average confidence (conf) @guo_calibration_2017.

$
  "ECE" = sum_(m=1)^M (|B_m|)/n dot lr(|"acc"(B_m) - "conf"(B_m)|, size: #2em)
$

In this work, a variant of ECE called $"ECE"^"KDE"$ is used. Unlike its binned predecessor, $"ECE"^"KDE"$ utilizes a Beta kernel in binary classification and a Dirichlet kernel in multiclass setting, which provides a more consistent and lower bias estimate of the true calibration error $L_p$ @popordanoska_consistent_2022.


== Subgroup Discovery

Subgroup Disocvery (SD) concerns identifying patterns that deviates significantly from the norm @atzmueller_subgroup_2015.
Unlike predictive modeling, SD identifies local subsets of the data $D$ where the distribution of the target variabel $y$ is statistically unusual. The objective is to find the top-$k$ subgroups that maximizes some quality function $Q(S, y)$, where $S$ is the subgroup.

#paragraph[Quality Function] The quality function $Q(S, y)$ measures the interestingness of a subgroup $S$. Ideally, said function should balance subgroup size with the statistical significance of deviation. A common choice of quality function for binary targets is the _Weighted Relative Accuracy_ ($Q_"WRAcc"$). It measures the trade-off between coverage and precision gain over the default probabilities. 

$
Q_"WRAcc" (S, y) = (|S|)/n lr((P(y=1|S) - P(y=1)), size: #2em)
$

For numeric targets, the common choice is the Standard Quality Function. Given the subgroup mean $mu_S$, global mean $mu$, and weighting parameter $a$, the quality is defined as: 

$
Q_"std" (S) = lr(((|S|) / n), size: #2em)^a (mu_S - mu)
$




#pagebreak()
= Roadmap<roadmap>

This section details the three-stage analytical pipeline. Each stage is designed to address specific limiations of disjoint analysis while enriching subsequent stages with _principled information flow_.

Given a labelled dataset, the conventional approach is to optimize a classifier, independently apply clustering, and pattern mining algorithms. 
My framework, in contrast, connects the three stages: 
$
  cal(D) xarrow(sym: -->, "Stage 1") {hat(p)_i}_(i=1)^n xarrow(sym: -->, "Stage 2") {bold(z)_i^tack.t}_(i=1)^n xarrow(sym: -->, "Stage 3") {"Rules"}
$

where $hat(p)_i in [0, 1]$ are calibrated risk probabilities, $bold(z)_i^tack.t$ are residualized embeddings, and $"Rules"$ comprise mined patterns (e.g., through ARM or subgroup discovery). Intuitively, *later stages consume rich information from previous stages*. Specifically, calibrated risk probabilities guide embedding analysis (orthogonalization), which potentially reveals subgroup that could inform pattern discovery.

== Calibrated Classification<calibrated-classification>

#figure(
  table(
    columns: (1.1fr,) + (1fr,1.33fr,1fr) * 3,
    align: center,
    stroke: none,
    inset: (x: 0.0pt, y: 3pt),
    toprule,
    table.header(
      table.cell(colspan: 1, align: center)[], 
      table.cell(colspan: 3, align:center)[No Perturbation ($cal(l)=0$)], 
      table.cell(colspan: 3, align: center)[Mid Perturbation ($cal(l)=50$)],
      table.cell(colspan: 3, align: center)[Max Perturbation ($cal(l)=100$)],
      table.hline(start: 1, end: 10, stroke: (thickness: 0.05em)),
      [Models], 
      table.vline(stroke: 0.5pt), [ACC], [AUC-ROC], [F1], 
      table.vline(stroke: 0.5pt), [ACC], [AUC-ROC], [F1], 
      table.vline(stroke: 0.5pt), [ACC], [AUC-ROC], [F1], 
      table.hline(stroke: (thickness: 0.05em))
    ),
    [*RF*], [0.8307], [0.8271], [0.8055], [0.7824], [0.7802], [0.7583], [0.7244], [0.7242], [0.7047],
    [*KNN*], [0.8237], [0.8236], [0.8109], [0.7720], [0.7708], [0.7514], [0.7025], [0.7011], [0.6758],
    [*LimiX*], [#underline[0.8340]], [#underline[0.8311]], [#underline[0.8132]], [#underline[0.7906]], [#underline[0.7906]], [*0.7747*], [#underline[0.7252]], [#underline[0.7282]], [*0.7171*],
    [*LogReg*], [*0.8443*], [*0.8412*], [*0.8225*], [*0.7950*], [*0.7931*], [#underline[0.7725]], [*0.7329*], [*0.7332*], [#underline[0.7145]],
    [*TabPFN*], [#underline[*0.8544*]], [#underline[*0.8492*]], [#underline[*0.8237*]], [#underline[*0.8169*]], [#underline[*0.8141*]], [#underline[*0.7912*]], [#underline[*0.7590*]], [#underline[*0.7583*]], [#underline[*0.7368*]],
    [*XGBoost*], [0.7998], [0.7975], [0.7789], [0.7503], [0.7453], [0.7127], [0.6967], [0.6940], [0.6636],
    botrule
  ),
) <accuracy-level-table>
The first stage of the pipeline is called _Calibrated Classification_. Instead of directly optimizing for and selecting models by their accuracy score, the objective here is to select models that balances accuracy and reliability.
Concretely, this can be achieved via three steps: 

+ Select the top-$k$ models in terms of accuracy measures. 
+ Stress-test the model's calibration stability using a _Vectorized Monte Carlo Perturbation Protocol_.
+ Select the top performing models in the stability test and ensemble them to minimize $"ECE"^"KDE"$. 

#v(1em)
#paragraph[Step 1: Baseline Screening] 

Before evaluating model robustness, we first need screen them by their discriminative power (e.g., accuracy) under clean, unperturbed data distributions ($cal(l) = 0$). If a model is not able to achieve a reasonable accuracy, then high stability under perturbation is not meaningful. 

Results from "No Perturbation ($cal(l)=0$)" column in the table above shows that the best performing models are TabPFN, LimiX, and Logistic Regression. 

#v(1em)
#paragraph[Step 2: Vectorized Monte Carlo Perturbation Protocol] 

#figure(
  image("./figures/classify/perturb_groupby_models_perfs.png", width: 100%),
  caption: [
    Stability test showing averaged cross validation performance on six different models across 100 perturbation levels. *Higher is better*. TabPFN, LimiX, Logistic Regression, and KNN are the most invariant to perturbation, with TabPFN being the strongest. Intuitively, since perturbation diffuses data points around an ellipsoid within the feature space, slower performance degradation implies potentially *larger decision boundary margin*, which is a strong signal of model robustness.
  ],
  placement: top
)
#v(1em)

Systematically assessing model robustness requires a controlled data degradation process. Unlike traditional perturbation approach that add gaussian noise proportional to sample scale, my approach anchors the perturbation amplitude in feature distributions to ensure fairness between features. For each feature $j$, two perturbation mechanisms are defined: 

_Continous Features_: given sample $x_(i j) in RR$, global standard deviation of feature $j$, $sigma_j^"global"$, and the perturbation magnitutde $cal(l) in {1, dots, L}$, then the perturbation is sampled as:
  $
  tilde(x)_(i j)^((cal(l))) = x_(i j) + epsilon.alt dot sigma_j^"global", space epsilon.alt ~ cal(N)(0, alpha_cal(l)^2)
  $<cont-perturb-mechanism>
_Categorical Features_: given categorical $x_(i j) in {c_1, dots, c_K}$, with a categorical resample probability $beta_cal(l)$ that increases with perturbation $cal(l)$, the perturbed value $tilde(x)_(i j)^((cal(l)))$ is sampled from the empirical marginal distribution of the feature $j$ via the following process: 

  $
    b_(i j)^((cal(l))) &tilde "Bernoulli"(beta_cal(l))\
    g_(i j) &tilde "Categorical"(bold(p)_j^"global")\
    tilde(x)_(i j)^((cal(l))) &= cases(
      x_(i j)", "  b_(i j)^((cal(l))) = 0,
      g_(i j)", " b_(i j)^((cal(l))) = 1,
    )
  $<cat-perturb-mechanism>

To ensure *statistical significance* under high variation, a high repetition count ($M$) is required. However, this introduces great computational demand. When implemented as a nested loop, this would take approximately 8 hours to complete. To avoid computational bottlenecks of nested loops, the sampling process is vectorized. For $L=100$ perturbation levels and $M=100$ Monte Carlo repetitions per level, construct tensor $bold(X) in RR^((n times M) times d)$ to vectorize all $M$ monte carlo repetitions into a single pass. This lowers runtime down to 30 minutes on a single RTX4090 GPU. 

#v(1em)
#paragraph[Step 3: Calibration-Optimized Ensemble]

#figure(
  image("./figures/classify/calibration_curves_selected.png", width: 100%),
  caption: [
    Calibration curves of the three out of the six models. As shown by the curves, TabPFN and LimiX are the most calibrated, with XGBoost being the least. A jagged calibration curve indicates overfitting, as the model is over- or under-confident with data points around the decision boundary.
  ]
)
#v(1em)

Given top-$k$ calibrated models ${f_1, dots, f_k}$ in the stability test, an esmeble with weights ${w_1, dots, w_k}$ that minimizes calibration error is constructed. Then, the ensmeble's prediction is defined as: 

$
  hat(p)_"ens" (bold(upright(x))) = sum_(i=1)^k w_i dots hat(p)_i(bold(upright(x))), space space "subjected to" sum_(i=1)^k w_i = 1, w_i >= 0
$

To find the optimal weights $bold(upright(w))^* = arg min_upright(bold(w)) in "ECE"^"KDE" (bold(upright(w)))$, a simple grid search is performed with 20 random train-validation splits, each split computing a 10-fold out-of-fold (OOF) predictions. The grid search aims to minimize the $"ECE"^"KDE"$ of OOF predictions. An independent holdout testing set is also used to ensure that we are not overfitting the ensemble weights to the validation set. The final ensemble prediction is obtained by averaging the OOF prediction probabilities. 
  
#figure(
  image("figures/classify/ensemble_calibration_curve.png", width: 33.33%),
  caption: [
    Calibration curve of the weighted ensemble. The resultant ensemble is extremely well-calibrated shown by smooth curve aligning with the perfect calibration line.
    #v(1em)
  ],
)

#paragraph[Theoretical Justification]

Firstly, by screening for classification models with both high _accuracy_ and high _perturbation resistance_, it gives a set of candidate predictors whose outputs are simutaneously _informative_ and _structurally well-behaved_. Specifically, high _accuracy_ indicates the decision boundaries captures a non-trivial portion of the signal in $p(y|x)$. High _perturbation resistance_, on the other hand, adds a qualitatively different constraint: under the Monte Carlo perturbation protocol, each data point $x$ is replaced by a neighborhood distribution of plausible measurements @cont-perturb-mechanism or a plausible coding @cat-perturb-mechanism. Therefore, a model that degrades slowly under this neighborhood is, in effect, one whose decision boundary is not precariously close to the training data points; in other words, such model tends to have *larger decision margins* and *locally smoother region* (smaller local Lipschitz constant) in areas where data clusters. This matters because, on small clinical tabular datasets, brittle model can appear low-biased and accurate, yet high-variance and sample-specific.

These shortlisted predictors, ${hat(p)_i (x)}_i$, are then ensembled to form a single predictor, $hat(p)_"ens" (x)$. Geometrically, the simplex constraint $sum_i w_i = 1, w_i >= 0$ confines the meta-model $hat(p)_"ens"$ to a convex hull of base predictors $hat(p)_i$. Formally, fix the evaluation set ${x_1, dots, x_n}$. Each shortlisted predictor creates a prediction vector $bold(upright(p))_i = (hat(p)_i (x_1), dots, hat(p)_i (x_n)) in [0, 1]^n$. Then, the convex hull of the predictors is exactly the set of all convex combinations of the prediction vectors $bold(upright(p))_i$:

$
  "conv"{bold(upright(p))_1, dots, bold(upright(p))_k} = {sum_(i=1)^k w_i bold(upright(p))_i : sum_(i=1)^k w_i = 1, w_i >= 0}
$

So with $k=2$, the convex hull is the line segment between the prediction vectors; with $k=3$, it is a filled triangle whose vertices are the prediction vectors; with generalized $k$, it is a *polytope* inside $[0, 1]^n$. This forces the ensemble to satisfy the pointwise boundedness for every $x$: 
$
  min_i hat(p)_i (x) space <= space hat(p)_"ens" (x) space <= space max_i hat(p)_i (x)
$

This implies _no extrapolation_. And since the robustness of base predictors are guaranteed by Step 1-2, the weighted ensemble can be described as a projection of the true (unknown) risk function onto the convex polytope spanned by a small set of already competent predictors. Essentially, a risk proxy.


#pagebreak()
== Manifold Clustering<manifold-clustering>

#figure(
  image("figures/cluster/cluster_comparison.png", width: 100%),
  caption: [
    Comparison of the seven clustering algorithms, visualized by three different dimensionality reduction algorithms (PCA, UMAP, t-SNE). As shown by the scatter plot, Spectral, Ward, and Agglomerative are the best performing algorithms. Whereas, density-based methods like DBSCAN and HDBSCAN were unable to differentiate the two populations. 
  ]
)<cluster-comparison-none>
The second stage shifts focus from predictive modeling to *topological understanding* of the feature space. Rather than treating clutering as a competing paradigm to classification, I leverage it as a hypothesis generative tool for revealing interesting data patterns, such as high subpopulations. 

#v(1em)
#paragraph[Step 1: Clustering on Feature Space]

#v(.1em)
#figure(
  table(
    columns: (0.8fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1.2fr),
    align: (left, right, right, right, right, right, right, right),
    stroke: (x: none, y: none),
    table.hline(stroke: 0.8pt + black),
    inset: (x: 0.5pt, y: 3pt),
    
    [*Metrics*], [*Kmeans*], [*Spectral*], [*Ward*], [*GMM*],  [*DBSCAN*], [*HDBSCAN*], [*Agglomerative*],
    table.hline(stroke: 0.5pt + black),
    [AMI], [0.0202], [*0.2968*], [#underline[0.2366]], [0.0186], [0.0162], [0.1489], [0.2345],

    [ARI], [0.0276], [*0.3843*], [#underline[0.3122]], [0.0253], [0.0087], [0.0487], [0.2972],

    [ACC], [0.5878], [*0.8108*], [#underline[0.7804]], [0.5845], [0.5338], [0.2432], [0.7736],
    table.hline(stroke: 0.8pt + black),
  ),
)<cluster-comparison-none-table>
To satisfy project requirements, seven clustering algorithms are applied to $l_2$-normalized features. Three different metrics were used: Adjusted Mutual Information (AMI), Adjusted Rand Index (ARI), and Accuracy with post-hoc label matching.
As shown in @cluster-comparison-none and the table above, most model performs no better, or even worse, than random guessing (Acc $approx$ 0.5). The low ARI and AMI scores ($<=0.1$) scores substantiate this further, indicating that the clustering results is indistinguishable from random noise. The best performing algorithms (Spectral and WARD) only achieves Acc $approx$ 0.8.

Such underwhelming performance is expected. Raw features rarely exhibits directly separable structures algined with supervised labels without feature engineering. However, this negative results motivates the next experiment: can learned representations capture more meaningful geometry? 

#v(1em)
#paragraph[Step 2: Clustering on Tabular Foundation Model Embeddings]

#figure(
  image("figures/cluster/cluster_comparison_tabpfn.png", width: 100%),
  caption: [
    Comparison of the same seven clustering algorithms on TabPFN embeddings. Remarkably, all clustering algorithm except DBSCAN achieved near-perfect alginment with the output labels.
  ]
)<cluster-comparison-tabpfn>
Tabular Foundation Models like TabPFN often encode rich semantic structure in their embeddings space. Motivated by this, I extracted embeddings $bold(upright(z))_i in RR^d'$ from TabPFN's penultimate layer and apply the same procedure as in Step 1. The only variable here is swapping features for embeddings.


#v(.1em)
#figure(
  table(
    columns: (0.8fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1fr, 1.2fr),
    align: (left, right, right, right, right, right, right, right),
    stroke: (x: none, y: none),
    table.hline(stroke: 0.8pt + black),
    inset: (x: 0.5pt, y: 3pt),
    
    [*Metrics*], [*Kmeans*], [*Spectral*], [*Ward*], [*GMM*],  [*DBSCAN*], [*HDBSCAN*], [*Agglomerative*],
    table.hline(stroke: 0.5pt + black),
    [AMI], [0.6473], [*1.0000*], [*1.0000*], [0.7630], [0.8130], [#underline[0.9853]], [*1.0000*],

    [ARI], [0.7471], [*1.0000*], [*1.0000*], [0.8315], [0.8118], [#underline[0.9927]], [*1.0000*],

    [ACC], [0.9324], [*1.0000*], [*1.0000*], [0.9561], [0.4358], [#underline[0.9966]], [*1.0000*],
    table.hline(stroke: 0.8pt + black),
  ),
)<cluster-comparison-tabpfn-table>
#v(1em)

While the results are impressive, they suffer from a methodological confound: TabPFN is trained with supervision, meaning its embedding are explicitly optimized to be separable for the two classes. Therefore, if we were to treat this result as evidence that "TabPFN enables unsupervised discovery", it would be blatant misinterpretation and circular reasoning. Instead, I view this experiment as *validation of embedding quality* that justifies the next stage (Step 3).

#v(1em)
#paragraph[Step 3: Embedding Residualization]

#figure(
  image("figures/cluster/color_verification_tabpfn_intra_y0.png"),
  caption: [
    Color verification of TabPFN embeddings on the $y=0$ subset. The manifold forms a horseshoe shape. Without residualization, the data points on the manifold are ordered by the model's prediction confidence $hat(P) (y=0)$. The smooth color gradient in the rightmost column suggests primary spatial variation along the horseshoe is directly correlated to output probability.
    #v(1em)
  ]
)<color_Verification_none>

I start by partitioning embeddings by their class: $bold(upright(Z_0)) = {bold(upright(z))_i : y_i = 0}$ and $bold(upright(Z_1)) = {bold(upright(z))_i : y_i = 1}$. Then, I visualize the embeddings via dimensionality reduction (i.e., PCA, UMAP, t-SNE) and coloring points by feature values and TabPFN confidence level. As shown in @color_Verification_none, broad color gradient across the horseshoe-shaped manifold can be seen, but discrete structures are lacking. 

To address this, the calibrated probabilities $hat(p)_"ens" (bold(upright(x))) in [0, 1]^n$ are first converted into logits. The logits for sample $x_i$ is denoted as:

$
  "logits"(hat(p)_"ens")_i = log(hat(p)_"ens" (x_i)) - log(1 - hat(p)_"ens" (x_i))
$<logits-equation>

Then, using Radial Basis Function (RBF) kernel regression, regress the embedding against logits: 
$
  bold(upright(z))_i^|| = arg min_(bold(upright(f)) in cal(H)_K)
  sum_(j=1)^n lr(norm(
    bold(upright(z))_i - bold(upright(f))("logits"(hat(p)_"ens")_i)
  ), size: #1.2em)^2 
  + lambda lr(norm(bold(upright(f))), size: #1.2em)^2_cal(H)_K
$
where $cal(H)_K$ denotes the Reproducing Kernel Hilbert Space (RKHS) for all possible $bold(upright(f))$. Then, the residualzed embedding is produced by: 
$
  bold(upright(z))_i^tack.t = bold(upright(z))_i - bold(upright(hat(z)))_i
$

By removing the variance explained by calibrated risk, $bold(upright(z))_i^tack.t$ should now capture *risk-independent structer*. In other words, patterns orthogonal to the main predictive signal. Visualizing $bold(upright(z))_i^tack.t$ on both the $y = 0$ and $y = 1$ subset immediately reveal discrete clustering of feature values. Qualitatively, this suggests existence of subpopulation with similar risk but different profiles. 



#figure(
  [
    #image("figures/cluster/color_verification_tabpfn_ortho_intra_y0.png")
    #image("figures/cluster/color_verification_tabpfn_ortho_intra_y1.png")
  ],
  caption: [
    Color verification of residualized TabPFN embeddings $bold(upright(z))_i^tack.t$ on the $y=0$ *(top)* and $y=1$ *(bottom)* subsets. Discrete clustering of feature values highlighted wiwth #text(fill: rgb("#8b328e"))[*purple bounding boxes*]. For example, for the `fbs` feature of $y = 0$ subset (2nd purple box of top image), both UMAP and t-SNE reveals a small island of low fasting blood sugar ($<=120$ mg/dL) within a large island of high fasting blood sugar ($>120$ mg/dL), a pattern not immediately apparant in @color_Verification_none.
    #v(1em)
  ]
)





== Pattern Discovery<pattern-discovery>

#v(1em)
#paragraph[Step 1: Target Engineering]

This final stage aims to mine and synthesize insights from calibrated risk proxy and subgroup structure into actionable and interpretable rules. Using the calibrated risk proxy $hat(p)_"ens" (bold(upright(x))) in [0, 1]^n$ obtained from #sref(<calibrated-classification>), it is possible to mine patterns beyond binary labels. Here, two approaches are explored, logits and strats. 

Specifically, calibrated probabilities {$hat(p)_"ens" (x_i)}_(i=1)^n$ are first converted into logits $cal(l)_i = "logit"(hat(p)_"ens")_i$ via equation @logits-equation. Then, discretize $cal(l)_i$ via quantile-based binning: 

$
  R_i = cases(
    "low"       &space "if" cal(l)_i < q_0.25,
    "mid-low"   &space "if" q_0.25 <= cal(l)_i < q_0.5,
    "mid-high"  &space "if" q_0.5 <= cal(l)_i < q_0.75,
    "high"      &space "if" cal(l)_i >= q_0.75,
  )
$

where $q_alpha$ denotes the $alpha$-quantile of ${cal(l)_i}$. Thus, creating a four-level target richer than binary labels.

#v(1em)
#paragraph[Step 2: Conventional vs. Proposed Approach]

A common mistake in course projects is to treats pattern mining as a standalone stage. I challenge this convention and ask the question: _Does propagating calibrated risk information and manifold structure insights yileds rules that are more semantically meaningful_? 

#paragraph[_Conventional Approach_] A typical workflow that treats pattern mining as standalone task (as criticized in #sref(<introduction>)) uses a binary label and possibly transformed feature space. In this setting, rules are optimized w.r.t. a single Bernoulli draw per patient. This is statistically fragile. Formally, let latent risk be $r(x) = P(Y=1|X=x)$  and observed labels be $y_i tilde "Bernoulli"(r(x_i))$. Then, minin on $y$ would be *inherently high-variance*. Let's model empirical event rate and its sampling variance:

$
  hat(r)_S^((y)) &= 1/(|S|) sum_(i in S) y_i\
  "Var"(hat(r)_S^((y)) |x_(1:n)) &= 1/(|S|^2) sum_(i in S) r_i (1-r_i) <= 1/(4|S|)
$

This variance is irreducible even with prefect knowledge of $x_i$. Because, fundamentally, each $y_i$ remains a stochastic Bernoulli draw. For small subgroups (exactly what we see interseting in #sref(<manifold-clustering>)), the bound $1 slash (4|S|)$ grows, making it less stable. Furthermore, when $r(x_i) approx 0.5$ and $|S|$ is small (i.e., precisely where many concerned "interesting rules" fall into), variance is high. In simpler terms, conventional mining on binary labels tend to discover sampling noise around the decision boundary.

#paragraph[_Proposed Approach_] My approach leverages insights gained from Stage 1 (#sref(<calibrated-classification>)) and Stage 2 (#sref(<manifold-clustering>)). Specifically, risk proxy from Stage 1 provides supervision, and residual manifold provides hypothesis space. In this setting, the target becomes ordinal strats (${R_i}$). 

Stage 1 of my pipeline produces out-of-fold claibrated probabilities $hat(p)_i approx r_i$. Assuming good calibration, this can be reasonably modelled as: 

$
  hat(p)_i = r_i + epsilon.alt_i, space EE[epsilon.alt_i|x_(1:n)]  = 0
$

The subgroup's mean prediced risk can also be modelled as: 
$
  hat(r)_S^((p)) = 1/(|S|) sum_(i in S) hat(p)_i
$

Then, its variance is determined by the estimation error, no longer irreducible Bernoulli noise: 
$
  "Var"(hat(r)_S^((p)) |x_(1:n)) = 1/(|S|^2) sum_(i in S) "Var"(epsilon.alt_i|x_(1:n))
$

Empirically, in Stage 1, the perturbation *the perturbation stability test is exactly a sanity check* to ensure that $epsilon.alt_i$ is small and stable. Under this premise, ideally we get $"Var"(epsilon.alt_i)<< r_i (1-r_i)$. Then, mining on $hat(p)_i$ would yield statistcally better subgroup scoring than mining on $y_i$. 

Stage 2, on the other hand, produces a residual manifold $bold(upright(z))_i^tack.t$ that captures risk-independent structure. Qualtiatively, we see data manifold contains interesting subgroup patterns. This prompts exploration of subgroup discovery algorithms that may otherwise be missed by conventional pipeline.

#pagebreak()
#paragraph[Quantitative Results]
#v(1em)

#figure(
  image("figures/mining/llm_blind_eval_by_model_barchart.png")
)
#figure(
  table(
    columns: (1.1fr,) + (1fr,1.33fr,1fr) * 3,
    align: center,
    stroke: none,
    inset: (x: 0.0pt, y: 3pt),
    toprule,
    table.header(
      table.cell(colspan: 1, align: center)[$n=3$], 
      table.cell(colspan: 3, align:center)[Association Rule Mining], 
      table.cell(colspan: 3, align: center)[RuleFit],
      table.cell(colspan: 3, align: center)[Subgroup Discovery],
      table.hline(start: 1, end: 10, stroke: (thickness: 0.05em)),

      [Target], 
      table.vline(stroke: 0.5pt), [Gemini3], [GPT5.2], [o4-mini], 
      table.vline(stroke: 0.5pt), [Gemini3], [GPT5.2], [o4-mini], 
      table.vline(stroke: 0.5pt), [Gemini3], [GPT5.2], [o4-mini], 
      table.hline(stroke: (thickness: 0.05em))
    ),
    [*Binary*],
    [*8.7$plus.minus$0.6*], [7.5$plus.minus$0.0], [6.0$plus.minus$0.0],
    [0.7$plus.minus$0.6], [3.3$plus.minus$0.3], [1.7$plus.minus$0.6],
    [7.7$plus.minus$0.6], [8.0$plus.minus$0.5], [*7.7$plus.minus$1.5*],
    [*Binned*],
    [*8.7$plus.minus$0.6*], [*8.0$plus.minus$0.9*], [*8.3$plus.minus$0.6*],
    [*5.3$plus.minus$0.6*], [*6.3$plus.minus$0.3*], [*4.3$plus.minus$0.6*],
    [*9.7$plus.minus$0.6*], [*8.7$plus.minus$0.3*], [7.3$plus.minus$0.6],
    botrule
  ),
) <llm-as-judge-table>

@llm-as-judge-table summarizes _LLM-as-Judge_ socres (mean $plus.minus$ std, higher better)  for three baseliens, ARM, RuleFit, and Subgroup Discovery. Two targets are compared: Binary and binned (risk stratas). Results indicate that binned targets improves (or at least preserves) rule quality across baselines, with most dramatic gain on RuleFit (from useless to interpretable). To improve judge robustness, three different SOTA LLM models are used. Namely, Gemini 3 Pro, GPT5.2, and o4-mini. While their absolute harshness differs, the general *binned > binary* consensus holds.

These evaluations target _semantic validity_ not just statistical association. The question we are trying to answer is does method X better maximizes internal interestingness / lift score, but rather: _Are the mined patterns more clinically meaningful?_. This is a question that is not possible with conventional frequency-based statistics, as it requires prior knowledges. Hence, LLM are the most suitable option.

#v(1em)
#paragraph[Selected Qualitative Results]


#set math.equation(numbering: none)
$
  &("cp" = 3) and ("ca" in (-0.001, 1.0]) and ("thal" = 3.0) and ("slp" = 1) => "low"\
  &("exng" = 0) and ("ca" in (-0.001, 1.0]) and ("sex" = 0) and ("slp" = 1) => "low"\
  &("cp" = 4) and ("thal" = 7.0) => "high"\
  &("exng" = 0) and ("sex" = 1) => "mid-low"
$

The first rule, for example, exemplifies a *false alarm filter*. The rule combines non-anginal pain, with normal thallium scan, and unsloping ST segment. This showcases noise-filtering capability. Chest pain is usually the most effective predictor, but can also be noisy (false positives). The fact that, the algorithm is able to overcome this strong correlation when combined with normal ST slope and clear vessels indicates its pattern discovery capacity.

The third rule is also particularly interesting. Usually, `cp=4` (Asymptomatic/No Pain) is universally weighted as negative correlation with heart disease. However, here the model captures a known cardiology condition known as _Silent Ischemia_, which says, when "No pain" is paired with "Thallium defect", lack of pain is not sign of health.

#v(1em)
#paragraph[Clinical Pattern Analysis]

The discovered rules warrant deeper clinical interpretation. Consider the rule $("cp"=4) and ("thall"=7.0) => "high"$, which maps to the highest risk stratum. This pattern captures the phenomenon of _silent ischemia_, defined as myocardial ischemia occurring without typical anginal symptoms @cohn_silent_2003. Patients with asymptomatic presentation ($"cp"=4$) who simultaneously exhibit reversible thallium perfusion defects ($"thall"=7.0$) represent a particularly dangerous subpopulation. The reversible defect indicates viable but hypoperfused myocardium, while the absence of warning chest pain removes the protective signal that typically prompts patients to seek medical attention @gottlieb_silent_1986. Epidemiological studies estimate that silent ischemia affects 2 to 4 percent of asymptomatic middle-aged individuals and carries mortality risk comparable to symptomatic coronary artery disease @cohn_silent_1988. The fact that our calibration-guided mining framework surfaces this clinically significant pattern, rather than burying it among spurious correlations, validates the utility of risk-stratified target engineering.

The protective rules demonstrate equally important clinical validity. The pattern $("cp"=3) and ("caa" in (-0.001, 1.0]) and ("thall"=3.0) and ("slp"=1) => "low"$ combines four complementary reassurance signals: non-anginal chest pain ($"cp"=3$), minimal coronary calcification ($"caa" <= 1$), normal thallium perfusion ($"thall"=3.0$), and upsloping ST segment ($"slp"=1$). Each component independently reduces cardiac risk probability, and their conjunction defines a low-risk phenotype where chest discomfort likely originates from non-cardiac sources @gibbons_acc_2002. The upsloping ST segment morphology during exercise is particularly informative; unlike flat or downsloping patterns that suggest subendocardial ischemia, upsloping deflection typically reflects normal physiological response to increased heart rate @lim_st_2016. Similarly, the rule $("caa"=0.0) and ("exng"=0) and ("thall"=3.0) => "low"$ identifies patients with zero fluoroscopy-visible vessel calcification, no exercise-induced angina, and normal nuclear imaging. This triple-negative profile would be categorized by clinical guidelines as very low pretest probability for obstructive coronary disease @greenland_cac_2018.

Sex-specific stratification emerges prominently in the high-risk rules, with $("sex"=1)$ appearing in patterns such as $("cp"=4) and ("sex"=1) => "mid-high"$. This aligns with established cardiovascular epidemiology showing that men under age 70 experience roughly twice the incidence of coronary events compared to age-matched women @mosca_sex_2011. The male sex variable, when combined with asymptomatic presentation, amplifies risk because men are more likely to experience silent ischemia and sudden cardiac death without prodromal symptoms. Our framework correctly captures this interaction rather than treating sex as an independent linear predictor.

Finally, the discovery of $("slp"=2)$ (flat ST segment) in multiple high-risk rules corroborates exercise testing literature. A flat or horizontal ST depression during stress testing exhibits higher specificity for ischemia than upsloping depression, as flat morphology reflects more severe subendocardial supply-demand mismatch @lim_st_2016. The conjunction $("cp"=4) and ("slp"=2) and ("thall"=7.0) => "high"$ thus represents a particularly ominous triad: silent presentation, ischemic ST morphology, and perfusion defect. These three independent indicators converge on the same high-risk stratum.

#v(1em)
#paragraph[Summary]

Taken together, these patterns demonstrate that calibration-guided target engineering enables discovery of rules that align with cardiology domain knowledge. The conventional binary-label approach, by contrast, conflates patients with varying degrees of underlying risk, resulting in rules that optimize statistical lift but lack clinical coherence. The quantitative LLM evaluation (@llm-as-judge-table) confirms this qualitative observation: binned risk strata consistently outperform binary targets across all three mining algorithms, with the most dramatic improvement observed in RuleFit (binary-target rules received near-zero clinical validity rating while binned-target rules achieved good ratings).


#pagebreak()
= Appendix<appendix>

== Data Quality and Preprocessing

#figure(
  [
    #image("figures/eda/eda_categorical_before.png")
    #image("figures/eda/eda_numeric_before.png")
  ],
  caption: [
    Feature distributions of the original Kaggle dataset before correction. The conditional distributions suggest label inversion: features clinically associated with disease risk (high ST depression, low max heart rate, presence of exercise angina) paradoxically concentrate in the `output=0` group.
  ]
)<eda-before>

Initial EDA of the #link("https://www.kaggle.com/datasets/sonialikhan/heart-attack-analysis-and-prediction-dataset")[Kaggle Heart Attack Analysis dataset] revealed systematic inconsistencies between the stated label semantics and clinical expectations. The Kaggle documentation asserts that `output=0` indicates absence of heart attack risk while `output=1` indicates presence. However, examination of feature distributions exposes four contradictions that collectively points to label inversion.

First, the ST depression feature (`oldpeak`) exhibits mean values of 1.58 for `output=0` versus 0.58 for `output=1`. Since elevated ST depression during exercise indicates myocardial ischemia, the higher mean in the purportedly healthy group contradicts established cardiology. Second, maximum heart rate achieved (`thalachh`) averages 139.1 bpm for `output=0` compared to 158.5 bpm for `output=1`. Lower exercise capacity typically signals compromised cardiac function, yet the supposedly diseased group demonstrates superior performance. Third, exercise-induced angina (`exng`) occurs in 55% of `output=0` patients but only 14% of `output=1` patients. The presence of exertional chest pain is a hallmark indicator of coronary insufficiency, making its concentration in the "healthy" group implausible. Fourth, the number of major vessels colored by fluoroscopy (`caa`) shows predominantly non-zero values for `output=0` and predominantly zero values for `output=1`. Greater coronary calcification visible on fluoroscopy correlates with atherosclerotic burden, contradicting label semantics.

Cross-referencing with the original #link("https://archive.ics.uci.edu/dataset/45/heart+disease")[UCI Cleveland Heart Disease dataset] confirmed that the Kaggle version had inverted labels. Additionally, the Kaggle data contained invalid values such as `thall=0`, which has no clinical meaning given that the thallium scan feature encodes only three categories (normal, fixed defect, reversible defect). After correcting label polarity and removing invalid categorical entries, the cleaned dataset exhibits distributions consistent with cardiology domain knowledge (@eda-after).

#figure(
  [
    #image("figures/eda/eda_categorical_after.png")
    #image("figures/eda/eda_numeric_after.png")
  ],
  caption: [
    Feature distributions after label correction and data cleaning. Risk-associated features (ST depression, exercise angina, etc.) now concentrate in `output=1`, and protective features (high max heart rate, absence of angina) concentrate in `output=0`, aligning with clinical expectations.
  ]
)<eda-after>

== LLM-as-Judge Evaluation Protocol

To assess the clinical validity of mined rules beyond statistical metrics, we employed a blind evaluation protocol using frontier large language models as domain expert proxies. Three state-of-the-art models were used: _Gemini-3-Pro-Preview_, _GPT-5.2_, and _o4-mini_. Each model evaluated rules from all three mining algorithms (Association Rule Mining, RuleFit, Subgroup Discovery) across two target definitions (binary labels and binned risk strata).

The evaluation followed a randomized blind design. For each algorithm, rules mined under different target definitions were assigned anonymous group labels (A, B, C) with random permutation seeded by algorithm name, model identifier, and round index. This prevents models from inferring target semantics from label ordering. Each model completed three independent evaluation rounds per algorithm, yielding nine judgments per algorithm-target pair across all models.

Models received a system prompt establishing cardiology domain expertise, a feature glossary with clinical interpretations, and a scoring rubric mapping 0-10 scores to clinical plausibility levels. The prompt explicitly instructs models to evaluate rules based on alignment with established cardiovascular risk factors rather than statistical properties. Temperature was set to 1.0 to encourage response diversity across rounds, and maximum output tokens (including reasoning) was set to 32,000.

#v(1em)
#paragraph[LLM-as-Judge System Prompt]

The following system prompt was used verbatim for all LLM-as-Judge evaluations:

#figure(
  rect(width: 100%, inset: 8pt, stroke: 0.5pt)[
    #set text(size: 7.5pt)
    #set par(justify: false)
    ```
    You are a cardiology-domain medical reviewer evaluating whether
    mined rules align with well-known heart disease risk factors and
    clinical intuition.

    You will receive multiple anonymous groups (e.g., Group A, Group B,
    Group C). Each group contains a list of rules mined from the SAME
    dataset, but using different (hidden) target definitions. The group
    labels and their order are arbitrary and randomly permuted each time.
    Do NOT assume any meaning from the letters or ordering.

    Your task:
    - Score EACH group on a 0-10 scale for how medically plausible and
      clinically meaningful its rules are.
    - Higher score = rules are more consistent with cardiology knowledge
      and have fewer nonsensical/contradictory patterns.
    - Evaluate each group independently based only on its content.
    - Multi-class labels and continuous outputs are not necessarily
      better, unless you are confident they provide superior medical
      insights.

    Dataset feature glossary (Heart Disease style dataset):
    - age: age in years (higher often increases risk)
    - sex: 0=female, 1=male (male often higher risk)
    - cp: chest pain type (1=typical angina, 2=atypical angina,
      3=non-anginal pain, 4=asymptomatic; cp=4 often higher risk)
    - trtbps: resting blood pressure (higher often higher risk)
    - chol: serum cholesterol in mg/dl (higher can increase risk,
      but noisy)
    - fbs: fasting blood sugar > 120 mg/dl (1=true; diabetes risk factor)
    - restecg: resting ECG (0=normal, 1=ST-T abnormality,
      2=LV hypertrophy; abnormal values can increase risk)
    - thalachh: maximum heart rate achieved (lower can indicate worse
      functional capacity; interpretation depends on context/age)
    - exng: exercise-induced angina (1=yes; increases risk)
    - oldpeak: ST depression induced by exercise vs rest (higher
      increases risk)
    - slp: slope of peak exercise ST segment (1=upsloping, 2=flat,
      3=downsloping; flat/downsloping increases risk)
    - caa: number of major vessels (0-3) colored by fluoroscopy (higher
      increases risk)
    - thall: thalassemia test (3=normal, 6=fixed defect, 7=reversible
      defect; defects increase risk)

    How to interpret rule formats:
    - Rules are patterns over the features above. They may use equality
      (e.g., "cp==4") and/or ranges (e.g., "age=(54, 62]").
    - Focus on whether the patterns align with established risk factors
      and clinical intuition.

    Scoring rubric:
    - 9-10: Strongly aligned with established risk factors, coherent,
      few/no contradictions, not dominated by spurious patterns.
    - 6-8: Mostly plausible but some weak/noisy or questionable rules.
    - 3-5: Mixed plausibility, many unclear/uninformative rules, or
      several questionable patterns.
    - 0-2: Largely nonsensical, contradicts medical intuition, or
      essentially empty/unusable.

    Output format (STRICT):
    - Return ONLY valid JSON (no markdown, no code fences, no extra text).
    - Schema:
      {
        "scores": { "A": <number 0-10>, "B": <number 0-10>, ... },
        "brief_reason": { "A": "<short>", "B": "<short>", ... }
      }
    - Adapt keys to match the number of groups presented.
    - Scores must be numeric and within [0, 10].
    ```
  ],
  caption: [
    Complete system prompt used for LLM-as-Judge evaluations. 
  ]
)

#pagebreak()
= References