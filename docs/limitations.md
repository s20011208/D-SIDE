# Scope and limitations

D-SIDE is a reproducible research prototype with both favorable and unfavorable simulation outcomes. The released code demonstrates a concrete source-selection and estimation workflow; it does not establish universal improvement, mechanism identification, or a negative-transfer guarantee.

## Source information and identifiability

Each source supplies only one noisy trajectory in the released benchmark. Equal numbers of observations do not imply equal information: position coverage, repeated-state availability, turning points, and matched velocity differences all matter. Disjoint or weakly overlapping supports cannot reveal whether source mechanisms agree where they were never observed together.

The working model is $X_g''+a_gX_g'=f_g(X_g)$. Restricting $f_g$ to position is substantive. If the right-hand side were allowed an arbitrary velocity term, that term could absorb damping changes. Likewise, an additional free linear position term in the operator could be absorbed by $f_g$. The present normalized model omits that extra operator term. Penalization selects a fitted solution; it does not prove unique recovery of physical mechanisms.

## Scalar statistics and density assumptions

Tikhonov inversion leaves filter bias. Estimated initial velocity, polynomial approximation, noisy matching locations, and numerical quadrature add further errors. A ratio can amplify these errors even after denominator filtering.

Repeated state levels and source pairs share curves. Hundreds of derived rows are not hundreds of independent trajectories. The concentration count $1/\sum p_r^2$ describes weight balance only. The density reliability weight is a heuristic, not a calibrated uncertainty measure, and cannot detect every systematically biased batch.

The one-truncated-normal-plus-uniform density is a working model. Related sources need not produce a narrow Gaussian peak; heterogeneous sources need not produce a diffuse uniform background. Peak responsibility is not a posterior probability of shared dynamics. The majority of related sources also does not imply a dominant majority of related source pairs: six of eight sources provide only 15 of 28 pure-related pairs before reliability reweighting.

The $\arctan$ transform uses fixed simulation units. Its shape, and hence the mixture fit, is sensitive to rescaling unless a reference damping scale is introduced consistently. This release does not claim invariance to arbitrary changes of time units.

## Clustering and borrowing

The default rule finds one main shared block of at least three sources, with hard inclusion/exclusion. It can miss useful sources or admit harmful ones. A partition's other blocks are split into independent models rather than fitted as additional shared mechanisms. There is no post-fit validation gate to reverse an unfavorable pooling decision.

Exact correlation clustering is feasible for eight sources but enumerates set partitions; the current implementation restricts the source count to at most nine. This is not a scalable large-source clustering implementation.

## Joint estimation

Inside the selected collection, the nonlinear function is exactly shared, while damping remains source-specific with a shrinkage penalty. The idealized generating model matches that within-collection function assumption. More realistic small functional deviations would require separately specified and tested estimation behavior.

The finite Nyström approximation has a rank cap of 16. Independently fitted sources and shared collections use different state dictionaries, scales, and sometimes feature counts. Their comparison does not isolate information borrowing from dictionary changes or operator shrinkage. A rank growth rule in code is not an asymptotic nonparametric consistency result.

The conditional ridge problem is convex and uniquely solvable under a positive penalty, but the outer operator optimization is nonconvex. Two starts do not certify a global optimum. Positive damping bounds do not by themselves prove stability of the fitted nonlinear system.

Correct source selection does not guarantee lower prediction error for every source. In the adapted-initial-condition fresh experiment, the true-group reference still had 45/160 source instances with prediction error more than 1% above independent fitting.

## Evaluation and generalization

The favorable original-initial-condition example is based on previously seen, retrospective seeds. The adapted design's fresh twenty-seed evaluation did not improve average all-source prediction over independent fitting. Both results are retained in [the experiment record](experiments.md).

Tests give every method exact initial position and velocity. They assess new trajectories within known sources, not prediction with uncertain initial states or adaptation to unseen sources. Test initial states cover a wider pattern than the controlled training starts, creating possible state-coverage and extrapolation difficulty.

The model is deterministic, scalar, autonomous, second order, and driven only by position. No real-data application, calibrated interval coverage, higher-order extension, general multivariate result, root-$n$ function-estimation claim, or advantage over published state-of-the-art algorithms is established by this release.

## What the results support

The software implements distributional source selection and conditional nonparametric dynamics estimation in a transparent, replayable way. Selective borrowing can improve trajectory prediction in a controlled setting with limited trajectories. The current evidence also shows where that approach fails, especially when derived-statistic quality is not captured by count and when joint estimation harms individual sources despite correct labels.
