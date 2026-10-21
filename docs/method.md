# Method

D-SIDE (**Distributional Selective Integration for Dynamics Estimation**) is a two-stage research algorithm: select a collection of sources from distributions of scalar dynamical contrasts, then estimate a common nonlinear mechanism and separate operator coefficients within that collection.

The released default is the **density-only reliability weighting** variant. It uses `weight_density=True`, `weight_edges=False`, and a saturation constant of 21. It is a hard selection method, with at most one shared collection. It does not use source-specific reference centers, a two-dimensional damping statistic, or a validation-based fallback after pooling.

## 1. Model and scope

For source $g$, position observations at times $t_{gj}$ satisfy

$$
Y_{gj}=X_g(t_{gj})+\epsilon_{gj},\qquad
X_g''(t)+a_g X_g'(t)=f_g\{X_g(t)\}.
$$

Here $X_g$ is the latent trajectory, $a_g$ is a source-specific damping coefficient, and $f_g$ is an unknown smooth function. The second-derivative coefficient is fixed to one. The release examples have one training trajectory per source; trajectories have their own unknown training initial position and velocity. The reference problem has eight sources, of which six have identical $f_g$ and close $a_g$; the other two have different functions and coefficients.

This is **heterogeneity in the governing dynamics**. Changing the initial state alone can produce distinct trajectories under one equation, but the present model also allows the equation itself to differ between sources. The right-hand side depends on position alone; a general function of both position and velocity is outside this working model. The explicit $a_g X_g'$ term would not be separately identifiable if an unrestricted velocity-dependent right-hand side could absorb it.

Time and position use normalized simulation units. Damping has units of inverse time. The scalar transform below uses the fixed units of the released experiments; changing time units requires a corresponding reference scale.

## 2. Training-only trajectory preparation

The source selector fits each observed training trajectory by least squares in a degree-five Chebyshev basis. Its analytic derivatives provide the initial-state estimates and the velocity used for pairing. This fixed-degree polynomial is a trajectory smoother, **not** the model for $f_g$.

A separate cubic smoothing spline is used for the subsequent dynamics fit, with residual budget $n_g\sigma_\epsilon^2$. Both steps see only training observations. The release uses 61 uniform quadrature nodes over $[0,3]$. The declared measurement standard deviation is $\sigma_\epsilon=0.035$.

## 3. Integral coordinates and regularized inversion

Write $x_{g0}=X_g(0)$ and $v_{g0}=X_g'(0)$. Integrating the differential equation twice gives

$$
q_g(t)=\int_0^t(t-s)f_g\{X_g(s)\}\,ds-a_g m_g(t),
$$

where

$$
q_g(t)=X_g(t)-x_{g0}-t v_{g0},\qquad
m_g(t)=\int_0^t X_g(s)\,ds-tx_{g0}.
$$

On the quadrature grid, let $G$ discretize integration against $t-s$, let $V$ discretize one integration, and let $F_g$ contain $f_g$ evaluated along the smoothed trajectory. The trapezoidal approximation is

$$
\boldsymbol q_g\approx G\boldsymbol F_g-a_g\boldsymbol m_g.
$$

Using estimated initial states, define

$$
R_\lambda=(G^\top G+\lambda I)^{-1}G^\top,\qquad
\boldsymbol b_g=R_\lambda\boldsymbol q_g,\quad
\boldsymbol A_g=R_\lambda\boldsymbol m_g,
$$

with $\lambda=10^{-7}s_{\max}(G)^2$. The implementation evaluates this regularized inverse through an SVD.

The correct resulting relation is

$$
\boldsymbol b_g+a_g\boldsymbol A_g\approx R_\lambda G\boldsymbol F_g,
$$

**not** an exact equality to $\boldsymbol F_g$. The causal quadrature matrix has a zero first row and a zero last column; regularization does not remove its null space. Smoothing, estimated initial velocity, quadrature, and inverse filtering all contribute error.

## 4. Equal-state pairs and scalar statistics

Split each smoothed curve into monotone branches after trimming two grid points at each end. A branch must contain at least three distinct states. For each pair of branches with a shared position interval of width at least 0.01, place 21 equally spaced interior state levels. Linear interpolation supplies two times at each level:

$$
\widetilde X_g(t_p)=\widetilde X_h(t_q)=x^*.
$$

A pair $r=(p,q)$ supplies

$$
N_r=b_q-b_p,\qquad D_r=A_p-A_q,\qquad
T_r=\frac{N_r}{D_r},\qquad z_r=\arctan(T_r).
$$

Every statistic retains its source labels $(g,h)$, branch labels, matched state, and both times. The 21 levels are deterministic interpolated evaluations, not independent random samples. Same-source repeated-state pairs may be generated, but only **cross-source pairs** enter the released global distribution and clustering.

For intuition, write the approximation error at a point as $b_p+a_g A_p=f_g(x^*)+e_p$. Then

$$
T_r=a_g+\frac{(a_g-a_h)A_q}{D_r}
+\frac{f_h(x^*)-f_g(x^*)}{D_r}
+\frac{e_q-e_p}{D_r}.
$$

Thus a cross-source statistic is a **dynamical contrast**, not an unbiased estimate of a common damping coefficient. Even small source differences and approximation errors can be amplified by a weak denominator.

## 5. Local reliability filters and weights

Let $s_v$ be the median across sources of the root-mean-square smoothed velocity. A pair is retained only if both matched speeds exceed $0.05s_v$ in absolute value and

$$
|D_r|\ge\max\{0.1s_v,\;2\widehat{\operatorname{sd}}(D_r)\}.
$$

Noise sensitivities are propagated through the linear polynomial fit and regularized inverse. A first-order adjustment accounts for the movement of matched times: at a fixed matched state, $\delta t\approx-\delta\widetilde X/\widetilde X'$. The derivative of the transformed ratio is

$$
\delta z_r=\frac{D_r\,\delta N_r-N_r\,\delta D_r}{D_r^2+N_r^2}.
$$

This yields a local delta-method variance proxy $\widehat v_r$ for $z_r$. The preliminary row weight is

$$
w_r=\frac{\rho_r}{\widehat v_r+0.05^2},
$$

where $\rho_r$ is the reciprocal of the retained candidate count in its branch-pair block, before the additional precision filtering. Within source pair $(g,h)$, normalize these weights:

$$
p_{gh,r}=\frac{w_r}{\sum_{s\in\mathcal R_{gh}}w_s}.
$$

The delta variance does not include all smoothing, branch selection, and inverse-filter bias. It is used as a reliability heuristic, not as a calibrated confidence interval.

## 6. Reliability-weighted global distribution

For the retained row set $\mathcal R_{gh}$ of each observed source pair, define

$$
n_{gh}^{\mathrm{bal}}=\frac{1}{\sum_r p_{gh,r}^2},\qquad
c_{gh}=\frac{n_{gh}^{\mathrm{bal}}}{n_{gh}^{\mathrm{bal}}+21}.
$$

The quantity $n_{gh}^{\mathrm{bal}}$ measures concentration of the precision weights. It is **not** an effective number of independent observations. The constant 21 comes from the fixed state-level sampling rule and was not selected using test outcomes. Exact duplicate records are removed.

The global empirical distribution uses weights

$$
W_{gh,r}=\frac{c_{gh}p_{gh,r}}{\sum_{u<v}c_{uv}},\qquad
\widehat P=\sum_{g<h}\sum_{r\in\mathcal R_{gh}}W_{gh,r}\delta_{z_r}.
$$

A source pair with little usable evidence therefore contributes less total mass to the global density. The graph edges in step 8 are **not** multiplied by $c_{gh}$ in the default variant.

## 7. One peak plus background

On $\mathcal I=(-\pi/2,\pi/2)$, let $q_{\mu,s}$ be the truncated normal density with location $\mu$ and scale $s$, and let the background density be $1/\pi$. Fit

$$
p(z)=\eta q_{\mu,s}(z)+(1-\eta)/\pi
$$

by maximizing $\sum_r W_r\log p(z_r)$. The bounds are $\mu\in[-1.5,1.5]$, $s\in[0.04,0.7]$, and $\eta\in[0.05,0.95]$. Nine starts use equally spaced locations from $-1.2$ to $1.2$, initial scale 0.25, and initial mixing weight 0.6. Bounded L-BFGS-B supplies the best fitted objective.

For a statistic $z_r$, calculate its fitted peak responsibility

$$
\gamma_r=\frac{\widehat\eta q_{\widehat\mu,\widehat s}(z_r)}
{\widehat\eta q_{\widehat\mu,\widehat s}(z_r)+(1-\widehat\eta)/\pi}.
$$

This is a component responsibility under a working density model. It is **not** a calibrated probability that its two sources share a mechanism. Likewise, $\eta$ is a fraction of distributional mass, not a fraction of sources. The shaded central range in the figures is where $q_{\widehat\mu,\widehat s}(z)\ge1/\pi$; it is descriptive, not a confidence interval.

## 8. From pair evidence to one shared collection

For each observed source pair, use

$$
E_{gh}=\sum_{r\in\mathcal R_{gh}}p_{gh,r}\gamma_r-\widehat\eta.
$$

Positive evidence favors combining a pair relative to the global reference; negative evidence favors separating it. Missing comparisons contribute zero evidence. The graph is partitioned by

$$
\widehat{\mathcal P}=\arg\max_{\mathcal P}
\sum_{C\in\mathcal P}\sum_{g<h,\,g,h\in C}E_{gh}.
$$

This is a weighted correlation-clustering objective, an established clustering formulation. For eight sources the implementation enumerates all 4,140 partitions; tied objectives favor fewer merged pairs. The current solver accepts at most nine sources.

Among partition blocks with at least three members, retain the one with the largest internal evidence sum, breaking ties in favor of larger blocks. Call it $\widehat C$. Fit one shared function only for this block. Every source outside $\widehat C$ uses its independent model. If there is no eligible block, all sources are fitted independently.

This final single-block rule is a modeling choice consistent with the one-peak working model. Other partition blocks are not interpreted as additional recovered mechanisms. There is no validation-based decision to undo the selected pooling after dynamics fitting.

## 9. Nonparametric dynamics estimation

For the selected block $C$, represent one shared function by whitened Matérn-5/2 Nyström features:

$$
f_C(x)=\boldsymbol\psi_C(x)^\top\boldsymbol\beta.
$$

Position is standardized using the block's training median and standard deviation (scale floor 0.15). The kernel length is 0.9 in these standardized units. Landmarks are training-state quantiles from 1% to 99%; kernel eigenvalues are floored at $10^{-9}$ before whitening. The feature count is bounded by 16 and a sample-size rule; with the released one-trajectory budget it is typically 14 independently and 16 when pooling multiple sources. Thus dictionary coverage and numerical rank also change with pooling.

For any fixed $a_g>0$, define $k_{a_g}(t)=(1-e^{-a_g t})/a_g$. The integral solution is

$$
X_g(t)=x_{g0}+k_{a_g}(t)v_{g0}
+\int_0^t k_{a_g}(t-s)f_C\{X_g(s)\}\,ds.
$$

It comes from the companion matrix $\left(\begin{smallmatrix}0&1\\0&-a_g\end{smallmatrix}\right)$. Discretize it as

$$
\breve{\boldsymbol x}_g\approx H_g(a_g)\boldsymbol\zeta_g
+G_g(a_g)\Psi_g\boldsymbol\beta,
$$

where $\breve{\boldsymbol x}_g$ is the fixed training spline, $\boldsymbol\zeta_g=(x_{g0},v_{g0})^\top$, and row $\ell$ of $\Psi_g$ is $\boldsymbol\psi_C\{\breve X_g(\tau_\ell)\}^\top$.

With normalized trapezoidal time weights $W_t$, solve

$$
\min_{\boldsymbol a,\boldsymbol\beta,\{\boldsymbol\zeta_g\}}
\frac1{|C|}\sum_{g\in C}
\left\|W_t^{1/2}\left[\breve{\boldsymbol x}_g-H_g\boldsymbol\zeta_g-G_g\Psi_g\boldsymbol\beta\right]\right\|^2
+\lambda_f\|\boldsymbol\beta\|^2
+\frac{\lambda_a}{|C|}\sum_{g\in C}(a_g-\overline a_C)^2.
$$

The release uses $\lambda_f=0.0002$, $\lambda_a=0.0005$, and $a_g\in[0.02,2.5]$ for the revised damping benchmark. Training initial states are profiled by weighted QR projection. For fixed operators and fixed training splines, the coefficient problem is a strictly convex ridge problem. The outer operator problem is **nonconvex**; two L-BFGS-B starts use the 25% and 70% positions of the bound interval, with at most 100 iterations per start. A unique conditional ridge solution does not establish unique recovery of the true dynamics.

The selected sources share exactly one $f_C$; this release does not add within-block functional deviations. Damping remains separate, with the stated shrinkage penalty. Unselected sources reuse independently fitted functions and coefficients.

## 10. Free trajectory prediction

From independently supplied test initial position and velocity, integrate

$$
\widehat X_g'=\widehat v_g,\qquad
\widehat v_g'=\widehat f_g(\widehat X_g)-\widehat a_g\widehat v_g.
$$

DOP853 uses relative tolerance $2\times10^{-8}$ and absolute tolerance $2\times10^{-10}$. The right-hand-side input is the model's own predicted state. No future observed or true position is supplied during simulation.

## Interpretation and computational scope

The statistics compress dynamical comparisons; their dependence and approximation bias remain part of the problem. The global mixture, reliability rule, and single-core extraction are working choices, not proven mechanism-recovery procedures. The exact clustering solver scales combinatorially with source count. A dense 61-node quadrature calculation and small Nyström ridge system make the released eight-source examples practical, but this release does not benchmark large-source scaling or provide an asymptotic nonparametric theory.

See [experiments](experiments.md), [limitations](limitations.md), and [references](references.md).
