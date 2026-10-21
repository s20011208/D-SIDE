# References and attribution

D-SIDE names this repository's distributional source-selection workflow. Its numerical building blocks have established precedents. The repository does not claim to introduce Green integral identities, Tikhonov inversion, kernel ridge estimation, Nyström approximation, or correlation clustering.

## Direct methodological background

**Jianbin Tan, Guoyu Zhang, Xueqin Wang, Hui Huang, and Fang Yao (2024).** *Green's matching: an efficient approach to parameter estimation in complex dynamic systems.* Journal of the Royal Statistical Society: Series B. [Published article](https://doi.org/10.1093/jrsssb/qkae031) · [Author preprint](https://arxiv.org/abs/2403.14531).

This is direct background for estimation through Green integral representations. The current release adds a particular scalar distributional source-selection procedure and uses a nonparametric function approximation. It does not inherit finite-dimensional parameter-estimation guarantees for its estimated functions, source decisions, or nonconvex outer optimization.

The project also consulted a user-supplied manuscript, *Semiparametric Differential Modeling for Trajectory Data*. A public publication identifier was not verified, and its separately referenced supplement was not supplied. That manuscript is not redistributed here. Existing unknown-operator and nonparametric integral-matching ideas in background work are not asserted as D-SIDE's novel contributions.

## Pairwise-evidence clustering

**Nikhil Bansal, Avrim Blum, and Shuchi Chawla (2004).** *Correlation Clustering.* Machine Learning 56:89–113. [Published article](https://doi.org/10.1023/B:MACH.0000033116.57574.95).

Correlation clustering groups objects to agree with positive and negative pairwise evidence, without fixing the number of groups. D-SIDE uses a weighted form of this established objective and exact enumeration for its small source count. The source-evidence construction and single-block extraction remain additional modeling choices, not consequences of a clustering recovery theorem.

## Related heterogeneous-dynamics methods

**Matthieu Kirchmeyer, Yuan Yin, Jeremie Dona, Nicolas Baskiotis, Alain Rakotomamonjy, and Patrick Gallinari (2022).** *Generalizing to New Physical Systems via Context-Informed Dynamics Model.* ICML, PMLR 162:11283–11301. [Official proceedings](https://proceedings.mlr.press/v162/kirchmeyer22a.html).

CoDA learns contextual adaptation across different dynamical environments. It motivates the broader heterogeneous-dynamics problem. No claim of reproducing or outperforming the complete original CoDA algorithm is made by the released D-SIDE benchmark.

**Julien Martinelli, Maksim Sinelnikov, Harri Lähdesmäki, Quentin Clairon, and Mélanie Prague (2026).** *Bayesian Nonparametric Mixed-Effect ODEs with Gaussian Processes.* [arXiv:2605.13088](https://arxiv.org/abs/2605.13088).

MEGPODE provides a population/shared vector field and subject-specific deviations using Gaussian-process models. Shared and individual nonparametric dynamics are therefore not, by themselves, a novelty claim for D-SIDE. The earlier project audit examined the first preprint version; the public landing page now also lists a revised version. The release makes no new comparison with that revision.

## Clustering alternatives examined during development

**Bruce Hajek, Yihong Wu, and Jiaming Xu (2016).** *Semidefinite Programs for Exact Recovery of a Hidden Community.* [COLT proceedings](https://proceedings.mlr.press/v49/hajek16.html).

**Christopher Aicher, Abigail Z. Jacobs, and Aaron Clauset.** *Learning latent block structure in weighted networks.* [Author preprint](https://arxiv.org/abs/1404.0431).

**Tiago P. Peixoto (2018).** *Nonparametric weighted stochastic block models.* [Author preprint](https://arxiv.org/abs/1708.01432).

These works informed an exploratory hidden-community/likelihood-score alternative. That alternative is not the released default. Dependent curve-derived statistics and fitted, potentially misspecified densities do not satisfy the independent-edge and known-distribution assumptions of many exact-recovery results. Their theorems are not guarantees for D-SIDE.

## Software dependencies

Numerical linear algebra, optimization, interpolation, and ODE integration use NumPy and SciPy; tables use pandas and figures use Matplotlib. Exact versions and development dependencies are declared in the repository environment files. Library licenses remain with their respective authors. No third-party paper PDFs or unrelated vendored method implementations are needed for the released default.

## Citing this repository

Cite the repository URL, https://github.com/s20011208/D-SIDE, and the exact Git commit used in an analysis. D-SIDE is presented as research software, not as a published journal article. No DOI, venue, institution, or author identity should be inferred from the GitHub account name alone.
