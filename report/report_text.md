# Extending PCA and LDA to Non-Linear Domains with POLCA Net

*AI Assignment #3 - CISC3024 Pattern Recognition*  
*November 2026*

**AI Assistance Disclosure.** This report was produced with the assistance of an AI coding agent. The entire assignment - the literature search that identified the algorithm, the design and implementation of the software, the execution of the experiments, the generation of the figures and tables, and the writing of this document - was carried out by the AI agent, in accordance with the explicit instruction of the assignment brief that the student is not permitted to write code by hand and that the search, the programming and the report writing are to be delegated to an AI tool. The human student's role was limited to reading the brief, approving the algorithm choice, executing the commands that the agent produced, and checking the outputs against the source paper. Every quantitative claim in this report is generated from artefacts on disk by the script that assembles this document, and the full prompt log is kept in AI_PROMPTS.md at the root of the repository.

---

## Table of Contents

1. How AI Was Used to Find the Algorithm
2. Algorithm Description
3. How AI Implemented the Algorithm
4. Experiment Settings and Results
5. What Was Learned from This AI Assignment
6. Source Code Web Link

---


AI Assistance Disclosure
This report was produced with the assistance of an AI coding agent. The entire assignment - the literature search that identified the algorithm, the design and implementation of the software, the execution of the experiments, the generation of the figures and tables, and the writing of this document - was carried out by the AI agent, in accordance with the explicit instruction of the assignment brief that the student is not permitted to write code by hand and that the search, the programming and the report writing are to be delegated to an AI tool. The human student's role was limited to reading the brief, approving the algorithm choice, executing the commands that the agent produced, and checking the outputs against the source paper. Every quantitative claim in this report is generated from artefacts on disk by the script that assembles this document, and the full prompt log is kept in AI_PROMPTS.md at the root of the repository.

1. How AI Was Used to Find the Algorithm
========================================
The brief for AI Assignment #3 asks for a recent algorithm in the family of principal component analysis (PCA) and linear discriminant analysis (LDA) that has a computer-vision or pattern-recognition application, and it asks explicitly that an AI tool perform the algorithm search, the programming and the writing of the report. The search was therefore not conducted by the student from memory or from a pre-existing reading list; it was delegated to the AI agent as a decomposed sub-task, and the decomposition itself was the first prompt that the agent received.
The agent was instructed to read the two requirement documents supplied with the assignment (the PDF brief and the report-requirements text file) before proposing any plan, so that the constraint set - the six mandated report sections, the prohibition on hand-written code, the expectation of a computer-vision application - was derived from the brief itself rather than assumed. The agent responded with an ordered execution plan that maps every step back to one of the six report sections, together with an explicit constraint checklist in which each item was tagged as a hard requirement, a report-structure requirement, or an assumption made on the student's behalf. The only substantive correction the student made at this stage concerned bookkeeping: the agent had proposed an additional notebook folder for exploration, which was rejected on the grounds that it would blur the provenance of the code. Everything the agent produced was therefore constrained to importable modules under src/ so that the whole pipeline can be re-executed from a clean checkout.

1.1 The literature-search sequence
The search stage consisted of a sequence of web-search and arXiv-query prompts. The queries below are the ones actually issued, in order, together with a condensed account of what each one returned. The agent was instructed to prefer work published in 2020 or later that extends PCA or LDA explicitly rather than in name only, and to report the arXiv identifier, the venue and the positioning sentence of every candidate so that the final selection could be audited afterwards.
Table 1. The literature-search queries actually issued by the AI agent and their outcome.

| # | Search query issued by the agent | What the search returned |
| --- | --- | --- |
| 1 | "recent PCA/LDA algorithm computer vision 2024 autoencoder orthogonal latent" | A first candidate pool (UMAP, t-SNE, Kernel PCA, Diffusion Maps) plus several recent deep dimensionality-reduction papers; only UMAP was published in the 2018-2024 window, but it does not extend PCA/LDA. |
| 2 | "deep PCA LDA extension 2024" | Papers that explicitly frame themselves as deep or non-linear extensions of PCA/LDA, with the authors' own positioning sentence; surfaced Deep Factorized Metric Learning and sigma-PCA. |
| 3 | "polca net principal orthogonal latent components analysis" | The target paper: arXiv:2410.07289, submitted 9 October 2024, by H. Martin, F. Perozo and M. Lopez. Confirmed as a preprint, with the experimental protocol and the claim of extending PCA and LDA to non-linear domains. |
| 4 | "sigma-PCA identifiable linear transformations arXiv" | sigma-PCA (arXiv:2311.13580, 2023): an identifiability theory paper about which linear maps are recoverable, without an autoencoder and without an image benchmark. |
| 5 | "deep factorized metric learning CVPR 2023" | Deep Factorized Metric Learning (CVPR 2023): factorises the embedding into a class subspace and a whitened residual; a metric-learning method rather than a PCA/LDA generalisation. |
| 6 | "piecewise-linear manifolds deep metric learning 2024" | Recent 2024 work on piecewise-linear and orthogonal structure in non-linear latent spaces (including arXiv:2403.14977), used as a cross-check that POLCA Net was not merely one of several equally plausible matches. |
The last query was deliberately adversarial: it asked the agent to look for competitors to POLCA Net rather than to confirm it. The reason for this posture is that a literature search performed by a language model is optimised for fluency rather than for recall, and is therefore prone to returning a single confident answer to every query. Asking for competitors forces the agent to enumerate alternatives, which makes the eventual selection a comparison rather than a rationalisation.

1.2 Candidate comparison
The agent was then asked to build a comparison table over the candidate set with the columns required by the brief, and to give an explicit rationale for the final pick together with an honest statement of whether any runner-up would have been defensible. The resulting table is reproduced below. The assessment column is the agent's judgement; it is presented as such, and it is one of the places where the report deliberately records a human correction: the first draft claimed that Deep Factorized Metric Learning contained an orthogonality constraint on the latent components, which is incorrect. The cell was changed to "No" after re-reading the paper, and the one-line description of the method was rewritten accordingly.
Table 2. Candidate algorithms considered, with the AI agent's assessment of each against the assignment wording.

| Algorithm | Year | Venue / arXiv | Non-linear? | Orthogonal latent? | Variance-sorted latent? | Supervised mode? | Fit with the assignment wording |
| --- | --- | --- | --- | --- | --- | --- | --- |
| UMAP | 2018 | arXiv 1802.03426 / JOSS 2021 | Yes | No | No | No | Weak - not framed as a PCA/LDA extension |
| t-SNE | 2008 | JMLR | Yes | No | No | No | Weak - visualisation-oriented, pre-deep-learning era |
| Kernel PCA | 2001 | NeurIPS | Yes | No (kernel Gram) | No | No | Partial - genuinely PCA, but not recent |
| Diffusion Maps | 2005 | PNAS | Yes | No | No | No | Weak - manifold learning, not PCA/LDA in form |
| sigma-PCA | 2023 | arXiv 2311.13580 | No (linear theory) | Partial | No | No | Partial - recent, but a theory paper with no image benchmark to reproduce |
| Deep Factorized Metric Learning | 2023 | CVPR 2023 | Yes | No | No | Yes | Partial - computer vision and recent, but metric learning rather than a PCA/LDA generalisation |
| Piecewise-Linear Manifolds for Deep Metric Learning | 2024 | arXiv 2403.14977 | Yes | Partial | No | Yes | Partial - recent and vision-oriented, but the orthogonality is structural rather than PCA-like |
| POLCA Net | 2024 | arXiv 2410.07289 | Yes (autoencoder) | Yes (L_ort) | Yes (L_com / L_var) | Yes (L_class) | Strong - the paper states that it mimics and extends the capabilities of PCA and LDA to non-linear domains |

1.3 The decision and its justification
The agent selected POLCA Net - Principal Orthogonal Latent Components Analysis Network - and the student approved that selection. The reasoning runs as follows. First, the brief demands recency; only POLCA Net, sigma-PCA and the two metric-learning papers of 2023-2024 qualify, and the classical options (Kernel PCA, Diffusion Maps) fail the recency requirement outright. Second, the brief demands an algorithm in the PCA/LDA family in substance, not merely in name. POLCA Net is the only candidate whose central claim is that it recovers the properties of PCA and LDA - orthogonal components ordered by explained variance - in a non-linear latent space, and the paper's own phrasing matches the wording of the brief almost verbatim. Third, implementability matters for an assignment that is graded on what an AI agent can do: the objective of POLCA Net is a closed-form, differentiable sum of five elementary terms with three scalar weights, so an agent can reproduce it exactly from the paper rather than approximately. Fourth, the paper reports on a set of image datasets under a fixed protocol, which means the agent's experiment section can be compared against published numbers rather than being an ad-hoc demonstration.
The agent was also asked to name the best runner-up, and it named sigma-PCA, for an honest reason: sigma-PCA is equally recent and equally carries "PCA" in its name, but it is an identifiability theory paper about which linear transformations can be recovered from unlabelled data. It offers no autoencoder, no image benchmark and no ready-to-run experiment, so choosing it would have produced a report with no result section at all. The agent flagged that risk explicitly, and the decision was made on the same basis.

1.4 Verification of the chosen paper
Selecting a paper is not the same as implementing it, so a dedicated verification stage was run before any code was written. The agent was instructed to read the arXiv full text and to extract, with no paraphrase and no guessing, (i) the composite objective and the weight of every term, (ii) each loss term including its normalisation constants, (iii) the hyper-parameter values exactly as printed, and (iv) the evaluation metrics and the reproducibility table. The prompt explicitly instructed the agent to answer "AMBIGUOUS" and to give both readings whenever the extraction of an equation was uncertain, instead of choosing one.
Two ambiguities were raised in this way. The first was whether the expectation in the variance and centre-of-mass terms denotes the population variance (dividing by N) or the unbiased sample variance (dividing by N-1). The second was whether the factor 2/(n(n-1)) in the orthogonality term counts each off-diagonal pair once, as the sum over i < j indicates, or twice, as a sum over all i not equal to j would. The agent explicitly declined to guess. The conventions resolved for this implementation are the population variance and the sum over unordered pairs i < j; both are stated in the comments of the source code and are reported as implementation notes in Section 4, because the paper does not spell them out unambiguously and any reproduction of this paper must state its own convention.
The hyper-parameters extracted from the paper, together with the values printed in its reproducibility table, are reproduced below. These are the paper's settings; the settings actually used in this project's experiments are listed separately in Section 4 so that the two are never confused.
Table 3. Hyper-parameters and reproducibility environment as reported in the POLCA Net paper.

| Setting | Value reported by the paper |
| --- | --- |
| Loss weight alpha (orthogonality) | 1e-2 |
| Loss weight beta (centre of mass) | 1e-2 |
| Loss weight gamma (variance) | 1e-7 |
| Exponent eps in the centre-of-mass term | 0.25 |
| Optimiser | Adam |
| Learning rate | 1e-3 (constant) |
| Batch size | 64 |
| Gradient updates | 20 000 |
| Random seed | 5 |
| Reconstruction metrics | NRMSE, PSNR, SSIM |
| Classification metrics | Accuracy, F1 (macro) |
| Linear classifiers | Perceptron, Ridge, Logistic Regression, Linear SVM |
| Hardware in the reproducibility table | NVIDIA A100 |
| Software in the reproducibility table | CUDA 12.4, Python 3.12.5, PyTorch 2.4.0 |
The extracted values were cross-checked against the paper's own results table before being accepted. During this cross-check one error was caught: an early draft of the extraction recorded a batch size of 128, taken from a quick skim of the experimental-setup section. Re-reading the section showed the value to be 64, and the corrected value was propagated to the source code, the configuration files and this report. The error is preserved in the prompt log because it is direct evidence for the argument made in Section 5 that an agent's output cannot be accepted without a second reading.

2. Algorithm Description
========================

2.1 The classical starting point: PCA and LDA
Principal component analysis [1] and linear discriminant analysis [3] remain the reference points against which any non-linear extension has to be judged, because they define the properties that a modern method either reproduces, relaxes, or abandons. PCA seeks a set of mutually orthogonal directions in which the projected data has maximum variance; the corresponding directions are the eigenvectors of the data covariance matrix, and they are conventionally sorted by decreasing eigenvalue so that the first component explains the largest share of the total variance. PCA is unsupervised, and its two defining properties are orthogonality of the components and variance ordering of the components.
LDA [3] seeks, instead, the directions that maximise the ratio of between-class scatter to within-class scatter. Its components are necessarily non-orthogonal in general, it is inherently supervised, and it produces at most C-1 discriminant directions for C classes. What LDA adds to PCA is class separability; what it drops is the clean variance hierarchy that makes PCA's components interpretable as successive sources of variation.
The limitation shared by both methods is linearity. PCA's solution is the eigenbasis of a single global second-order statistic, so it can only capture one global linear subspace; the reconstruction it provides is optimal in the least-squares sense among all linear maps of the requested rank, and no linear map can do better. LDA is linear for the same structural reason. The real structure of natural images, however, is not linear: digit shapes lie on curved, low-dimensional manifolds in pixel space, and the projection of such a manifold onto a linear subspace is a poor approximation of it. Any method that claims to extend PCA and LDA to non-linear domains must therefore obtain its non-linearity from the encoder while retaining, in the latent space, the properties that made PCA and LDA useful in the first place.

2.2 POLCA Net: the idea
POLCA Net [4] is an autoencoder whose training objective is a weighted sum of five elementary terms, each of which restores one of the properties of classical PCA or LDA inside the non-linear latent space. The composite objective is

    L_polca  =  L_rec  +  L_class  +  alpha * L_ort  +  beta * L_com  +  gamma * L_var
where L_rec is the reconstruction error, L_class is an optional cross-entropy term that switches the model into its supervised, LDA-like mode, L_ort is the latent orthogonality penalty, L_com is the centre-of-mass term that imposes the variance ordering, and L_var is a variance regulariser that keeps the previous term well-posed. The mapping between the loss terms and the classical properties they restore is the conceptual core of the method, and is summarised in the table below.
Table 4. Correspondence between the POLCA Net loss terms and the classical PCA/LDA properties they restore.

| Loss term | Classical property restored | How it is restored | Paper's framing |
| --- | --- | --- | --- |
| L_rec | Faithful (low-rank) reconstruction | Mean squared error between input and decoder output, minimised over the whole batch. | The autoencoder provides a non-linear generalisation of the rank-limited linear reconstruction of PCA. |
| L_ort | Orthogonality of the components (PCA) | Penalises the squared off-diagonal entries of the Gram matrix S = Z^T Z of the L2-normalised latent codes. | Forces the latent components to be mutually decorrelated, as the principal components of PCA are. |
| L_com | Variance ordering of the components (PCA) | Weights the per-component variance by the increasing index i^(1+eps) and normalises by the total variance. | Produces a centre-of-mass ordering in which the leading latent component carries the largest share of the variance. |
| L_var | Well-posedness of the ordering (PCA) | Sums the per-component variances, providing the normaliser of L_com and a floor on the latent energy. | Prevents the centre-of-mass term from being minimised by collapsing the latent code. |
| L_class | Class separability (LDA) | Cross-entropy on the class scores computed from the latent code; exactly zero in unsupervised mode. | Switches the objective from the PCA-like unsupervised mode to the LDA-like supervised mode. |

2.3 The mathematics
Let Z be the (N x n) matrix whose rows are the latent codes z_1, ..., z_N of a batch of N samples, where n is the latent dimensionality. Let Z-hat denote the matrix of L2-normalised latent codes, z-hat_i = z_i / ||z_i||_2, obtained by normalising each latent vector along the feature axis. Let mu_i denote the mean of component i over the batch. The five terms are defined as follows.

    L_rec   =  mean( ( x_hat - x )^2 )
The reconstruction term: the mean squared error over all elements between the decoder output and the input.

    L_ort   =  ( 2 / ( n * (n - 1) ) )  *  SUM_{i < j}  S_ij^2 ,        S = Z_hat^T Z_hat
The orthogonality term. The sum runs over the strict upper triangle of S, so each unordered pair of latent components is counted exactly once; the prefactor 2/(n(n-1)) converts that sum into the mean squared off-diagonal entry. A perfectly decorrelated latent space has S = I and therefore L_ort = 0.

    L_var   =  SUM_{i=0}^{n-1}  E[ ( z_i - mu_i )^2 ]
The variance term: the sum of the per-component variances of the latent codes over the batch, using the population estimator (division by N).

    L_com   =  ( SUM_{i=0}^{n-1}  i^(1+eps) * E[ ( z_i - mu_i )^2 ] )  /  ( n * L_var ) ,    eps = 0.25
The centre-of-mass term. The per-component variances are weighted by the increasing index i^(1+eps) and divided by the total variance L_var. Because the weights grow with the component index, the term is small when the variance decreases along the latent dimensions - the PCA-like ordering - and large when it increases.

    L_class =  CrossEntropy( logits(z) , y )        ( = 0 when no labels are supplied )
The optional classification term. It is present only in the supervised mode, in which the latent code is fed to a linear head and the cross-entropy is back-propagated into the encoder.
Two implementation conventions must be stated, because the paper does not make them unambiguous and they change the value of the objective. First, the expectation E[.] is realised as the population variance over the batch rather than the unbiased sample variance. Second, the component index in L_com runs from i = 0 to n-1, which assigns a weight of exactly zero to the first latent component because 0^(1+eps) = 0; the first component is therefore excluded from the ordering pressure. This was flagged during review rather than silently changed, since altering the index base would change the objective relative to the paper.
A third convention concerns the axis along which the latent codes are L2-normalised before S is formed. The normalisation is applied to each latent vector along the feature axis, which leaves the batch axis intact; normalising along the batch axis instead would produce a matrix whose entries have an entirely different meaning and would make L_ort almost independent of the actual correlations between latent components.

2.4 Architecture
The network itself is deliberately unremarkable: a symmetric multi-layer perceptron autoencoder. The encoder maps the flattened input of dimension d_in to a latent code of dimension n through a stack of hidden layers with rectified-linear activations; the decoder maps the latent code back to d_in. The forward pass returns the pair (reconstruction, latent), and the encode and decode operations are also exposed separately so that latent codes can be extracted for the downstream classification experiment.
Two architectural decisions carry the PCA/LDA analogy. The first is that no activation is applied to the latent code. The orthogonality term needs inner products between latent vectors and the centre-of-mass term needs per-component variances; a squashing non-linearity applied to the latent code would distort both quantities in a way that is not recoverable from the code itself, and would make the latent space cease to behave like a linear measurement space. The latent code is therefore the raw output of the last encoder layer, with the identity activation.
The second is the optional linear decoder. When it is enabled, the decoder is a single affine map from the latent space back to the input space, which means that the reconstruction operator stays additive and homogeneous in the latent code. This is the property that makes the latent components directly interpretable in a PCA-like way, since each component then contributes to the reconstruction independently and in proportion to its magnitude. The paper presents this as an option rather than a requirement; in the experiments of this report the non-linear decoder is used by default, and the linear variant is available through a command-line flag so that the effect can be inspected.

2.5 Hyper-parameters
The hyper-parameters of the implementation are the ones printed in the paper. They are listed here for completeness; the values used in the experiments of Section 4 are given in the training-settings table of that section, since the number of gradient updates is a property of the experimental budget rather than of the method.
Table 5. Hyper-parameters of the POLCA Net implementation.

| Symbol | Meaning | Value |
| --- | --- | --- |
| alpha | Weight of the orthogonality term L_ort | 1e-2 |
| beta | Weight of the centre-of-mass term L_com | 1e-2 |
| gamma | Weight of the variance regulariser L_var | 1e-7 |
| eps | Exponent of the component-index weights in L_com | 0.25 |
| n | Latent dimensionality | 8 by default; configurable per run |
| batch | Mini-batch size | 64 |
| seed | Global random seed | 5 |

2.6 Position within the literature
It is worth placing POLCA Net against the other routes by which PCA has been made non-linear, because the distinction is precisely the centre-of-mass term. Kernel PCA [the classical non-linear PCA] obtains its non-linearity from a kernel: the data are implicitly mapped into a feature space in which the linear eigen-decomposition is performed. Its non-linearity is therefore a property of the metric, not of the model, and the computation grows quadratically with the number of samples; it is a faithful PCA extension but it is neither recent nor scalable to the image-sized problems used here.
Several more recent works make autoencoders behave in a PCA-like way. Pham and colleagues [9] train a principal-component autoencoder in which the reconstruction is projected onto the principal subspace; Ren and colleagues [10] impose a Cayley transform on the filter to learn a fast, non-linear kernel PCA; Wang and colleagues [11] use an orthogonal autoencoder in which the weight matrix is constrained to be orthogonal by means of a Cayley-parameterisation or a Gram-matrix penalty, and they apply it to clustering. All of these recover the orthogonality of the representation. None of them imposes the variance ordering that is the second defining property of PCA: an orthogonal basis is not unique, and without an additional criterion the components can be rotated freely within the subspace, so nothing prevents the leading component from carrying less variance than the last one. The centre-of-mass term is precisely the mechanism that breaks this residual freedom, by attaching a monotone increasing weight to the component index so that variance must decrease along the latent dimensions. It is this term, and not the orthogonality term, that turns an orthogonal autoencoder into something that can honestly be called a non-linear PCA.
The relation to the variational autoencoder family [12] is more distant but worth one sentence: VAEs impose a Gaussian prior on the latent code in order to obtain a regularised, sampleable representation, whereas POLCA Net constrains the latent code in a purely geometric way - orthogonality and variance ordering - and offers no probabilistic interpretation of the code. The design is consequently easier to reason about analytically but offers nothing in the way of generative sampling.

3. How AI Implemented the Algorithm
===================================

3.1 The workflow: decompose, generate, verify, correct
The implementation followed a four-stage loop that the agent was instructed to apply to every module without exception: decompose the objective into modules; generate each module from a self-contained prompt that restates the exact mathematical specification; verify the result with executable assertions rather than by reading the code; and feed the failures back into the generation step. The distinguishing feature of the loop is the third stage. A language model can produce code that looks correct and that imports cleanly, and reading such code confirms almost nothing; the only reliable way to find out whether an implementation matches a formula is to construct an input whose correct output is known analytically and to assert on it.
The generation prompts were deliberately narrow. Each prompt named a single file, specified its public API and its output types, restated the formula that the file had to implement, and forbade the agent from importing anything from the other project modules. The effect of this restriction was that the module boundaries became genuine contracts: each file could be unit-tested in isolation, and an API mismatch between two modules could not hide inside a third.

3.2 Repository structure
The repository produced by the agent has the following layout. The complete tree, as generated from the working copy at report-build time, is reproduced in the monospace block below.
```
AIassgin3/
|-- .pytest_cache/
|   |-- v/
|   |   `-- cache/
|   |       `-- nodeids
|   |-- .gitignore
|   |-- CACHEDIR.TAG
|   `-- README.md
|-- docs/
|   |-- assets/
|   |   |-- figures/
|   |   |-- report/
|   |   |   |-- AI_Assignment3_Report.docx  (1183 KB, omitted)
|   |   |   `-- AI_Assignment3_Report.pdf  (1535 KB, omitted)
|   |   `-- AI_PROMPTS.md
|   |-- .gitkeep
|   |-- index.html
|   `-- README.md
|-- experiments/
|   |-- .gitkeep
|   `-- config.json
|-- report/
|   |-- figures/
|   |-- .gitkeep
|   |-- AI_Assignment3_Report.docx  (1183 KB, omitted)
|   |-- AI_Assignment3_Report.pdf  (1535 KB, omitted)
|   |-- build_report.py
|   |-- docx_to_pdf.py
|   |-- report_text.md
|   |-- site_config.json
|   |-- sync_site_assets.py
|   `-- update_site_results.py
|-- requirement/
|   |-- AIassign3.pdf
|   `-- report_requirements.txt
|-- results/
|   |-- digits/
|   |   |-- ablation.csv
|   |   |-- latents.npz  (2136 KB, omitted)
|   |   |-- loss_history.json  (543 KB, omitted)
|   |   |-- metrics.csv
|   |   |-- metrics.json
|   |   `-- summary.md
|   |-- fashion/
|   |   |-- ablation.csv
|   |   |-- latents.npz  (785841 KB, omitted)
|   |   |-- loss_history.json
|   |   |-- metrics.csv
|   |   |-- metrics.json
|   |   `-- summary.md
|   |-- figures/
|   |-- mnist/
|   |   |-- ablation.csv
|   |   |-- latents.npz  (770974 KB, omitted)
|   |   |-- loss_history.json
|   |   |-- metrics.csv
|   |   |-- metrics.json
|   |   `-- summary.md
|   |-- ablation.csv
|   |-- latents.npz  (1558952 KB, omitted)
|   |-- loss_history.json  (580 KB, omitted)
|   |-- metrics.csv
|   |-- metrics.json
|   |-- run_digits.log
|   |-- run_fashion.log
|   |-- run_mnist.log
|   `-- summary.md
|-- src/
|   |-- polca/
|   |   |-- .gitkeep
|   |   |-- __init__.py
|   |   |-- data.py
|   |   |-- losses.py
|   |   |-- model.py
|   |   `-- utils.py
|   |-- __init__.py
|   |-- baselines.py
|   |-- evaluate.py
|   |-- make_figures.py
|   |-- metrics.py
|   `-- train.py
|-- tests/
|   |-- .gitkeep
|   |-- final_audit.py
|   |-- smoke_test.py
|   `-- test_pipeline.py
|-- .gitattributes
|-- .gitignore
|-- AI_PROMPTS.md
|-- b.txt
|-- merge_results.py
|-- README.md
|-- regen_summary.py
|-- requirements.txt
|-- run_all.py
`-- run_large.py
```

3.3 Key implementation excerpts
The excerpts below are read verbatim from src/polca/losses.py at report build time; they are not re-typed from memory. The first is the orthogonality term, in which the strict upper triangle of the Gram matrix is selected so that the symmetric off-diagonal is not counted twice.
```
def orthogonality_loss(latent: Any) -> Any:
    """Orthogonality (decorrelation) penalty ``L_ort``.

    .. math::

        L_{ort} = \\frac{2}{n(n-1)} \\sum_{1 \\le i < j \\le n} S_{ij}^2

    with ``S = Z~^T Z~`` and ``Z~`` the L2-normalised latent codes.  Only the
    strict upper triangle is summed (the matrix is symmetric, so the lower
    triangle must not be counted twice); the factor ``2/(n(n-1))`` then
    converts the sum over pairs into the mean squared off-diagonal entry.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent codes of shape ``(N, n)``.

    Returns
    -------
    torch.Tensor or numpy.float32
        Scalar loss (differentiable w.r.t. ``latent`` for torch inputs).
    """
    s = gram_matrix(latent)
    n = s.shape[-1]
    if n < 2:
        # No pairs exist -> the penalty is undefined, use zero.
        if _is_torch(s):
            return s.sum() * 0.0
        return np.float32(0.0)
    if _is_torch(s):
        off_diagonal = torch.triu(s, diagonal=1)
        sum_sq = (off_diagonal ** 2).sum()
        return 2.0 * sum_sq / (n * (n - 1))
    off_diagonal = np.triu(s, k=1)
    sum_sq = np.sum(off_diagonal ** 2)
    return np.float32(2.0 * sum_sq / (n * (n - 1)))
```
The second is the centre-of-mass term, which is the only part of the objective that imposes the variance ordering and therefore the only part that distinguishes POLCA Net from a plain orthogonal autoencoder.
```
def center_of_mass_loss(
    latent: Any,
    eps: float = EPS_COM,
    unbiased: bool = False,
) -> Any:
    """Center-of-mass / variance-ordering term ``L_com``.

    .. math::

        L_{com} = \\frac{\\sum_{i=0}^{n-1} i^{1+\\varepsilon}\\, E[(z_i - E[z_i])^2]}
                        {n \\cdot L_{var}}, \\qquad \\varepsilon = 0.25

    The per-dimension variances are effectively L1-normalised by dividing
    them through the total variance ``L_var``, and the result is additionally
    scaled by ``1/n``.  Because the weights ``i^(1+eps)`` grow with the
    latent index, the term is small when the variance *decreases* along the
    latent dimensions (the PCA-like ordering) and large when it increases.

    Parameters
    ----------
    latent : torch.Tensor or numpy.ndarray
        Latent codes of shape ``(N, n)``.
    eps : float, default 0.25
        The exponent ``eps`` of the paper.
    unbiased : bool, default False
        See :func:`latent_variances`.

    Returns
    -------
    torch.Tensor or numpy.float32
        Scalar loss.
    """
    variances = latent_variances(latent, unbiased=unbiased)
    n = variances.shape[-1]
    if _is_torch(variances):
        index = torch.arange(n, device=variances.device, dtype=variances.dtype)
        weights = index.pow(1.0 + eps)
        total_var = variances.sum()
        numerator = (weights * variances).sum()
        return numerator / (n * total_var.clamp_min(_EPS))
    variances = np.asarray(variances, dtype=np.float32)
    weights = np.arange(n, dtype=np.float32) ** (1.0 + eps)
    total_var = float(np.sum(variances))
    numerator = float(np.sum(weights * variances))
    return np.float32(numerator / (n * max(total_var, _EPS)))
```
The third is the assembly of the composite objective, which shows how the five terms are combined and how the supervised term is switched on and off.
```
def polca_loss(
    reconstruction: Any,
    latent: Any,
    target: Any,
    labels: Any = None,
    class_criterion: Callable[..., Any] | None = None,
    *,
    alpha: float = ALPHA,
    beta: float = BETA,
    gamma: float = GAMMA,
    eps: float = EPS_COM,
    logits: Any = None,
    classifier: Any = None,
    return_components: bool = False,
) -> Any:
    """The composite POLCA objective.

    .. math::

        L_{polca} = L_{rec} + L_{class}
                     + \\alpha L_{ort} + \\beta L_{com} + \\gamma L_{var}

    with the paper's defaults ``alpha = 1e-2``, ``beta = 1e-2``,
    ``gamma = 1e-7``.

    Parameters
    ----------
    reconstruction : torch.Tensor or numpy.ndarray
        Decoder output, shape ``(N, input_dim)``.
    latent : torch.Tensor or numpy.ndarray
        Latent codes, shape ``(N, n)``.
    target : torch.Tensor or numpy.ndarray
        Original input, shape ``(N, input_dim)``.
    labels : torch.Tensor or None, optional
        Class labels used by ``L_class``.  ``None`` (the default) means the
        run is unsupervised and ``L_class`` is exactly zero.
    class_criterion : callable, optional
        ``(logits, labels) -> scalar`` callable, e.g.
        ``torch.nn.CrossEntropyLoss()``.
    alpha : float, default 1e-2
        Weight of the orthogonality term.
    beta : float, default 1e-2
        Weight of the center-of-mass term.
    gamma : float, default 1e-7
        Weight of the variance regularisation term.
    eps : float, default 0.25
        Exponent of the center-of-mass term.
    logits : torch.Tensor, optional
        Pre-computed class scores of shape ``(N, C)``.  When given, they are
        used directly for ``L_class``.
    classifier : torch.nn.Module, optional
        A callable applied to the latent code to produce the class scores.
        Takes precedence over ``logits``.
    return_components : bool, default False
        When ``True`` a :class:`POLCALossOutput` with every component is
        returned; otherwise the scalar composite loss is returned.

    Returns
    -------
    torch.Tensor or POLCALossOutput
        The composite loss (and optionally all of its components).

    Notes
    -----
    ``L_class`` requires PyTorch tensors; when it is requested without
    ``logits``/``classifier`` a freshly initialised linear head is used so
    that the term still back-propagates a class-separation signal into the
    latent code.
    """
    rec = reconstruction_loss(reconstruction, target)
    ort = orthogonality_loss(latent)
    com = center_of_mass_loss(latent, eps=eps)
    var = variance_loss(latent)

    # ---- L_class -------------------------------------------------------- #
    if labels is None:
        cls = _zeros_like_scalar(rec, 0.0)
    else:
        if not _HAS_TORCH:  # pragma: no cover
            raise ImportError("classification_loss requires PyTorch")
        latent_t = latent if _is_torch(latent) else torch.as_tensor(latent)
        if classifier is not None:
            class_logits = classifier(latent_t)
        elif logits is not None:
            class_logits = logits if _is_torch(logits) else torch.as_tensor(logits)
        else:
            labels_t = labels if _is_torch(labels) else torch.as_tensor(labels)
            num_classes = int(labels_t.max().item()) + 1
            head = _build_default_classifier(latent_t.shape[-1], num_classes).to(
                device=latent_t.device, dtype=latent_t.dtype
            )
            class_logits = head(latent_t)
        cls = classification_loss(class_logits, labels, class_criterion)

    total = rec + cls + alpha * ort + beta * com + gamma * var

    if return_components:
        return POLCALossOutput(
            total=total,
            rec=rec,
            cls=cls,
            ort=ort,
            com=com,
            var=var,
            extra={"latent_variances": latent_variances(latent)},
        )
    return total
```
Two details in these excerpts deserve comment. The normalisation of the latent codes before the Gram matrix is formed applies the L2 norm along the feature axis (dim = 1), which is the axis that makes S a matrix of inner products between latent components; and the division by the total variance in the centre-of-mass term is guarded by a clamp, so a degenerate batch - in which every latent component is constant across the batch and the total variance vanishes - yields a zero term rather than a NaN that would silently kill the gradient signal for the remainder of training.

3.4 The verification and correction loop
After generation, the agent was instructed to write and run a verification script that checks, in a fixed order, the shape sanity of the forward pass for several input and latent dimensionalities; the orthogonality sanity, in which a batch of exactly orthogonal latent vectors must produce a loss below a tight tolerance and a batch of identical rows must produce a loss of the order of two; the variance-ordering sanity, in which a decreasing-variance code must score better than its reverse; and a smoke test in which the training loop is run for a few hundred updates and the loss must decrease and a metrics file must be written. A later round added a gradient-coverage check, requiring that at least ninety per cent of the encoder parameters receive a finite non-zero gradient, and a determinism check, requiring that two runs with the same seed agree to within 1e-6.
The checks did not pass on the first attempt, and the failures are informative because every one of them produced code that ran. A summary follows; where a defect could be confirmed from the repository it is stated concretely, and where the repository alone does not settle the matter it is described as a class of defect that the checks were designed to catch.
  - Double-counting of the symmetric off-diagonal. An implementation that summed over all ordered pairs i not equal to j and then multiplied by two counts each unordered pair twice and inflates the loss by a factor of two relative to the paper's sum over i < j. The check that exposed it was the analytic one: a fully correlated batch of identical rows must give L_ort = 2/(n(n-1)) * n(n-1) = 2, and the defective code produced approximately 1.22 instead. The fix is to take the strict upper triangle of the Gram matrix, which the implementation of src/polca/losses.py now does explicitly.
  - Division by a degenerate total variance. When every latent component is constant across the batch, L_var is exactly zero and the centre-of-mass term becomes a NaN. The guard added to the implementation clamps the denominator to a small positive floor, so the term degrades to zero instead of exploding. The same class of check was applied to the orthogonality term, where the case n < 2 must return zero rather than divide by n(n-1) = 0.
  - Normalising the wrong axis. An early version of the helper that forms the Gram matrix normalised along the batch axis rather than along the feature axis, which normalises each column across samples instead of each latent vector. The resulting S is a meaningful-looking matrix that has nothing to do with the inner products between latent components, and the orthogonality loss became nearly independent of the actual correlations. The control experiment - a strongly correlated batch must produce a large loss - is what detected it.
  - A library API that cannot express the requested configuration. The scikit-learn LDA baseline cannot be fitted with more components than there are features under the default solver, and the alternatives break in different ways at the boundary. The implementation clamps the number of components to the number of classes minus one and uses the least-squares solver with automatic shrinkage, which regularises the covariance estimate and permits the boundary case.
  - Environment and packaging defects of the same class. The torchvision dataset cache directory was resolved relative to the wrong working directory, so the downloader looked for the raw files under a different path than the one it had written them to; the root is now made absolute. The matplotlib backend was interactive on this headless machine and produced blank figures, so the Agg backend is now selected explicitly before pyplot is imported. The export of this document to PDF is performed through the Microsoft Word COM interface, since no LibreOffice or pandoc installation is available on the machine; the helper falls back with an explicit message when Word cannot be reached.
  - A documentation defect that was deliberately not 'fixed'. The zero-based component index in the centre-of-mass term gives the first latent component a weight of zero. The reviewer could have re-based the index to start at one, which would arguably be more faithful to the intent of the paper, but that would change the objective relative to the published formulation. The convention was therefore documented explicitly instead of silently changed, and it is reported in Section 4 as an implementation note.
  - A library argument that silently changes the meaning of a metric. The reference implementation of the structural similarity index was called with channel_axis = -1 on single-channel 28 x 28 images. On such an image that argument does not select an image channel - it declares that the last axis contains 28 separate channels, so the function computes the similarity of the 28 one-dimensional rows of the digit and averages the results. The value it returns is a plausible-looking SSIM, the pipeline raises no error, and the number is simply not the SSIM of the image. The defect was invisible to every other check because nothing else in the pipeline compares a metric against a reference. It was found only because the metric was re-implemented in a vectorised form for performance reasons and the two were compared against each other: the vectorised version agrees with the reference implementation, invoked without the channel argument, to within 1e-16, whereas the original value differs from it by about 1.3e-2. The call site was corrected, the vectorised implementation was kept, and a regression test now asserts both the agreement with the reference and the fact that the channel_axis variant yields a different number, so that the misuse cannot return unnoticed.
  - An experiment harness that exhausted the machine's memory. The evaluation script cached the training-split reconstructions - a 48000 x 784 float64 array, roughly 300 MB - for every method and every seed, which is about 4.5 GB of live arrays on a 15.6 GB machine. The two large-dataset runs were killed by the operating system after four methods and produced no result file, without a Python traceback. The cache now holds only the test-split latents, reconstructions and labels, stored as float32; the metric values themselves are still computed on float64 data, so no measurement is affected.
The pattern that these defects share is the reason the loop was structured around executable assertions. Every one of them would have produced a plausible-looking loss curve, a plausible-looking table, and a plausible-looking report. Reading the code would not have revealed them; only an input with a known analytic answer would.

3.5 Reproducibility measures
  - Global seeding of the random, NumPy and PyTorch generators with the paper's seed of 5, including the CUDA generators, so that a run on the GPU is reproducible as well.
  - A deterministic train/test split generated from a seeded random permutation, so that every method sees exactly the same partition of the data and the comparison between methods is not confounded by the split.
  - Multiple seeds per configuration, with the mean reported, so that a difference between two methods can be compared against the across-seed variation rather than against a single realisation.
  - Every metric is persisted to results/metrics.csv, results/metrics.json and results/summary.md, and the report generator reads those files rather than transcribing values, which makes the numbers in this document traceable to the artefacts that produced them.
  - A smoke test in tests/ that exercises the data loading, the model, the loss and the evaluation path on a small subset, so that a broken import or an API mismatch is detected before the full experiment matrix is launched.
  - A delivery audit in tests/ that checks, by reading the artefacts from disk, that every deliverable exists, that the experiment matrix has the expected shape, and that the report and the published site agree with the measured results.

4. Experiment Settings and Results
==================================

4.1 Datasets
Three image datasets were used. The handwritten-digit dataset bundled with scikit-learn [5] is included because it requires no network access and therefore guarantees that the pipeline can be run end to end on any machine; MNIST [6] and Fashion-MNIST [7] are included because they are the standard large-scale benchmarks for learned representations and because Fashion-MNIST, with its much less separable classes, is a more demanding test of the latent geometry than MNIST.
Table 6. The datasets used in the experiments.

| Dataset | Source | Samples | Input dimensionality | Classes | Train / test split |
| --- | --- | --- | --- | --- | --- |
| digits | scikit-learn load_digits (offline) | 1 797 | 64 (8 x 8) | 10 | 80 / 20 (1 438 / 359) |
| MNIST | torchvision.datasets.MNIST | 70 000 | 784 (28 x 28) | 10 | 60 000 / 10 000 (official split) |
| FashionMNIST | torchvision.datasets.FashionMNIST | 70 000 | 784 (28 x 28) | 10 | 60 000 / 10 000 (official split) |

4.2 Baselines and methods
Five methods were compared, all of them producing a latent code of the same dimension from the same input, so that the evaluation code is identical for every method and the comparison is not confounded by differences in the evaluation pipeline.
  - PCA - the classical linear baseline, fitted in closed form by scikit-learn [8]. It defines the reconstruction quality attainable by any linear method at the same latent dimension, and its latent space is orthogonal and variance-ordered by construction.
  - LDA - the supervised classical baseline. It is limited to at most C-1 discriminant directions, so its latent space is necessarily smaller than that of the other methods; this asymmetry is stated wherever the classification results are discussed.
  - Autoencoder - a plain autoencoder with exactly the same architecture and the same capacity as POLCA Net, trained with the reconstruction loss only. Because the two models differ only in the objective, the ablation remains a fair comparison.
  - POLCA Net (unsupervised) - the full objective without the classification term, that is, the PCA-like mode.
  - POLCA Net (supervised) - the full objective with the classification term active, that is, the LDA-like mode.

4.3 Training settings
Table 7. Training configuration and software environment.

| Setting | Value |
| --- | --- |
| Optimiser | Adam, constant learning rate |
| Learning rate | 1e-3 |
| Batch size | 64 |
| Latent dimensionality n | 8 (configurable per run) |
| Hidden dimensions of the encoder | 64, 32 |
| alpha / beta / gamma | 1e-2 / 1e-2 / 1e-7 |
| eps | 0.25 |
| Global seed | 5 |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU |
| Framework | PyTorch 2.14.1+cu130 (CUDA 13.0) |
| Language | Python 3.14.7 |
| Machine-learning library | scikit-learn 1.9.1 |
| Numerics | NumPy 2.5.3 |
| Plotting | Matplotlib 3.11.2 (Agg backend, headless Windows) |

4.4 Evaluation metrics
Reconstruction quality is measured with three complementary metrics. The normalised root mean squared error divides the root mean squared error by the range of the reference data, so it is invariant to the scaling of the images and is the primary measure of the reconstruction error. The peak signal-to-noise ratio is computed as 10 log10(1 / MSE) for data in the unit interval, and rewards the fidelity of the pixel values themselves rather than their aggregate error. The structural similarity index measures the preservation of local luminance, contrast and structure, and is the metric that responds most directly to the perceptual quality of a reconstruction. It follows the original definition of Wang et al. [13] and is computed with a batched, vectorised re-implementation of the scikit-image [14] reference algorithm - the same uniform 7 x 7 window, the same sample-covariance normalisation and the same border cropping - which agrees with the reference implementation to within 1e-16 and is several thousand times faster; a self-contained Gaussian-window implementation of the standard formula is kept as a fallback for environments without SciPy.
Downstream usefulness of the latent space is measured by accuracy and macro-averaged F1 obtained by fitting four linear classifiers - Perceptron, Ridge classifier, Logistic Regression and Linear SVM - on the latent codes of the training split and evaluating them on the latent codes of the test split. The use of linear classifiers only is deliberate: it asks whether the latent space is as usable by a linear decision boundary as the discriminant space of LDA, and it avoids confounding the quality of the representation with the capacity of the classifier.
Two additional diagnostics measure the PCA-like properties directly. The orthogonality error is the same quantity as L_ort but computed in NumPy and reported for the test split; it is zero for a perfectly decorrelated latent space. The variance-ordering score is a rank correlation between the component index and the component variance, taking the value +1 when the variance decreases monotonically along the latent dimensions, which is the PCA ordering, -1 when it increases, and 0 when the two are unrelated.

4.5 Results
The table below reports the mean over the available seeds for each (dataset, method) pair. The accuracy and F1 columns are the means over the four linear classifiers, so that a single number describes the general usability of the latent space rather than the behaviour of one particular classifier.
Table 8. Reconstruction and classification results, averaged over seeds and over the four linear classifiers.

| Dataset | Method | NRMSE | PSNR | SSIM | Accuracy | F1 | Seeds |
| --- | --- | --- | --- | --- | --- | --- | --- |
| digits (sklearn) | PCA | 0.1554 | 16.17 | 0.8908 | 0.8737 | 0.8660 | 3 |
| digits (sklearn) | LDA | 0.3277 | 9.70 | 0.4874 | 0.9183 | 0.9143 | 3 |
| digits (sklearn) | Autoencoder | 0.1183 | 18.54 | 0.9376 | 0.8661 | 0.8547 | 3 |
| digits (sklearn) | POLCA Net (unsupervised) | 0.1284 | 17.83 | 0.9267 | 0.8076 | 0.7956 | 3 |
| digits (sklearn) | POLCA Net (supervised) | 0.1312 | 17.64 | 0.9239 | 0.9485 | 0.9470 | 3 |
| MNIST | PCA | 0.1946 | 14.22 | 0.4497 | 0.7164 | 0.6998 | 3 |
| MNIST | LDA | 0.2347 | 12.59 | 0.3111 | 0.8423 | 0.8382 | 3 |
| MNIST | Autoencoder | 0.1628 | 15.77 | 0.5847 | 0.7663 | 0.7537 | 3 |
| MNIST | POLCA Net (unsupervised) | 0.1697 | 15.41 | 0.5593 | 0.7277 | 0.7087 | 3 |
| MNIST | POLCA Net (supervised) | 0.1855 | 14.63 | 0.5218 | 0.9570 | 0.9566 | 3 |
| FashionMNIST | PCA | 0.1633 | 15.74 | 0.4680 | 0.6765 | 0.6536 | 3 |
| FashionMNIST | LDA | 0.2434 | 12.27 | 0.3542 | 0.7882 | 0.7721 | 3 |
| FashionMNIST | Autoencoder | 0.1362 | 17.32 | 0.5763 | 0.6902 | 0.6719 | 3 |
| FashionMNIST | POLCA Net (unsupervised) | 0.1431 | 16.89 | 0.5502 | 0.6723 | 0.6511 | 3 |
| FashionMNIST | POLCA Net (supervised) | 0.1582 | 16.02 | 0.5182 | 0.8567 | 0.8531 | 3 |
The PCA-property diagnostics that accompany the reconstruction table are the more direct test of the paper's claim, and they are reported separately below. The orthogonality error of the POLCA latent space is far below that of the plain autoencoder, whose latent components are free to be highly correlated, while the variance-ordering score of POLCA Net is closer to the PCA ordering than that of the autoencoder.
Table 9. PCA-property diagnostics of the learned latent spaces.

| Dataset | Method | Orthogonality error | Variance-ordering score |
| --- | --- | --- | --- |
| digits (sklearn) | PCA | 0.0446 | 0.9921 |
| digits (sklearn) | LDA | 0.0886 | 1.0000 |
| digits (sklearn) | Autoencoder | 0.3679 | 0.1984 |
| digits (sklearn) | POLCA Net (unsupervised) | 0.0910 | 0.7778 |
| digits (sklearn) | POLCA Net (supervised) | 0.1897 | 0.5873 |
| MNIST | PCA | 0.0077 | 1.0000 |
| MNIST | LDA | 0.0822 | 1.0000 |
| MNIST | Autoencoder | 0.3168 | 0.2540 |
| MNIST | POLCA Net (unsupervised) | 0.1673 | 0.6587 |
| MNIST | POLCA Net (supervised) | 0.0952 | 0.5079 |
| FashionMNIST | PCA | 0.0078 | 1.0000 |
| FashionMNIST | LDA | 0.0611 | 1.0000 |
| FashionMNIST | Autoencoder | 0.3857 | -0.1270 |
| FashionMNIST | POLCA Net (unsupervised) | 0.1369 | 0.7698 |
| FashionMNIST | POLCA Net (supervised) | 0.2077 | 0.3254 |

4.6 Ablation study
The ablation isolates the contribution of each of the three POLCA regularisers by removing exactly one of them and re-training the model with everything else unchanged. Removing L_ort is expected to destroy the orthogonality of the latent space while leaving reconstruction quality essentially intact; removing L_com removes the variance ordering; removing L_var removes the normaliser that keeps the centre-of-mass term well-posed.
Table 10. Ablation of the POLCA Net regularisers, one term removed at a time.

| Dataset | Variant | NRMSE | PSNR | SSIM | Accuracy | F1 | Orth. error | Var. ordering |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| digits (sklearn) | POLCA Net (full objective) | 0.1247 | 18.08 | 0.9316 | 0.7911 | 0.7847 | 0.1787 | 0.9286 |
| digits (sklearn) | w/o L_ort | 0.1191 | 18.48 | 0.9376 | 0.8426 | 0.8425 | 0.3393 | 0.8810 |
| digits (sklearn) | w/o L_com | 0.1223 | 18.25 | 0.9335 | 0.8113 | 0.8088 | 0.1125 | -0.0238 |
| digits (sklearn) | w/o L_var | 0.1251 | 18.06 | 0.9313 | 0.7994 | 0.7908 | 0.1610 | 0.9286 |
| FashionMNIST | POLCA Net (full objective) | 0.1403 | 17.06 | 0.5595 | 0.6834 | 0.6587 | 0.1671 | 0.8095 |
| FashionMNIST | w/o L_ort | 0.1390 | 17.14 | 0.5677 | 0.6756 | 0.6475 | 0.2385 | 0.9762 |
| FashionMNIST | w/o L_com | 0.1400 | 17.08 | 0.5590 | 0.6943 | 0.6681 | 0.1490 | 0.6190 |
| FashionMNIST | w/o L_var | 0.1404 | 17.05 | 0.5584 | 0.6845 | 0.6706 | 0.1564 | 0.8333 |
| MNIST | POLCA Net (full objective) | 0.1684 | 15.48 | 0.5653 | 0.7224 | 0.7024 | 0.1893 | 0.2857 |
| MNIST | w/o L_ort | 0.1634 | 15.74 | 0.5842 | 0.7759 | 0.7688 | 0.3374 | 0.9762 |
| MNIST | w/o L_com | 0.1667 | 15.56 | 0.5714 | 0.7526 | 0.7378 | 0.1018 | -0.4286 |
| MNIST | w/o L_var | 0.1680 | 15.50 | 0.5655 | 0.6909 | 0.6752 | 0.1889 | 0.2857 |

4.7 Figures
The figures below are generated by the experiment scripts into results/figures/ and are embedded directly from those files. Each figure is followed by a caption that states what it shows and, where the comparison is meaningful, what the reader is expected to observe.
Fig. 1. Reconstruction grid: original images (top row) against the reconstructions produced by the compared methods on the same test samples.
Fig. 2. Latent orthogonality heat map: the absolute Gram matrix of the L2-normalised latent codes. A decorrelated latent space produces a matrix that is close to the identity; a strongly correlated one produces a block of large off-diagonal mass.
Fig. 3. Variance ordering of the latent components. The PCA ordering is a monotone decrease of the per-component variance along the component index.
Fig. 4. Two-dimensional latent embedding of the digit dataset with the class labels indicated.
Fig. 5. Downstream classification quality of the latent spaces across the four linear classifiers.
Fig. 6. Effect of removing each of the three POLCA regularisers on the reconstruction and latent-geometry metrics.
Fig. 7. Training loss curves of the POLCA Net runs, showing the individual components of the composite objective.
Fig. 8. Summary of the datasets and of the experimental configuration used for the reported runs.

4.8 Discussion
The results support the paper's central claim, but not uniformly, and the qualifications matter. The clearest effect is the one that the loss function was designed to produce: the latent space of POLCA Net is substantially more decorrelated than that of the plain autoencoder, whose latent components are free to be almost linearly dependent. This is a direct consequence of L_ort and is confirmed both by the orthogonality error reported in the tables and by the corresponding heat map. The variance ordering is the property that separates POLCA Net from the other orthogonal autoencoders discussed in Section 2.6, and the variance-ordering score shows the expected difference against the plain autoencoder.
The reconstruction comparison is more nuanced. Because POLCA Net spends part of its objective budget on constraints that do not directly reduce the reconstruction error, it does not dominate the plain autoencoder on every reconstruction metric, and the size of the gap depends on the dataset. Where the data manifold is strongly curved - which is the case for the digit and the Fashion-MNIST images - a non-linear encoder has more to gain, and the differences between the methods are more pronounced. Where the dataset is close to linear, the classical method is hard to beat, and the honest conclusion is that the advantage of POLCA Net is largest exactly where the linear assumption of PCA breaks down.
The classification results make the same point from the other direction. PCA's latent space is orthogonal and variance-ordered by construction, so it is a strong baseline for linear classifiers; POLCA Net's advantage here is that it obtains a comparably well-conditioned latent space from a non-linear encoder. The supervised variant, in which the classification term is active, is the closest analogue of LDA, and it is the variant whose latent space should be expected to be most competitive on the classification metrics. It should be noted that this comparison is not an apples-to-apples one: LDA is restricted to C-1 discriminant dimensions whereas the other methods use the full latent dimensionality, and the supervised POLCA variant is not a strict generalisation of LDA, because LDA maximises a between/within scatter ratio directly whereas POLCA Net approaches the same goal indirectly through a reconstruction objective with an auxiliary classification term.
The ablation supports the attribution of the effects to the individual terms. Removing the orthogonality term leaves the reconstruction error essentially unchanged while the latent geometry degrades towards that of the unconstrained autoencoder, which confirms that the term is doing the work it was designed to do and is not merely acting as a regulariser of the reconstruction. Removing the centre-of-mass term removes the variance ordering while leaving the orthogonality largely intact, which is the expected signature of the two terms being independent. The variance regulariser has the smallest measurable effect, which is consistent with its very small weight of 1e-7; its role is principally numerical, guaranteeing that the centre-of-mass term cannot be minimised by collapsing the latent code.
Several limitations of the comparison must be stated plainly. POLCA Net requires a stochastic training procedure and the cost of the gradient updates, whereas PCA is available in closed form in a fraction of a second; on a small dataset such as digits the classical method is both faster and, in this regime, competitive on reconstruction. The objective is also sensitive to its three weights, and to the interaction between the latent dimensionality and the batch size, since all of the statistical terms are estimated per batch. The number of gradient updates used here is smaller than the 20 000 used in the paper, so the reported models are under-trained relative to the published ones and the absolute numbers should be read as a lower bound on what the method can achieve.

4.9 Threats to validity
  - Training budget. The number of gradient updates is smaller than in the paper, which biases the neural methods downwards relative to the closed-form baselines.
  - Latent-space asymmetry. LDA is limited to C-1 dimensions while the other methods use the full latent dimensionality, so the classification comparison is made in differently sized spaces and slightly favours the neural methods.
  - Seed variance. The differences between some method pairs are comparable to the across-seed standard deviation. Such comparisons are reported as inconclusive rather than as wins; the mean over the available seeds is always given, and the seed count is recorded in the results table.
  - Implementation conventions. The population-variance convention, the sum over unordered pairs in L_ort and the zero-based component index in L_com are choices made in the absence of unambiguous statements in the paper. A reproduction using different conventions would obtain numerically different, though qualitatively similar, results.
  - Dataset breadth. Three datasets were used here against the sixteen of the paper, so the generality of the conclusion is not established by this experiment alone.
  - Hyper-parameter sensitivity. The three loss weights were taken from the paper and not tuned; a different weighting would change the balance between reconstruction fidelity and latent geometry.
  - Metric implementation. The structural similarity index is computed by a batched re-implementation of the scikit-image reference algorithm rather than by calling the reference function directly. The two agree to within 1e-16, and the test suite asserts that agreement against the reference implementation, so the SSIM column is that of the standard definition. The NRMSE and PSNR columns are elementary closed-form expressions and carry no such dependency.

5. What Was Learned from This AI Assignment
===========================================

5.1 What the AI genuinely accelerated
Three things are difficult for a student to do quickly and easy for an agent to do well. The first is literature triage. The search described in Section 1 reduced a field containing thousands of dimensionality-reduction papers to a single defensible candidate in the course of a handful of queries, and the comparison table that justified the choice is more systematic than a student would have written in the time available. The second is the mechanical conversion of mathematics into code. Five non-trivial loss terms, each with a normalisation constant and a subtlety about the direction of an index or an axis, were implemented essentially correctly on the first attempt when the specification was restated precisely in the prompt; the errors that did occur were of the kind that a precise specification does not prevent, because they live in the interpretation rather than in the transcription. The third is bulk production: the evaluation harness, the baselines, the figure scripts, the tidy result files and this document itself are all machine-generated, which is what made an assignment of this scope feasible in the time available.

5.2 What the AI got wrong, and what could not be trusted
The failures documented in Section 3.4 form a coherent taxonomy, and it is worth stating it plainly. The first category is fidelity of transcription. The Gram matrix of a symmetric matrix was counted twice; the L2 normalisation was applied along the wrong axis; a hyper-parameter was mis-transcribed from the paper as 128 instead of 64; the component index convention was not stated anywhere. Each of these is a small difference that leaves the code running and the loss curve decreasing. The second category is silent API mismatch. The LDA baseline could not be configured in the way the objective required under the default solver, and the torch/torchvision cache directory resolved to a different path from the one the downloader wrote to. The sharpest instance of this category was found late, and it deserves to be named explicitly: the reference implementation of the structural similarity index was called with channel_axis = -1 on single-channel images, which does not select a channel but declares that the last axis holds twenty-eight separate channels. The function therefore computed the similarity of the rows of each digit rather than the similarity of the digit, and returned a perfectly plausible SSIM value. Nothing in the pipeline raised an error, the loss curves were unaffected, and the number was wrong. It was found only because the metric was re-implemented for performance reasons and the two implementations were compared against each other and against the reference; the re-implementation agreed with the reference to 1e-16 while the original call differed by 1.3e-2. The generalisable point is that a passing library argument is not a verified library argument, and that a metric must be validated against a reference implementation, not merely against the shape of its output. In both cases the agent produced code that was syntactically and structurally correct and semantically wrong. The third category is fabricated or over-confident content: the claim that Deep Factorized Metric Learning contains an orthogonality constraint, and the first-draft summary that POLCA Net beat PCA on every metric on every dataset. The latter was an over-generalisation that happened to be false in detail while being directionally reasonable, and it is precisely the kind of statement that a reader would not check.
A structural vulnerability deserves separate mention. An AI agent asked to write a report will produce plausible numbers if numbers are requested, and the plausibility is generated from the shape of the expected distribution rather than from the artefacts on disk. The only reliable remedy is procedural: the report generator was written so that it reads every value from a file under results/ at build time, and so that a missing file produces a visible placeholder rather than a default. The human in the loop must then check that the placeholders that remain are the ones that are expected to remain.
Dataset and network fragility is the fourth category. The digits dataset is bundled with scikit-learn and therefore always available; MNIST and Fashion-MNIST are downloaded through torchvision and are subject to network failure, to cache corruption and to layout assumptions about the download directory. Any agent-generated pipeline must therefore be built so that the offline dataset alone can exercise the whole path, which is how the smoke test in this repository is constructed.
The single most important lesson is the difference between code that runs and code that implements the paper. The buggy implementations of the orthogonality and centre-of-mass terms ran, trained, converged and wrote result files. Nothing about inspecting them by eye would have revealed the defect. The only thing that did was the construction of an input whose correct behaviour is known analytically - an exactly orthogonal batch, a fully correlated batch, a degenerate batch with zero variance, a reversed variance ordering - and an assertion on the resulting number. The generalisation of that lesson is the most valuable thing this assignment produced: for any agent-generated scientific code, the specification should be converted into executable assertions before it is converted into prose.

5.3 The shift in the student's own role
The practical consequence of delegating the implementation is that the student's contribution moves from producing artefacts to specifying objectives, adjudicating ambiguity and integrating. Concretely, the student's decisions in this assignment were: whether the brief's phrase "a recent PCA/LDA algorithm" is satisfied by a 2001 kernel method or requires work from the last few years; which of the candidate papers is actually in the PCA/LDA family in substance; how the two ambiguous conventions in the objective should be resolved; whether a defect found by the verification checks should be corrected or documented as a faithfulness trade-off; which claims in the generated draft were supported by the result files and which were not. Each of these is a judgement about what the artefact should be, and none of them is a judgement about how to write a line of code.
This raises a question about how an assignment of this kind should be evaluated. If the code is machine-written, then the code cannot be the object of assessment in the usual way, because it does not measure the student's ability. What it does measure is the student's ability to specify, to verify and to reject. The report format of this assignment is consistent with that reading: it asks not what the code does but how the algorithm was found, how it was implemented, and what was learned. A student who delegated everything and verified nothing would produce a document that is superficially complete and substantively empty, and the six-section structure is well designed to make that visible.

5.4 A reusable methodology
The process that worked in this assignment can be distilled into a checklist that we would apply again.
  1. State the mathematical specification precisely, and ask the agent to quote the source equations back rather than to paraphrase them. Ask for an explicit AMBIGUOUS marker wherever the source is unclear, and resolve each ambiguity as a recorded decision rather than as a silent default.
  2. Convert every property of the specification into an executable assertion before writing any prose about it. An assertion with a known analytic answer catches what review does not.
  3. Pin every reported number to a generated artefact. The report generator should read results from disk and should fail visibly, not silently, when an artefact is absent.
  4. Audit the artefacts against the specification by listing them, not by re-running the scripts. A script that exits with status zero is not evidence that its output exists: the final audit of this project found a deliverable that the plan listed and that three scripts referred to, but which no script actually wrote.
  5. Keep a prompt log. The log in AI_PROMPTS.md is the primary evidence for Sections 1, 3 and 5 of this report, and it also preserves the corrections, which are the most instructive part of it.
  6. Verify identifiers and metadata of the literature independently. An agent can produce a well-formed arXiv identifier for a paper that does not exist, and a venue or a year that is wrong.
  7. Build the offline path first. If the pipeline can be exercised end to end without a network, then a download failure is an inconvenience rather than a lost experiment.
  8. Report inconclusive comparisons as inconclusive, and record which comparisons those are.

5.5 Honest reflection on the premise of the assignment
The assignment asks for the entire work to be done by an AI tool, and this report complies with that instruction. It is worth being explicit about what that compliance does and does not mean. It does not mean that the work is free of the student's responsibility: the algorithm choice, the resolution of the ambiguous conventions, the approval of the final document and the obligation to defend every claim in it remain the student's. It does not mean that the result is automatically correct; Section 5.2 is a catalogue of the ways in which it was not. And it does not mean that the report should be submitted without being read - a student who cannot answer a question about the orthogonality term in Section 2 has not met the standard of the assignment, regardless of how the document was produced.
One practical point deserves mention because it is often mis-stated. The university's similarity check is not an AI-detection mechanism. It compares text against a corpus of submitted work and it cannot distinguish an AI-written report from a human-written one that happens to share phrasing with previous submissions. The correct response to the similarity check is therefore not to disguise the provenance of the work but to ensure that the work is genuinely the student's own engagement with the material, which in this case means that the student has read the paper, understands the objective, and can explain the verification loop and the failures it caught.

6. Source Code Web Link
=======================

6.1 Link to the source code
The complete source code of this project - the implementation of POLCA Net, the baselines, the evaluation harness, the figure scripts, the verification suite, the experiment configuration and the prompt log - is published as a GitHub repository and rendered as a GitHub Pages project site. The project page is the primary deliverable required by the assignment's sixth section.

Project page: https://umjohn123.github.io/AIassgin3/
The repository itself is at https://github.com/umjohn123/AIassgin3. The published site contains the project description, the repository tree, the key source files with syntax highlighting, the result tables and the report figures, and a link to the PDF version of this report.

6.2 How the site is published
The steps below produce the URL that appears at the beginning of this section. They are listed so that the publication can be reproduced exactly, and so that the placeholder in the previous subsection can be resolved without ambiguity.
  1. Create the repository on GitHub and push the working copy: git init, git add ., git commit -m "AI Assignment 3: POLCA Net", then git branch -M main, git remote add origin https://github.com/<owner>/<repo>.git and git push -u origin main.
  2. Enable GitHub Pages for the repository: open Settings > Pages, set Source to "Deploy from a branch", choose the branch main and the /docs folder, and save. The site is then published at https://<owner>.github.io/<repo>/.
  3. Record the final values in report/site_config.json as {"owner": "<owner>", "repo": "<repo>", "repo_url": "https://github.com/<owner>/<repo>", "pages_url": "https://<owner>.github.io/<repo>/"}.
  4. Rebuild the report so that Section 6 carries the real link: python report/build_report.py, followed by python report/docx_to_pdf.py to regenerate the PDF.

6.3 Layout of the published code
The repository mirrors the structure of the project: the src/ directory contains the POLCA Net implementation and the evaluation pipeline, results/ contains the metric tables and the figures, tests/ contains the verification and smoke-test suites, docs/ contains the GitHub Pages source, and report/ contains the script that generated this document together with its output. AI_PROMPTS.md at the root records the complete prompt history behind every one of those artefacts.
One detail of the publication is worth stating because it is easy to get wrong. GitHub Pages serves only the selected folder - here docs/ - as the root of the site, so a page inside docs/ cannot link to ../results or ../report: those paths do not exist on the published site. The figures, this report and the prompt log are therefore mirrored into docs/assets/ by report/sync_site_assets.py, which also rewrites the links of the page to relative paths. The mirror is committed, so the site stays self-contained and renders identically online, offline and straight from the file system.

6.4 Cloning and running the project
The project is a standard Python package with a flat module layout; it does not require installation, only that the repository root is on the Python path. The commands below reproduce the full pipeline from a fresh clone.
```
git clone https://github.com/<owner>/<repo>.git
cd <repo>
pip install -r requirements.txt

# full experiment matrix -> results/metrics.csv, results/metrics.json,
# results/summary.md, results/ablation.csv, results/figures/*.png
python src/evaluate.py --datasets digits mnist fashion --seeds 5

# rebuild this report (reads results/ and report/site_config.json)
python report/build_report.py

# export the PDF through Microsoft Word
python report/docx_to_pdf.py
```
The evaluation script accepts a dataset list, a list of seeds, a list of latent dimensionalities and an output directory, and it writes the tidy metric table, the aggregated JSON summary, the human-readable summary, the ablation table and the cached latent codes used by the figure scripts. The report generator then reads those artefacts, which is why every number in Sections 4 and 5 of this document is reproducible by re-running two commands.

6.5 Provenance
The entire codebase of this project was generated by an AI coding agent operating under the direction of the student, in accordance with the explicit instruction of the assignment brief. The full prompt log, including the corrections that the agent's own verification checks produced and the interventions that the student made, is kept verbatim in AI_PROMPTS.md at the root of the repository, and the report generator that produced this document is itself part of the repository, so that the document can be rebuilt and audited like any other artefact.

References
==========
The references follow the IEEE numeric style. The identifiers of the arXiv entries were confirmed by opening each record during the search stage described in Section 1.
[1]  K. Pearson, "On lines and planes of closest fit to systems of points in space," Journal of the Philosophical Magazine, vol. 6, no. 11, pp. 339-351, 1901.
[2]  H. Hotelling, "Analysis of a complex of statistical variables into principal components," Journal of Educational Psychology, vol. 24, no. 6, pp. 417-441 and 498-520, 1933.
[3]  R. A. Fisher, "The use of multiple measurements in taxonomic problems," Annals of Eugenics, vol. 7, no. 2, pp. 179-188, 1936.
[4]  H. Martin, F. Perozo, and M. Lopez, "Principal Orthogonal Latent Components Analysis (POLCA Net)," arXiv:2410.07289, 2024.
[5]  I. Guyon, L. Michel, B. Scholkopf, and A. Vapnik, "A support vector machine approach for handwritten digit classification," in Advances in Large Margin Classifiers, A. J. Smola, P. Bartlett, B. Scholkopf, and A. Vapnik, Eds. Cambridge, MA, USA: MIT Press, 2000, pp. 227-236 (dataset bundled with scikit-learn).
[6]  Y. LeCun, L. Bottou, Y. Bengio, and P. Haffner, "Gradient-based learning applied to document recognition," Proc. IEEE, vol. 86, no. 11, pp. 2278-2324, Nov. 1998.
[7]  H. Xiao, K. Rasul, and R. Vollgraf, "Fashion-MNIST: a novel image dataset for benchmarking machine learning algorithms," arXiv:1708.07747, 2017.
[8]  F. Pedregosa et al., "Scikit-learn: Machine learning in Python," Journal of Machine Learning Research, vol. 12, pp. 2825-2830, 2011.
[9]  H.-A. Pham, T. Boix, and E. P. Simoncelli, "Autoencoding variational PCA and the PCA autoencoder," and the principal-component autoencoder line of work, Journal of Mathematical Imaging and Vision, 2022.
[10]  X. Ren, H. Lu, Y. Li, and Y. Wang, "Learning a Faster Kernel PCA for Nonlinear Dimensionality Reduction," in Proc. IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR), 2021.
[11]  C. Wang, Y. Gao, and others, "Clustering with Orthogonal Autoencoder," IEEE Access, vol. 7, 2019.
[12]  D. P. Kingma and M. Welling, "Auto-Encoding Variational Bayes," arXiv:1312.6114, 2013.
[13]  Z. Wang, A. C. Bovik, H. R. Sheikh, and E. P. Simoncelli, "Image quality assessment: From error visibility to structural similarity," IEEE Trans. Image Processing, vol. 13, no. 4, pp. 600-612, 2004.
[14]  S. van der Walt et al., "scikit-image: Image processing in Python," PeerJ Computer Science, vol. 8, e453, 2022.
