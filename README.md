# POLCA-Net — CISC3024 Pattern Recognition, AI Assignment #3

A from-scratch implementation of **POLCA Net — Principal Orthogonal Latent Components
Analysis** (arXiv:2410.07289, 2024), an autoencoder that explicitly *mimics and extends
the capabilities of PCA and LDA to non-linear domains* by constraining the latent space
to be **orthogonal** and **ordered by decreasing variance**.

> **AI assistance notice (assignment requirement).**
> The assignment brief states: *"You are expected to use an AI tool or agent to do
> everything for you, including, but not limited to, algorithm searching, programming,
> and even report writing."* and *"You are NOT allowed to write any code by yourself!"*
> This entire project — the literature search, the algorithm selection, every line of
> source code, the experiments and the report — was produced with an AI coding agent.
> The complete, verbatim prompt log is in **[`AI_PROMPTS.md`](AI_PROMPTS.md)**; the
> student's role was to review the agent's output, approve the algorithm choice,
> run the commands, and verify the results.

---

## 1. The algorithm in one page

Classical PCA learns an orthogonal set of latent directions ordered by explained
variance; LDA learns supervised, discriminant-aligned directions. POLCA Net keeps both
properties but allows the projection to be non-linear, by training an autoencoder with a
composite objective:

```
L_polca = L_rec + L_class + alpha * L_ort + beta * L_com + gamma * L_var
```

| Term | Definition | PCA / LDA property recovered |
|---|---|---|
| `L_rec` | `mean((x_hat - x)^2)` | Least-squares fidelity |
| `L_class` | cross-entropy of a linear head (supervised mode only) | LDA's class-discriminant directions |
| `L_ort` | `(2 / (n(n-1))) * sum_{i<j} S_ij^2`, with `S = Z~^T Z~` and `Z~` the row-wise L2-normalised latent codes | Orthogonality / decorrelation of the components |
| `L_com` | `sum_i i^(1+eps) * E[(z_i - Ez_i)^2] / (n * L_var)`, `eps = 0.25` | Variance **ordering** of the components |
| `L_var` | `sum_i E[(z_i - Ez_i)^2]` | Well-scaled latent space; stabiliser for `L_com` |

Paper hyper-parameters: `alpha = 1e-2`, `beta = 1e-2`, `gamma = 1e-7`, `eps = 0.25`,
batch size 64, 20 000 gradient updates, seed 5.

Two conventions the paper does not state explicitly, chosen and documented in the code
and in the report:

1. `E[.]` is the **population** variance (divide by `N`), because the paper's notation
   denotes an expectation over the data distribution.
2. `sum_{i<j}` counts each unordered off-diagonal pair of `S` **exactly once**.

Evaluation follows the paper: reconstruction quality via **NRMSE, PSNR, SSIM**, and
downstream linear separability via **Accuracy / F1** using four linear classifiers
(Perceptron, Ridge, Logistic Regression, Linear SVM).

---

## 2. Repository layout

```
AIassgin3/
├── src/
│   ├── polca/
│   │   ├── losses.py       # L_rec, L_ort, L_com, L_var, L_class, polca_loss
│   │   ├── model.py        # POLCANet + the plain Autoencoder baseline
│   │   ├── data.py         # digits / MNIST / FashionMNIST loaders
│   │   └── utils.py        # seeding, device selection, logging
│   ├── baselines.py        # PCA, LDA and plain-autoencoder baselines
│   ├── metrics.py          # NRMSE, PSNR, SSIM, orthogonality, variance ordering
│   ├── train.py            # training loop + ablation switches
│   ├── evaluate.py         # benchmark driver, four linear classifiers, tidy CSV
│   └── make_figures.py     # all eight report figures
├── tests/
│   ├── smoke_test.py       # shape, orthogonality and variance-ordering assertions
│   ├── test_pipeline.py    # end-to-end evaluation and figure generation
│   └── final_audit.py      # checks every deliverable of the plan is on disk and consistent
├── experiments/config.json # the run configuration used for the reported results
├── results/                # metrics.csv, metrics.json, summary.md, ablation.csv, figures/
├── report/                 # build_report.py, docx_to_pdf.py + the .docx / .pdf report
├── docs/                   # the GitHub Pages site (index.html)
├── run_all.py              # evaluate the three datasets (two waves)
├── merge_results.py        # merge the per-dataset bundles back into results/
├── AI_PROMPTS.md           # full prompt log for the whole assignment
├── requirements.txt
└── README.md
```

---

## 3. How to run

### Install

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

### Quick check (no training)

The `digits` dataset ships with scikit-learn, so this works fully offline:

```powershell
python tests/smoke_test.py
```

### Audit the delivered artefacts

The audit script reads every artefact the assignment plan promises straight from
disk and checks that it exists, is internally consistent and is wired up to the
others. It needs no GPU and re-runs in a couple of seconds:

```powershell
python tests/final_audit.py
```

It verifies the deliverable checklist, the shape of the experiment matrix (180
metric rows, 12 ablation rows, 8 figures), the six sections of the generated
report, the live URLs in it, and that every asset the published site references
resolves inside `docs/`.

### Train POLCA-Net

```powershell
python src/train.py --dataset digits --latent-dim 8 --max-updates 20000 --seed 5 --out-dir results/runs/polca_digits_s5
```

Useful switches:

| Flag | Meaning |
|---|---|
| `--dataset {digits,mnist,fashion}` | which dataset to train on |
| `--latent-dim` | latent dimensionality `n` (default 8) |
| `--hidden-dims` | encoder hidden widths (default `64 32`) |
| `--epochs` | number of passes over the training set (default 50) |
| `--max-updates` | stop after this many gradient updates (the paper uses 20000); overrides `--epochs` |
| `--batch-size` | default 64 |
| `--lr` | Adam learning rate (default 1e-3) |
| `--alpha / --beta / --gamma` | loss weights (1e-2 / 1e-2 / 1e-7) |
| `--supervised` | add the `L_class` term (LDA-style mode) |
| `--no-ort / --no-com / --no-var` | ablate one loss term for the ablation study |
| `--linear-decoder` | use the paper's optional linear decoder |
| `--seed` | random seed (default 5) |
| `--device {cpu,cuda}` | device selection (defaults to CUDA when available) |

### Evaluate and reproduce the results

```powershell
# the full matrix: three datasets x three seeds x five methods + the ablation
python src/evaluate.py --datasets digits mnist fashion --seeds 5 6 7 --max-updates 10000 --out-dir results
python src/make_figures.py --out-dir results
```

`src/evaluate.py` writes a tidy `results/metrics.csv` (one row per dataset × method ×
seed × classifier), an aggregated `results/metrics.json`, a human-readable
`results/summary.md` and the ablation table `results/ablation.csv`.

On a 16 GB machine the three datasets cannot be evaluated concurrently (the LDA and
autoencoder baselines each hold several float64 copies of the 60000 × 784 design
matrix), so `run_all.py` and `run_large.py` evaluate them in two waves and
`merge_results.py` merges the per-dataset bundles back into `results/`:

```powershell
python run_all.py          # digits, then mnist + fashion
python merge_results.py
python src/make_figures.py --out-dir results
```

`run_large.py` starts the MNIST and FashionMNIST runs as **detached** processes, so
they keep running even when the shell that launched them is closed, and the raw
stdout of every run is kept in `results/run_<dataset>.log`.

`merge_results.py` also rebuilds the merged `results/summary.md`. If the per-dataset
bundles are already merged and only the summary needs refreshing, use the lighter
script, which touches neither `latents.npz` nor any other heavy artefact:

```powershell
python regen_summary.py
```

The project site carries the same numbers as the report; regenerate its results
table from the metric file with:

```powershell
python report/update_site_results.py
```

MNIST and FashionMNIST are downloaded once by torchvision into `data/` and cached
there; `digits` needs no network access at all.

### Build the report

```powershell
pip install python-docx pywin32
python report/build_report.py
python report/docx_to_pdf.py
```

The PDF is exported from the `.docx` with Microsoft Word via COM automation
(`win32com.client`), because LibreOffice and pandoc are not installed on this machine.
If Word automation is unavailable, submit the `.docx` alone.

The report generator refuses to insert any number that is not present in `results/`, so
a fabricated measurement cannot reach the document.

---

## 4. Reproducing the results

- **Determinism.** The global seed is 5 by default (the paper's seed), and the run is
  repeated with seeds 5, 6 and 7; the report quotes the mean and the standard deviation
  across seeds.
- **Honest comparisons.** When the spread across seeds exceeds the gap between two
  methods, the comparison is reported as *inconclusive* rather than as a win.
- **No invented numbers.** Every figure in the report comes from `results/metrics.csv`
  or `results/summary.json`.
- **The paper's claims are quoted as claims.** The paper (arXiv:2410.07289) reports that
  POLCA-Net improves NRMSE/PSNR/SSIM over PCA and over a plain autoencoder and achieves
  higher classification accuracy and F1 than PCA and LDA. Those are attributed to the
  paper in the report and on the project page; they are not measurements of this
  repository.

---

## 5. Enabling GitHub Pages

The project page lives in `docs/` and is a **single self-contained HTML file** — no
build step, no external CDN, no web fonts — so it renders identically online and
offline.

GitHub Pages publishes only the `docs/` folder as the site root, so the figures, the
report and the prompt log that the page links to are mirrored into `docs/assets/`
before publishing. Regenerate that mirror whenever the figures or the report change:

```powershell
python report/sync_site_assets.py
```

1. Push the repository to GitHub with the `docs/` folder committed.
2. Go to **Settings → Pages**.
3. Under **Build and deployment → Source**, choose **Deploy from a branch**.
4. Select the **`main`** branch and the **`/ (root)`** folder — *if* the repository
   already has a `docs/` folder at its root, GitHub will offer `/docs` instead; choose
   whichever entry corresponds to `docs/`.
5. Save. The site is published at `https://<user>.github.io/<repo>/` after the first
   build (usually under a minute).

See [`docs/README.md`](docs/README.md) for the longer version of these instructions.

---

## 6. Repository documents

| File | Contents |
|---|---|
| [`AI_PROMPTS.md`](AI_PROMPTS.md) | Verbatim log of every AI prompt, what was accepted and what was corrected — the raw material for report Sections 1, 3 and 5 |
| [`docs/index.html`](docs/index.html) | The GitHub Pages project site: algorithm, repository tree, source showcase, experiments, results, report links |
| [`docs/assets/`](docs/assets/) | The figures, the report and the prompt log mirrored into the site so that every link resolves on GitHub Pages |
| [`docs/README.md`](docs/README.md) | How to publish `docs/` as GitHub Pages |
| [`report/`](report/) | `build_report.py` plus the generated `.docx` and `.pdf` |

---

## 7. Reference

H. Martin, F. Perozo, M. Lopez, *"Principal Orthogonal Latent Components Analysis"*
(POLCA Net), arXiv:2410.07289, 2024.

Classical references: Jolliffe, *Principal Component Analysis*; Fisher, *"The use of
multiple measurements in taxonomic problems"*, 1936.

---

## 8. Licence

MIT — placeholder, to be confirmed by the author before submission.
