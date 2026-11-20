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
│   ├── baselines.py        # PCAEncoder, LDAEncoder, autoencoder baseline
│   ├── metrics.py          # NRMSE, PSNR, SSIM, orthogonality, variance ordering
│   ├── train.py            # training loop + ablation switches
│   ├── evaluate.py         # four linear classifiers, Accuracy / F1, tidy CSV
│   └── make_figures.py     # all plots
├── tests/
│   └── smoke_test.py       # shape, orthogonality and variance-ordering assertions
├── results/                # metrics.csv, summary.json, figures/
├── report/                 # build_report.py + the .docx / .pdf report
├── docs/                   # this GitHub Pages site (index.html)
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

### Train POLCA-Net

```powershell
python src/train.py --dataset digits --latent-dim 16 --updates 20000 --seed 5 --out results/runs/polca_digits_s5
```

Useful switches:

| Flag | Meaning |
|---|---|
| `--dataset {digits,mnist,fashion_mnist}` | which dataset to train on |
| `--latent-dim` | latent dimensionality `n` (default 16) |
| `--updates` | number of gradient updates (default 20000) |
| `--batch-size` | default 64 |
| `--lr` | Adam learning rate (default 1e-3) |
| `--alpha / --beta / --gamma` | loss weights (1e-2 / 1e-2 / 1e-7) |
| `--supervised` | add the `L_class` term (LDA-style mode) |
| `--ablate {none,ort,com,var}` | remove one loss term for the ablation study |
| `--seed` | random seed (default 5) |
| `--device {auto,cpu,cuda}` | device selection |

### Evaluate and reproduce the results

```powershell
python src/evaluate.py --dataset digits --out results
python src/make_figures.py --results results --out results/figures
```

`src/evaluate.py` writes a tidy `results/metrics.csv` (one row per dataset × method)
and a `results/summary.json`. The full matrix used in the report is reproduced with:

```powershell
foreach ($seed in 5, 6, 7) {
    python src/train.py --dataset digits --seed $seed --out "results/runs/polca_digits_s$seed"
}
python src/evaluate.py --dataset digits --seeds 5,6,7 --out results
python src/make_figures.py --results results --out results/figures
```

Repeat the same three commands for `--dataset mnist` and `--dataset fashion_mnist`.
MNIST and FashionMNIST are downloaded once by torchvision into `data/` and cached
there; `digits` needs no network access at all.

### Build the report

```powershell
pip install python-docx pywin32
python report/build_report.py --results results --out report/AI_Assignment3_Report.docx
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
