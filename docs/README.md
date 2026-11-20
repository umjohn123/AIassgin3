# `docs/` — GitHub Pages source

This folder contains the project page for **CISC3024 Pattern Recognition — AI
Assignment #3 (POLCA-Net)**.

| File | Purpose |
|---|---|
| `index.html` | The entire site. Self-contained: inline CSS, no JavaScript, no external fonts, no CDN. It renders correctly with the network disconnected. |

## Publishing it

GitHub Pages can serve this folder directly — no build step, no Jekyll front matter, no
`CNAME` required.

### Step 1 — push the folder

Make sure `docs/index.html` is committed and pushed to the default branch:

```powershell
git add docs
git commit -m "docs: add the GitHub Pages project site"
git push origin main
```

### Step 2 — turn on Pages

1. Open the repository on GitHub and go to **Settings**.
2. In the left sidebar choose **Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Under **Branch**, pick **`main`** and the folder **`/docs`** (GitHub shows it as
   `/ (root)` on some account layouts when no other site exists — if that is what your
   dropdown shows, choose it and GitHub will use the `docs/` folder automatically).
5. Click **Save**.

### Step 3 — wait and open the site

The first build takes up to a minute. The site then appears at:

```
https://<your-github-username>.github.io/<repository-name>/
```

GitHub also shows a green **"Visit site"** link on the same Settings → Pages page as
soon as the deployment succeeds.

## Notes

- The page uses **relative paths** for everything it links to: the report at
  `../report/AI_Assignment3_Report.pdf` and `../report/AI_Assignment3_Report.docx`, the
  prompt log at `../AI_PROMPTS.md`, and the figures at
  `../results/figures/*.png`. Because Pages publishes the `docs/` folder as the site
  root, `../` escapes the published root and those links resolve to the raw repository
  files on `github.com/<user>/<repo>` rather than to the Pages site. If you want the
  figures and the report to be served from the site itself, copy them into `docs/` (for
  example `docs/figures/` and `docs/report/`) and update the `src`/`href` attributes
  accordingly.
- The metrics tables on the page are intentionally filled with placeholders (`—`).
  They are populated from `results/metrics.csv` once the experiment run has completed;
  the numbers in the submitted report are read from the same files.
- The page adapts to the operating system's light or dark appearance via
  `prefers-color-scheme`, and collapses to a single column below 760 px.
