# `docs/` — GitHub Pages source

This folder contains the project page for **CISC3024 Pattern Recognition — AI
Assignment #3 (POLCA-Net)**.

| Path | Purpose |
|---|---|
| `index.html` | The entire site. Self-contained: inline CSS, no JavaScript, no external fonts, no CDN. It renders correctly with the network disconnected. |
| `assets/figures/*.png` | The eight report figures, mirrored from `results/figures/`. |
| `assets/report/` | The report in PDF and in Word, mirrored from `report/`. |
| `assets/AI_PROMPTS.md` | The prompt log, mirrored from the repository root. |

## Why the `assets/` folder exists

GitHub Pages publishes **only** the selected folder — here `docs/` — as the site
root. A page inside `docs/` therefore cannot link to `../results/figures/...` or
`../report/...`: those paths do not exist on the published site, and the links
would fall back to the raw repository files on `github.com/umjohn123/AIassgin3` (or
break entirely).

`report/sync_site_assets.py` therefore mirrors every referenced asset into
`docs/assets/` and rewrites the `src` / `href` attributes of `index.html` to
relative paths (`assets/figures/...`, `assets/report/...`, `assets/AI_PROMPTS.md`).
The mirrored copies are committed, so the site stays self-contained.

Re-run it whenever the figures or the report change:

```powershell
python report/sync_site_assets.py
```

## Publishing it

GitHub Pages can serve this folder directly — no build step, no Jekyll front matter, no
`CNAME` required.

### Step 1 — push the folder

Make sure `docs/` is committed and pushed to the default branch:

```powershell
git add docs
git commit -m "docs: update the GitHub Pages project site"
git push origin main
```

### Step 2 — turn on Pages

1. Open the repository on GitHub and go to **Settings**.
2. In the left sidebar choose **Pages**.
3. Under **Build and deployment**, set **Source** to **Deploy from a branch**.
4. Under **Branch**, pick the entry that reads **`main /docs`**. The branch and the
   folder are combined into a single dropdown in the current GitHub interface, and the
   `/docs` suffix is what makes this folder the root of the published site.

   > **Do not pick `main / (root)`.** That publishes the *repository root*, and GitHub
   > then renders the root `README.md` as the landing page instead of this site. The
   > project page would only be reachable at `/docs/`, and a link to the `docs/assets/`
   > folder would return a 404 because no directory index is generated for it.

5. Click **Save**.

### Step 2b — if the dropdown offers no `/docs` entry

The site is published either way. The repository root carries a small redirect page
(`index.html`) that sends the visitor to `docs/`, and both the root and `docs/` carry a
`.nojekyll` marker, which tells GitHub Pages to serve the files as they are instead of
running them through Jekyll. With Jekyll disabled the files are published verbatim,
which is what this self-contained page needs.

### Step 3 — wait and open the site

The first build takes up to a minute. The site then appears at:

```
https://umjohn123.github.io/AIassgin3/
```

GitHub also shows a green **"Visit site"** link on the same Settings → Pages page as
soon as the deployment succeeds.

## Notes

- Every path on the page is **relative to `docs/`**, so the site renders identically
  whether it is opened from a local web server, from the published Pages URL, or
  straight from the file system.
- The metrics table is regenerated from `results/metrics.csv` by
  `report/update_site_results.py`; a cell shows an em dash when the corresponding
  measurement is not present in the result file, so the page can never display a
  number that was not measured.
- The mathematics is **native MathML**, which the browser renders itself with real
  fraction bars, real summation limits and a proper maths font. MathML is markup
  rather than script, so the page still needs no JavaScript, no CDN and no web
  font. The equations come from `report/equations.py` — the same module the Word
  report reads — and are written into the page by `report/render_site_math.py`:

  ```powershell
  python report/render_site_math.py
  ```

  Edit the LaTeX in `report/equations.py`, never the MathML in `index.html`.
- The page adapts to the operating system's light or dark appearance via
  `prefers-color-scheme`, and collapses to a single column below 760 px.
