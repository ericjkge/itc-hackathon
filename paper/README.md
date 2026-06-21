# AgentHN paper

ACL-formatted preprint of "AgentHN: Self-Editing Agents via Hypernetworks." This
directory is a self-contained LaTeX project: everything needed to compile and
to submit to arXiv lives here.

```
paper/
  main.tex            the paper
  references.bib      bibliography (verified against the cited papers themselves)
  acl.sty              official ACL style file (github.com/acl-org/acl-style-files)
  acl_natbib.bst        official ACL bibliography style
  figures/*.png        the five generated result charts, copied from
                        src/agenthn/webapp/static/ and skill-acquisition/
  main.pdf             compiled output (12 pages, A4), checked in for convenience
```

All other figures (the architecture diagram, the NapLoRA algorithm box, and the
self-improvement accuracy trajectory) are drawn directly in LaTeX with
TikZ/pgfplots/algorithm2e rather than as images, so there is nothing to
regenerate for those.

## Compiling

Any of the following works. The style file pins `[preprint]` mode (non-anonymous,
page-numbered) — switch to `[review]` for anonymous submission or `[final]` for
camera-ready, per `acl.sty`'s header comments.

**Option A — Overleaf.** Create a new project, upload every file in this
directory (keep `figures/` as a subfolder), set `main.tex` as the main document,
compile with pdfLaTeX. Done.

**Option B — tectonic** (what was used to produce `main.pdf` here; no system
TeX install required):
```bash
curl --proto '=https' -fsSL https://drop-sh.fullyjustified.net | sh
./tectonic main.tex
```

**Option C — a standard TeX Live install:**
```bash
pdflatex main.tex
bibtex main
pdflatex main.tex
pdflatex main.tex
```

## Submitting to arXiv

Upload the whole directory (or a tarball of it) — `main.tex`, `references.bib`,
`acl.sty`, `acl_natbib.bst`, and `figures/`. `acl.sty`/`acl_natbib.bst` are
non-standard files arXiv's TeX Live won't already have, so they must be
included; every other package used (`tikz`, `pgfplots`, `algorithm2e`,
`booktabs`, `enumitem`, ...) is a standard CTAN package arXiv already provides.
arXiv runs bibtex automatically when it finds a `.bib` + `\bibliography`
command, so no `.bbl` needs to be pre-generated — though you may commit one
(`bibtex main` produces `main.bbl`) if you want a fully deterministic build.

Before uploading, fill in real author emails in the `\author{}` block — they
were intentionally left out rather than fabricated. Update the affiliations,
title, and date if anything changes between now and submission.

## Reproducing every number in the paper

Every figure, table, and inline statistic is read programmatically from a
script-generated JSON file in the parent repository, not hand-transcribed:

| Claim | Source |
|---|---|
| Curated memory demo (17/18, 16/18, 0/18) | `results/memory_demo.json` |
| Ablation ladder (Table 1) | `results/ablations.json`, `scripts/ablation_study.py` |
| Scaling sweep (Figure 2, Appendix B) | `results/scaling.json`, `scripts/scaling_sweep.py` |
| Statistical validation (Figure 3, Appendix A) | `results/large_scale.json`, `scripts/large_scale_eval.py` |
| Cartridges reimplementation | `results/cartridge.json`, `scripts/cartridge_eval.py` |
| Cost model (Figure 4, Table 2) | `results/cost.json`, `scripts/cost_model.py` |
| Physics skill acquisition (Figure 5) | `skill-acquisition/physics_benchmark_results.json` |
| Personalization transcript | `src/agenthn/webapp/static/fixtures/personalization.json` |
| Self-improvement trajectory (Figure 6) | `src/agenthn/webapp/static/fixtures/skills_product.json` |
| Skill router / debugging-skill examples | `src/agenthn/webapp/static/fixtures/skills_router.json`, `task1-part1/outputs/systematic-debugging.result.json` |
