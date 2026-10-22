# Report and presentation

This directory preserves the author's combined Prism export, including shared figures, chapter sources, bibliography, supporting uploads, and the two supplied PDFs. Original files and filenames are retained.

| Document | PDF | LaTeX entry point |
| --- | --- | --- |
| Internship report | [Rapport de stage_LisnhuoCHEN.pdf](Rapport%20de%20stage_LisnhuoCHEN.pdf) | `report.tex` |
| Defense presentation | [defense-presentation.pdf](defense-presentation.pdf) | `defense-presentation.tex` |

Keep `figures/`, `frontmatter/`, `sections/`, `appendices/`, and `references.bib` together with the entry points. The `prism-uploads/` directory contains supporting files from the original export; its filenames are preserved as supplied.

For local compilation with a LaTeX distribution and latexmk, run from this directory:

```bash
latexmk -pdf -outdir=build report.tex
latexmk -pdf -outdir=build defense-presentation.tex
```

These commands use pdfLaTeX; the report uses BibTeX through natbib. Output goes to `build/` to preserve the supplied PDFs. The build directory is ignored by Git.

Upload validation confirmed that all copied files match their originals, both PDFs have PDF headers, and literal local LaTeX input, bibliography, and image references resolve. No LaTeX compiler was available locally, so compilation and exact PDF/source correspondence were not verified. The provided PDFs have not been regenerated.
