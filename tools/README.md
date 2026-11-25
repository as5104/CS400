# Report generation

Run these optional scripts from the project root. They do not train models.

- `build_dashboard_demo.py` creates the synthetic upload ZIP locally. Generate it before using the dashboard's example download; the archive itself is ignored by Git.

- `build_report_figures.py` reads saved CSV/JSON evidence under `output/final-results` and produces the three black-and-white result figures. Requires matplotlib.
- `build_report_flow.py` draws the executed eight-phase framework as a vector PDF. Requires reportlab.
- `build_latex_report.py` assembles `output/latex/CS400_Final_Report/main.tex` from the report text and the formal structure, copying the four PDF figures alongside it. It overwrites generated LaTeX; preserve any manual edits first.

Use the `reporting` optional dependencies for the figure scripts. The LaTeX source can be compiled with pdfLaTeX (twice), XeLaTeX, or Tectonic; upload the tracked `output/latex/CS400_Final_Report` folder to Overleaf if preferred. Compilation does not need Python or the original training data. The preferred way to make individual report edits is to edit `main.tex` directly. Compiled auxiliary files, generated plots, source ZIPs and duplicate report exports are ignored by Git. The selected final report PDF is tracked.

The frozen experiments and result records are the source of empirical claims. The diagram shows the executed 4/2/1 cohort schedule, while the earlier planning document retains its initial 12/6/3 pilot schedule.
