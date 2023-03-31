SPHINXOPTS    = -nWT --keep-going
SPHINXBUILD   = sphinx-build
PAPER         =
MPROF         = SG_STAMP_STARTS=true mprof run -E --python sphinx

# Internal variables.
PAPEROPT_a4     = -D latex_paper_size=a4
PAPEROPT_letter = -D latex_paper_size=letter
ALLSPHINXOPTS   = -d _build/doctrees $(PAPEROPT_$(PAPER)) $(SPHINXOPTS) .

.PHONY: help clean html dirhtml pickle json htmlhelp qthelp latex changes linkcheck doctest

# make with no arguments will build the first target by default, i.e., build standalone HTML files
first_target: html-noplot

help:
	@echo "Please use \`make <target>' where <target> is one of"
	@echo "  html             to make standalone HTML files (dev version)"
	@echo "  html-pattern     to make standalone HTML files for one example dir (dev version)"
	@echo "  *-noplot         to make standalone HTML files without plotting"

clean:
	-rm -rf _build auto_examples auto_tutorials generated *.stc *.fif *.nii.gz

html:
	$(SPHINXBUILD) -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html."

html-pattern:
	$(SPHINXBUILD) -D sphinx_gallery_conf.filename_pattern=$(PATTERN) -D sphinx_gallery_conf.run_stale_examples=True -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html"

html-noplot:
	$(SPHINXBUILD) -D plot_gallery=0 -b html $(ALLSPHINXOPTS) _build/html
	@echo
	@echo "Build finished. The HTML pages are in _build/html."

view:
	@python -c "import webbrowser; webbrowser.open_new_tab('file://$(PWD)/_build/html/index.html')"

show: view
