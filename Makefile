.PHONY: install

install:
	Rscript -e "remotes::install_github('thisisamirv/fastLowess-R@develop')"
