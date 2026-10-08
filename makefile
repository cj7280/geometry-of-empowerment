FIGURES ?=

.PHONY: all test recompute-fig4

all:
	uv run python make_all_figures.py $(FIGURES)

test:
	uv run pytest
	uv run ruff check .

recompute-fig4:
	uv run python make_all_figures.py 4 --recompute-fig4
