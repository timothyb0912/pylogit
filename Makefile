## install     : Install project package locally and install pre-commit.
.PHONY : install
install :
	pip install pip-tools
	pip-compile requirements.in
	pip install -r requirements.txt

## help        : Documentation for make targets.
.PHONY : help
help : Makefile
	@sed -n 's/^##//p' $<
