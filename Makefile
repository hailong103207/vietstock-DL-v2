#################################################################################
# GLOBALS                                                                       #
#################################################################################
SHELL := /bin/bash
.ONESHELL:

#################################################################################
# VARIABLES                                                                     #
#################################################################################
PROJECT_NAME := vnstock-forecast
# Get conda environment name from environment.yml
ENV_NAME = $(shell grep 'name:' environment.yml | head -n1 | awk '{print $$2}')
# Activate conda environment
CONDA_ACTIVATE = source $$(conda info --base)/etc/profile.d/conda.sh ; conda activate $(ENV_NAME)

##################################################################################
# COMMANDS                                                                       #
##################################################################################

.PHONY: help setup update-env update-hooks pre-commit-test delete-env clean

help: ## Show help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "\033[36m%-30s\033[0m %s\n", $$1, $$2}'

setup: ## Set up the conda environment
	@echo "Creating conda environment '$(ENV_NAME)'..."
	conda env create -f environment.yml
	@echo "Setting up pre-commit ..."
	$(CONDA_ACTIVATE); pre-commit install
	$(CONDA_ACTIVATE); pre-commit install --hook-type commit-msg
	@echo "Setup succesfully, activate environment: conda activate $(ENV_NAME)"

update-env: ## Update the conda environment
	@echo "Updating conda environment '$(ENV_NAME)'..."
	conda env update -f environment.yml --prune

update-hooks: ## Update hooks from pre-commit
	@echo "Updating pre-commit hooks ..."
	$(CONDA_ACTIVATE); pre-commit autoupdate

pre-commit-test: ## Run pre-commit to fix code
	@echo "pre-commit checking ..."
	$(CONDA_ACTIVATE); pre-commit run --all-files

delete-env: ## Delete the conda environment
	@echo "Deleting conda environment '$(ENV_NAME)'..."
	conda env remove -n $(ENV_NAME)

clean: ## Clean up unnecessary files
	@echo "Cleaning up unnecessary files..."
	find . -type d -name '__pycache__' -exec rm -rf {} +
	find . -type f -name '*.pyc' -delete
	find . -type f -name '*.pyo' -delete
	rm -rf .pytest_cache
	rm -rf .ipynb_checkpoints
