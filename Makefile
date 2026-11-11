# One target per demo step. Every target prints what to look at next.
# Settings come from config/demo.env (created by `make setup`, documented in the README).

ENV_NAME ?= candetect-demo
# Override with PY=python if the environment is already active.
PY ?= conda run --no-capture-output -n $(ENV_NAME) python
CONDA_RUN ?= conda run --no-capture-output -n $(ENV_NAME)

-include config/demo.env
export

CANDETECT_METAFLOW_ENV ?= fast-bakery
PR_ARGS ?=
RUN ?=
TRAIN_FLOW := flows/train/flow.py
GIT_BRANCH := $(shell git branch --show-current)
PROJECT := $(shell python3 -c "import tomllib;print(tomllib.load(open('obproject.toml','rb'))['project'])")

.PHONY: setup data train-local flow-local run-remote load-local compare promote build-image \
        deploy-dashboard pr-preview pr-preview-clean reset test lint

setup: ## Create the environment and config, check the platform connection
	@test -f config/demo.env || cp config/demo.env.example config/demo.env
	@conda env list | grep -q "^$(ENV_NAME) " || conda env create -y -f environment.yml
	@$(CONDA_RUN) python -W ignore -c "import torch, torchvision, onnxruntime, safetensors, metaflow; print('python deps ok, torch', torch.__version__)"
	@$(CONDA_RUN) outerbounds check
	@echo
	@echo "Next: edit config/demo.env if needed, then 'make data'."

data: ## Step 1a: generate the synthetic dataset and print its hash
	$(PY) -W ignore -m candetect.data
	@echo
	@echo "Next: open data/preview.png (top row normal, bottom row fallen can), then 'make train-local' or open notebooks/01_train_locally.ipynb."

train-local: ## Step 1b: run notebook 01 headless (dataset, training, ONNX export)
	$(CONDA_RUN) jupyter nbconvert --to notebook --execute --inplace --ExecutePreprocessor.timeout=3600 notebooks/01_train_locally.ipynb
	@echo
	@echo "Next: artifacts/local/ holds model.safetensors, model.onnx and summary.json."
	@echo "      Then 'make run-remote' to run the same code as a flow on the platform."

flow-local: ## Debug only: run the flow on this machine with a tiny configuration
	CANDETECT_LOCAL=1 $(PY) -W ignore $(TRAIN_FLOW) run --n-images 300 --epochs 1
	@echo
	@echo "Next: 'make run-remote' for the real run."

run-remote: ## Step 2-3: run the flow on platform compute
	$(PY) -W ignore $(TRAIN_FLOW) --environment=$(CANDETECT_METAFLOW_ENV) run --with kubernetes
	@echo
	@echo "Next: open the run link printed above in the Outerbounds UI and look at the 'summary' card:"
	@echo "      metrics, confusion matrix and the dataset sha256. Then 'make load-local'."

load-local: ## Step 4: run notebook 02 (load weights from a platform run, infer locally)
	$(CONDA_RUN) env CANDETECT_RUN=$(RUN) jupyter nbconvert --to notebook --execute --inplace notebooks/02_load_model_locally.ipynb
	@echo
	@echo "Next: 'make compare' to see all runs side by side."

compare: ## Step 5a: compare recent runs
	$(PY) -W ignore scripts/compare.py $(if $(RUN),$(RUN),)
	@echo
	@echo "Next: pick the best run and 'make promote RUN=<id>'."

promote: ## Step 5b: gate on the evaluation threshold, tag production, start the image build
	@test -n "$(RUN)" || (echo "usage: make promote RUN=<run id>"; exit 2)
	$(PY) -W ignore scripts/promote.py $(RUN)

build-image: ## Step 6: (re)start the edge image build for the current production model
	$(PY) -W ignore scripts/promote.py --build-only
	@echo
	@echo "Next: watch the build-image workflow in GitHub Actions, then 'make deploy-dashboard'."

deploy-dashboard: ## Step 8: deploy the monitoring dashboard (what CI does on main)
	@test "$(GIT_BRANCH)" = "main" || (echo "The dashboard deploys from main only (deployments/monitor/obproject_deploy.toml). You are on $(GIT_BRANCH)."; exit 2)
	$(CONDA_RUN) obproject-deploy --skip-flows --skip-assets
	@echo
	@echo "Next: open the app URL from 'outerbounds app info --name candetect-monitor --project $(PROJECT) --branch main'."

pr-preview: ## Rehearse ci.yml on this git branch: deploy a branch copy, train, compare with production
	@test "$(GIT_BRANCH)" != "main" || (echo "Switch to a feature branch first: this deploys a branch copy."; exit 2)
	$(CONDA_RUN) obproject-deploy --skip-apps --skip-assets
	$(PY) -W ignore scripts/pr_report.py --train-args "$(PR_ARGS)"
	@echo
	@echo "Next: build/pr-report/ holds report.md (the PR comment) and both model cards as HTML."
	@echo "      Clean up the branch deployment with 'make pr-preview-clean'."

pr-preview-clean: ## Tear down everything pr-preview deployed for this git branch
	$(CONDA_RUN) outerbounds flowproject teardown-branch --id $(PROJECT)/$(shell $(PY) -c "from candetect.config import project_branch; print(project_branch('$(GIT_BRANCH)'))") --yes

reset: ## Clear local outputs and the production tag so the demo can start over
	rm -rf artifacts data build
	$(PY) -W ignore scripts/promote.py --clear
	@echo
	@echo "Next: 'make data' to start again."

test: ## Run unit tests
	$(PY) -m pytest -q

lint: ## Run ruff with the repository config
	$(CONDA_RUN) ruff check src tests flows scripts deployments edge manual_tests
