# One target per demo step. Every target prints what to look at next.
# Settings come from config/demo.env (created by `make setup`, documented in the README).

ENV_NAME ?= candetect-demo
# Override with PY=python if the environment is already active.
PY ?= conda run --no-capture-output -n $(ENV_NAME) python
CONDA_RUN ?= conda run --no-capture-output -n $(ENV_NAME)

-include config/demo.env
export

CANDETECT_METAFLOW_ENV ?= fast-bakery
CANDETECT_APP_DOMAIN ?= gtm-01.outerbounds.xyz
DASHBOARD_BRANCH ?= prod
RUN ?=
TRAIN_FLOW := flows/train_flow.py

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

deploy-dashboard: ## Step 8: deploy the monitoring dashboard as a platform app
	rm -rf build/dashboard && mkdir -p build/dashboard
	cp dashboard/app.py dashboard/requirements.txt build/dashboard/
	cp -R src/candetect build/dashboard/candetect && find build/dashboard -name __pycache__ -prune -exec rm -rf {} +
	$(CONDA_RUN) outerbounds app deploy --config-file dashboard/config.yml \
	  --package-src-path build/dashboard --dep-from-requirements build/dashboard/requirements.txt \
	  --python 3.12 --project can_detection_demo --branch $(DASHBOARD_BRANCH) --cpu 1 --memory 2Gi \
	  --env "DASHBOARD_WS_ORIGIN=*.$(CANDETECT_APP_DOMAIN)" \
	  --env "GLIBC_TUNABLES=glibc.rtld.execstack=2" \
	  --readiness-condition at_least_one_running --force-upgrade
	@echo
	@echo "Next: open the app URL printed above (Outerbounds login required)."

PR_BRANCH ?= preview
PR_ARGS ?=

pr-preview: ## Rehearse ci.yml for a pull request: branch deploy, train, compare with production
	for flow in train_flow build_image_flow; do \
	  $(PY) -W ignore flows/$$flow.py --environment=$(CANDETECT_METAFLOW_ENV) --branch $(PR_BRANCH) argo-workflows create || exit 1; \
	done
	$(PY) -W ignore scripts/pr_report.py --branch $(PR_BRANCH) --train-args "$(PR_ARGS)"
	@echo
	@echo "Next: build/pr-report/ holds report.md (the PR comment) and both model cards as HTML."
	@echo "      Clean up the branch deployments with 'make pr-preview-clean'."

pr-preview-clean: ## Delete the branch deployments made by pr-preview
	for flow in train_flow build_image_flow; do \
	  $(PY) -W ignore flows/$$flow.py --environment=$(CANDETECT_METAFLOW_ENV) --branch $(PR_BRANCH) argo-workflows delete; \
	done

reset: ## Clear local outputs and the production tag so the demo can start over
	rm -rf artifacts data build
	$(PY) -W ignore scripts/promote.py --clear
	@echo
	@echo "Next: 'make data' to start again."

test: ## Run unit tests
	$(PY) -m pytest -q

lint: ## Run ruff with the repository config
	$(CONDA_RUN) ruff check src tests flows scripts dashboard edge
