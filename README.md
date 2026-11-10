# Dead plate detection on a can line: Anaconda AI Orchestration demo

A synthetic top-down view of a conveyor plate is classified as `normal` or `fallen_can`. The demo follows one model from a laptop notebook to a production tag, an edge container image and a monitoring dashboard, on Anaconda AI Orchestration (Outerbounds and Metaflow). Everything runs on CPU in a few minutes and uses only Apache, BSD or MIT dependencies. There is no Ultralytics or other AGPL code.

PLAN.md records what was verified against the installed CLIs and the docs before any code was written, and the decisions that followed.

## Single perimeter, so `@project` branches stand in for two perimeters

The deployment this was built against (gtm-01) exposes one perimeter, `default`. There is no separate experimentation and production perimeter, so `@project` branches play both roles:

| Role | `@project` branch | Written by |
| --- | --- | --- |
| Experiments, laptop runs | `user.<you>` | `make run-remote` |
| Pull request copies | `test.pr<N>` | `ci.yml` |
| Production | `prod` | `deploy.yml` |

If two perimeters are available, set `CANDETECT_PERIMETER_EXPERIMENT` and `CANDETECT_PERIMETER_PROD` and the workflows select one per job. `--production` still applies.

## What was verified, and what was not

Run against the real platform and GitHub (repository `philippjfr/can-demo`): the training flow and its card, loading a platform run in a notebook, compare and promote (including the refusal), the promotion event starting the deployed build flow, the dashboard app deployment, and all three GitHub workflows. That includes OIDC authentication with both machine users, the production deployment from `deploy.yml`, a pull request that trained on a branch deployment and received the comparison comment and model cards, and `build-image.yml`, which built the edge image, checked its labels, ran a prediction inside it, pushed it to GHCR and recorded it on the model run.

The dashboard was tested against live platform data locally and in a browser. The deployed app uses Browser SSO, so it could not be opened headless. Its startup log (`outerbounds app logs --name candetect-monitor --project can_detection_demo --branch prod --worker-id <id>`) confirms that the Metaflow Client API works from inside the app container: `read 4 runs from platform; model run TrainCanDetectorFlow/73007`. The page itself has not been viewed in a browser on the platform.

Not verified: a local `docker build` (the image was built by GitHub Actions instead), and whether `@anaconda_base` uses organisation channels automatically.

## Setup

```
make setup
```

This creates the `candetect-demo` conda environment (Python 3.12, conda-forge), copies `config/demo.env.example` to `config/demo.env`, and runs `outerbounds check`. The environment is separate from any shared one because `onnx` needs a newer protobuf than many other packages pin. Pass `PY=python` to the Makefile if you have already activated an environment.

## Configuration

All deployment-specific values come from environment variables, loaded from `config/demo.env` by the Makefile.

| Variable | Default | Meaning |
| --- | --- | --- |
| `CANDETECT_METAFLOW_ENV` | `fast-bakery` | Metaflow `--environment` for platform runs |
| `CANDETECT_CONDA_CHANNELS` | `https://repo.anaconda.com/pkgs/main` | Comma separated channels for `@anaconda_base` |
| `CANDETECT_PERIMETER_EXPERIMENT`, `CANDETECT_PERIMETER_PROD` | `default` | Perimeter names used by the workflows |
| `CANDETECT_MIN_F1` | `0.90` | Promotion gate on test F1 |
| `CANDETECT_APPROVER` | your Metaflow username | Recorded as `approved_by` |
| `CANDETECT_BASE_IMAGE` | `python:3.12-slim-bookworm` | Base image of the edge image |
| `CANDETECT_REGISTRY` | `ghcr.io/<repo owner>` in CI | Where `build-image.yml` pushes |
| `CANDETECT_GITHUB_REPO` | `philippjfr/can-demo` | `owner/repo`, used by `promote.py` to start `build-image.yml` |
| `CANDETECT_APP_DOMAIN` | `gtm-01.outerbounds.xyz` | Domain suffix of app URLs, for the dashboard websocket origin |

`CANDETECT_CONDA_CHANNELS` must be set to your organisation's channel if you do not want the public `main` channel. Whether `@anaconda_base` picks up organisation channels automatically was not confirmed, so the channel is always passed explicitly.

### GitHub repository variables

The workflows read these repository variables (Settings, Secrets and variables, Actions, Variables). No secrets are needed.

| Variable | Value |
| --- | --- |
| `OB_DEPLOYMENT_DOMAIN` | `gtm-01.outerbounds.xyz` (derived from the API host `api.gtm-01.outerbounds.xyz`; confirm in the UI) |
| `OB_MACHINE_USER_PR` | Machine user restricted to `pull_request` runs of this repository |
| `OB_MACHINE_USER_MAIN` | Machine user restricted to this repository's `main` branch |
| `OB_PERIMETER_EXPERIMENT`, `OB_PERIMETER_PROD` | Optional, default `default` |
| `CANDETECT_REGISTRY`, `CANDETECT_BASE_IMAGE`, `CANDETECT_CONDA_CHANNELS` | Optional |

Create the machine users in the Outerbounds UI (Users, Machines, Create New) with the GitHub organisation and repository, and a branch or workflow restriction. Workflows authenticate with `outerbounds service-principal-configure --github-actions`, which exchanges GitHub's OIDC token. No personal tokens are stored. Only the `main` machine user should ever deploy production: the first production deploy of a flow issues a production token, and any other user would need it passed with `--authorize`. For the same reason do not run `--production` deployments from a laptop.

## Runbook

### 1. Generate the dataset and train locally

```
make data
make train-local        # or open notebooks/01_train_locally.ipynb and run it
```

Expected output of `make data`:

```
generated 2000 images (128, 128, 3) in 6.0s
fallen share: 0.50
dataset sha256: ced936f2033808af0715c636ebf61d8c650fba793d44902618eb0a08d857bc02
preview: data/preview.png
```

Look at `data/preview.png` (top row normal, bottom row fallen can), then the notebook: dataset hash, a test confusion matrix such as `[[150, 0], [3, 147]]` (F1 about 0.99), and an ONNX parity line with `passed: True`. On a laptop training takes one to five minutes depending on load. Files land in `artifacts/local/`.

The dataset hash is the same on every machine (verified between an arm64 laptop and the x86 platform). Trained metrics differ in the third decimal between machines because floating point training is not bit-identical.

If this fails live: `notebooks/01_train_locally.ipynb` is committed with its outputs from a real run. Open it without running it. If the MobileNet weights cannot be downloaded, training falls back to a small CNN from scratch and prints a warning; the card and metrics name the architecture used.

### 2. The same logic as a flow

Open `flows/train_flow.py` next to the notebook. The flow imports the same `generate_dataset`, `make_splits`, `train`, `export_onnx` and `check_parity` from `src/candetect/` and calls them in the same order. The differences are steps, decorators, artifacts and the card. Artifacts are plain bytes and dicts: weights are safetensors bytes, never a pickled model.

If this fails live: the code is the demo here, nothing needs to run.

### 3. Run it on the platform

```
make run-remote
```

This runs `python flows/train_flow.py --environment=fast-bakery run --with kubernetes`. Expected: an image bake (about 30 seconds, cached afterwards), a pod that can wait a minute or two to be scheduled, then roughly 30 seconds of training on 4 CPUs, ending with `model 20261103-<run id> f1=0.9933`. The log line `dataset sha256: ced936f2...` matches the laptop.

Look at the run in the Outerbounds UI and open the card `summary`: test metrics, confusion matrix, dataset sha256, ONNX parity, training history and a sheet of sample predictions with green or red borders. `docs/recorded/train_card.html` is a saved copy.

If this fails live: open `docs/recorded/train_card.html` in a browser, or pick any earlier run in the UI. Run the first remote run once before the demo so the image is cached.

### 4. Load platform weights in a notebook

```
make load-local RUN=TrainCanDetectorFlow/73007    # RUN is optional, default is the newest run
```

Open `notebooks/02_load_model_locally.ipynb`. It loads weights and the ONNX file through the Metaflow Client API, regenerates the dataset from the run's recorded config, asserts the hash matches, and compares onnxruntime and PyTorch probabilities (difference about 1e-5). Local metrics reproduce the recorded ones exactly.

If this fails live: the notebook is committed with outputs from a real run.

### 5. Compare runs and promote one

```
make compare
make promote RUN=<id>
```

`docs/recorded/compare.txt` shows `make compare` output. Promote the weak run first to show the gate, for example the 2-epoch small CNN (`python flows/train_flow.py --environment=fast-bakery run --with kubernetes --arch small_cnn --epochs 2`):

```
Run TrainCanDetectorFlow/73006: F1 0.7657, threshold 0.90
REFUSED: not promoted, no tags changed.
  - test F1 0.7657 is below the threshold 0.9000
```

Then the good run:

```
PROMOTED
  tags added: production, approved_by:<you>, approved_at:2026-11-03T16:24:50Z, eval_f1:0.9933
  retired production runs: none
```

The gate also refuses runs that did not finish or whose ONNX parity did not pass. The new tag is added before the old one is removed, so one production run exists at all times. The tags are visible on the run in the UI and through `Run.tags`. `docs/recorded/promote.txt` has both outputs.

If this fails live: show `docs/recorded/promote.txt`. `python scripts/promote.py <id> --dry-run` shows the decision without changing tags.

### 6. The promotion starts an edge image build

A tag cannot start anything on the platform (the documented triggers are events, flow completion and schedules). So `promote.py` publishes a project namespaced event, `mfns.can_detection_demo.prod.candetect_model_promoted`, and `flows/build_image_flow.py` listens for it with `@trigger`. That flow checks on the platform that the run is production tagged and approved, regenerates the data and compares the hash, reproduces the recorded F1 through the ONNX file, and writes a build manifest card (`docs/recorded/build_manifest_card.html`).

#### Image build fallback

The platform has no supported way to build a standalone image with a model file and custom labels. `metaflow.apps.bake_image` only installs dependencies and its images live in the platform registry. So the image itself is built by GitHub Actions. This is a fallback and is documented as one:

1. `promote.py` runs `gh workflow run build-image.yml` with the presenter's own `gh` login after publishing the event. No long-lived token is stored anywhere.
2. `build-image.yml` authenticates to the platform with the `main` machine user over OIDC, runs `python -m candetect.publish fetch` (which refuses a run without the `production` tag or with a different dataset hash), builds `edge/Dockerfile`, checks the labels, runs a prediction inside the image, pushes it, and writes `edge_image:<ref>` back onto the model run as a tag.
3. The image has `BASE_IMAGE` as a build arg, contains onnxruntime, `model.onnx` and `serve.py`, and carries the labels `com.candetect.run-id`, `com.candetect.model-version`, `com.candetect.dataset-hash` and `com.candetect.base-image`.

Run an edge container with `docker run -p 8080:8080 <image>`, then `curl -X POST --data-binary @sample.png localhost:8080/predict`. `docker run <image> info` prints the provenance.

```
make build-image    # redo the build for the current production run
```

Two platform details found while building:

- onnxruntime requests an executable stack, which glibc 2.41 (the platform's Debian 13 base image) refuses to load. The flows set `GLIBC_TUNABLES=glibc.rtld.execstack=2` on the steps that import it, and the Dockerfile sets it too so a Debian 13 base image also works.
- After `argo-workflows create` the event sensor takes a long time to become active. On the test branch it was a few minutes. For the production deployment, events published in the first 30 minutes or so were lost, and the first one that worked came about 35 minutes after the deploy. Events published while the sensor is not active are not queued. Deploy production well ahead of the demo, and do not redeploy shortly before it: any push to `main` runs `deploy.yml` and redeploys production, which resets the sensor. Use `[skip ci]` in commit messages for documentation-only changes.

If this fails live: show `docs/recorded/build_manifest_card.html` and the `edge/Dockerfile` labels, and walk through `.github/workflows/build-image.yml`.

### 7. A pull request trains and compares against production

Repository: `philippjfr/can-demo`. Set the variables above and create the two machine users first.

Open a pull request that changes the training code, for example lowering `epochs` in `flows/train_flow.py` from 6 to 3, and `ci.yml` does the following:

1. Runs ruff and the tests.
2. Deploys branch copies of both flows as `test.pr<N>`, tagged with the commit and the CI run.
3. Triggers a full training run on the branch copy, so the PR's code is what trains, and waits for it.
4. Compares that run with the run tagged `production` and posts the result as a PR comment (edited in place on later pushes). The comment has the metric deltas, whether both runs used the same dataset hash, the promotion gate result, and a better, worse or no-change verdict.
5. Attaches both model cards as HTML (the `model-cards` artifact). The run's own card on the platform also has a "Compared with the production model" section.
6. Deletes the branch copies when the PR closes. Fork PRs run tests only because they get no OIDC token.

Example comment from a real rehearsal (3 epochs against production at 6), saved in `docs/recorded/pr-preview/`:

```
| | This PR dapqv-9mt94 | Production 73007 | Change |
| F1 | 0.9831 | 0.9933 | -0.0102 |
| Recall | 0.9667 | 0.9867 | -0.0200 |
| Epochs | 3 | 6 | |
| Dataset hash | ced936f203 | ced936f203 | |
- Promotion gate: passes (test F1 0.9831 against 0.90).
- Against production: worse, F1 -0.0102.
```

The comparison needs a run tagged `production`, so promote one first (step 5). Without one, the comment says there is nothing to compare with.

Merging to `main` runs `deploy.yml`: tests, then `--production argo-workflows create` for both flows. `build-image.yml` is described in step 6.

```
make pr-preview PR_ARGS="--epochs 3"    # the same sequence from your laptop, no GitHub needed
make pr-preview-clean                   # delete the preview deployments afterwards
```

`pr-preview` deploys branch `preview`, trains, and writes `build/pr-report/` with `report.md` and both cards. It takes about 6 minutes, most of it pod scheduling and training.

`obproject-deploy` from the Outerbounds docs is an alternative to the explicit `argo-workflows create` commands used here.

If this fails live: show `docs/recorded/pr-preview/report.md` and the two cards next to it, or run `make pr-preview` ahead of time. Nothing in the other steps depends on this one.

### 8. Monitoring dashboard

```
make deploy-dashboard
```

This stages `app.py` and the `candetect` package into `build/dashboard` and runs `outerbounds app deploy`. The dashboard (Panel and hvPlot) shows the production model with its tags, test metrics, metrics across runs with the promotion threshold, a sample of predictions on a fresh batch, and a drift indicator: population stability index on brightness, contrast and sharpness of a deliberately shifted synthetic batch against the training distribution. Switch the incoming batch to "same as training" in the sidebar to see the indicator return to OK. The app uses Browser authentication, so only people with access to the Outerbounds UI can open it. The URL is printed at the end of the deploy. The first start can take several minutes when the cluster is short of capacity.

To run it locally: `panel serve dashboard/app.py --dev --show`. With `DASHBOARD_MOCK=1` it reads the notebook 01 output in `artifacts/local` instead of the platform.

If this fails live: `docs/recorded/dashboard.png` is a screenshot, and the local `panel serve` works with live platform data or `DASHBOARD_MOCK=1`.

### Reset

```
make reset
```

Removes `data/`, `artifacts/` and `build/`, and clears the `production` and approval tags on all runs. Platform runs and deployments are not deleted. To remove a branch deployment: `python flows/train_flow.py --environment=fast-bakery --branch <name> argo-workflows delete`. To remove the dashboard: `outerbounds app delete --name candetect-monitor --project can_detection_demo --branch prod`.

## Tests and lint

```
make test    # 63 tests: data, model, safetensors, ONNX parity, training, promotion, drift, edge entrypoint
make lint    # ruff with the repository config
```

## Layout

```
src/candetect/   data, model, train, export, inference, promotion, drift, publish, config
flows/           train_flow.py, build_image_flow.py, _env.py
scripts/         compare.py, promote.py, pr_report.py
notebooks/       01_train_locally.ipynb, 02_load_model_locally.ipynb
dashboard/       app.py, config.yml, requirements.txt
edge/            Dockerfile, serve.py
docs/recorded/   saved cards, screenshot and console output for when a live step fails
.github/         workflows, shared setup action
manual_tests/    throwaway platform probes (onnxruntime import in a pod)
```

`make flow-local` runs the training flow on the laptop with a tiny configuration for debugging. It sets `CANDETECT_LOCAL=1`, which skips the conda environment declaration.
