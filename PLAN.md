# Plan: dead plate detection demo on Anaconda AI Orchestration

Status: verification complete, awaiting review. No application code has been written.

## 1. How verification was done

- Local CLIs: `metaflow` 2.19.34.1 (ob-metaflow 2.19.34.1, ob-metaflow-extensions 1.6.24) and `outerbounds` 0.12.39, inspected with `--help` and by reading decorator docstrings from the installed packages.
- Docs: docs.outerbounds.com, docs.metaflow.org and the Anaconda platform docs, via targeted search because the docs root page exceeds the fetch limit.
- Package availability: Anaconda package intelligence for linux-64 on the `main` channel.
- Nothing was deployed, triggered or modified on the platform. The local Metaflow config was only read for non-secret keys.

Tag legend: `[CLI]` confirmed from installed `--help` or docstrings, `[DOCS]` confirmed from documentation, `[OPEN]` could not be confirmed.

## 2. Findings

### 2.1 Environment declaration

- `[CLI]` Available top-level `--environment` values: `local, uv, anaconda, fast-bakery, conda, pypi`.
- `[CLI]` Step decorators: `@pypi`, `@conda`, `@anaconda`. Flow decorators: `@pypi_base`, `@conda_base`, `@anaconda_base`.
- `[CLI]` `@anaconda` is "identical to `@conda` but defaults to the `anaconda` channel" and takes `packages`, `python`, `disabled`, `channels` (default `["https://repo.anaconda.com/pkgs/main"]`).
- `[DOCS]` With `--environment=fast-bakery`, `@pypi` and `@conda` are baked into a container image by the platform and cached.
- `[DOCS]` `@conda` accepts `channel::package` syntax and a `CONDA_CHANNELS` setting.
- `[OPEN]` Whether `@anaconda` resolves from the organisation's configured channels (for example `repo.anaconda.cloud/repo/<org>/...`) rather than the public default. The decorator exposes `channels=`, but nothing I read says it picks up org channels automatically. The local platform config sets no `CONDA_CHANNELS`.
- `[DOCS]` On linux-64, `@pypi` torch wheels pull the CUDA stack. The `main` channel has CPU builds (`pytorch=*=cpu_mkl*`, `torchvision=*=cpu*`), so the conda route is the better fit for CPU-only runs.
- All needed packages exist on `main` for linux-64: pytorch 2.15, torchvision 0.29, onnx 1.22, onnxscript 0.7, onnxruntime 1.30, safetensors 0.8, panel 1.10, hvplot 0.12, panel-material-ui 0.16. Licenses are BSD, MIT or Apache. Panel 1.10 has one non-critical CVE.

Decision: `@anaconda_base` with `fast-bakery`, Python 3.12, channel list read from `CANDETECT_CONDA_CHANNELS` with the `main` channel as default. Open question 1 below decides the channel URL.

### 2.2 Platform compute, deployment, `@project`

- `[CLI]` Run on platform compute: `python flow.py --environment=<env> run --with kubernetes`. Step-level `@kubernetes(cpu, memory, disk, image, ...)` exists.
- `[CLI]` Deploy: `python flow.py argo-workflows create`, then `argo-workflows trigger [--run-id-file F]`. `create` also supports `--tag`, `--authorize`, `--generate-new-token`, `--auto-emit-argo-events` (default on).
- `[CLI]` `@project(name=...)` adds top-level `--branch` and `--production`. Resulting branch names: `user.<name>` by default, `test.<branch>` with `--branch`, `prod` with `--production`, `prod.<branch>` with both.
- `[CLI]` `outerbounds perimeter list` on this machine shows exactly one perimeter, `default`.

Decision: single perimeter, so `@project` branches stand in for the experimentation and production perimeters. The README will say so. Everything perimeter-specific comes from `CANDETECT_PERIMETER_EXPERIMENT` and `CANDETECT_PERIMETER_PROD`, which are both `default` unless set. If two perimeters exist the CI workflows select one per job and `--production` still applies.

### 2.3 Tags and triggers

- `[CLI]` `tag add/list/remove/replace` exist. The client has `Run.add_tag`, `replace_tag`, `user_tags`, `system_tags`, and `Flow.runs(*tags)` filters by tag.
- `[DOCS]` The documented triggers on Argo are `@trigger` (named events), `@trigger_on_finish` and `@schedule`. I found no mechanism where a tag starts anything.
- `[CLI]` `ArgoEvent(name, payload).publish()` exists and works from any environment with a valid Metaflow config. `ARGO_EVENTS_WEBHOOK_URL` is set in the local platform config, so it should work from a laptop and from CI after service-principal login.
- `[OPEN]` Exact resolved name of `namespaced_event_name('x')` under `@project`. The promotion script must publish the identical string. I will read it from the deployed workflow template and from `current.trigger.event` in step 6 instead of guessing the format. I will avoid un-namespaced names because they would also fire the PR branch copy.

Decision: `promote.py` adds the `production` tag, then publishes a project-namespaced event, and `build_image_flow.py` listens with `@trigger`.

### 2.4 Standalone container image build

- `[CLI]` `metaflow.apps.bake_image(pypi, conda, requirements_file, pyproject_toml, base_image, python, logger, cache_name)` returns an image URL plus Python path. It installs dependencies only. There is no parameter for adding files, `COPY`, labels or a Dockerfile.
- `[DOCS]` Fast Bakery stores images in the platform's own registry (object-store backed, zstd OCI layers) and is meant for platform execution, not for exporting images to edge hosts.
- `[CLI]` `outerbounds app deploy --image` consumes an existing image. It does not produce one.
- `[OPEN]` I found no supported way to build a standalone image with an embedded ONNX file and custom labels on the platform. I am treating this as "does not exist" per your instruction, but please confirm if you know of one.

Decision (fallback, documented as such in the README): `build_image_flow.py` runs on the platform, triggered by the event. It pulls the promoted run's ONNX file and metadata, validates parity again, then asks GitHub Actions to build and push the image using `.github/workflows/build-image.yml` and `edge/Dockerfile` (base image as `--build-arg`, labels for run ID, model version, dataset hash). Registry defaults to GHCR with the workflow's own `GITHUB_TOKEN`, so no registry secret is needed.
How the platform flow asks GitHub to start the workflow is open question 3.

### 2.5 GitHub Actions authentication

- `[CLI]` `outerbounds service-principal-configure` flags: `--name`, `--deployment-domain`, `--perimeter`, `--github-actions`, `--jwt-token`, `--from-obproject-toml`, `--profile`, `--force`.
- `[DOCS]` With `--github-actions` it reads `ACTIONS_ID_TOKEN_REQUEST_TOKEN` and `ACTIONS_ID_TOKEN_REQUEST_URL`, so the job needs `permissions: id-token: write`. No stored token.
- `[DOCS]` The machine user is created in the Outerbounds UI (Users, Machines, Create New) with GitHub org, repo and optionally branch, workflow, environment. The platform validates those claims.
- `[DOCS]` Documented example names the machine user `<repo>-cicd`.
- `[DOCS]` The documented pipeline uses `obproject.toml` plus `obproject-deploy` (package `ob-project-utils`). I did not inspect that package, so I will deploy with the verified `argo-workflows create` commands directly. `obproject-deploy` stays a documented alternative in the README.

Decision: one machine user per perimeter, in the UI, restricted by branch or GitHub environment claims. With one perimeter I will create two machine users instead (PR-restricted and main-restricted) if you want the separation preserved. See open question 4. The machine users themselves must be created by you in the UI. I cannot do that from here.

### 2.6 Apps

- `[CLI]` `outerbounds app deploy` flags include `--config-file`, `--name`, `--port`, `--image`, `--auth-type [Browser|API|BrowserAndApi]`, `--public-access/--private-access`, `--secret`, `--env`, `--cpu`, `--memory`, `--dep-from-requirements`, `--dep-from-pyproject`, `--python`, `--package-src-path`, `--project`, `--branch`, `--generate-static-url`, `--readiness-condition`.
- `[DOCS]` Config file keys confirmed by examples: `name`, `port`, `image`, `commands`, `environment`, `auth.type`. The Panel docs give a working Outerbounds example: `panel serve app.py --port 6000 --allow-websocket-origin="*.outerbounds.com"` with `BOKEH_RESOURCES: cdn`.
- `[OPEN]` The dedicated config schema page returned 404. I will only use the keys above and the CLI flags confirmed from `--help`.
- `[OPEN]` Whether the Metaflow Client works inside the app container. Docs say apps run with the perimeter's default task role, which suggests yes. I will test at step 8 and, if it fails, have the dashboard read a metrics JSON artifact written by the train flow.
- `[OPEN]` The domain in `--allow-websocket-origin` for this deployment (`*.gtm-01.outerbounds.xyz` vs `*.outerbounds.com`). The doc notes the wildcard matches one level only. I will read the URL from the first deploy and set it via an env var.

### 2.7 Other platform facts used

- `[CLI]` `@card` with `current.card.append(...)`. I will use Markdown and Table components for metrics and the confusion matrix, plus a PNG `Image` for sample predictions. No plotting dependency on the platform.
- `[CLI]` Local config: datastore `s3`, metadata `service`, Kubernetes namespace `jobs-default`, default container registry a public ECR mirror.
- `[OPEN]` Whether pods can reach `download.pytorch.org` for pretrained MobileNetV3 weights. That is why the from-scratch fallback flag exists. I will test it on the platform and print a clear warning if the fallback triggers.

## 3. Design

### Layout and responsibilities

The layout is exactly as requested. All logic lives in `src/candetect/`:

- `data.py`: seeded dataset generator (cairo-free, NumPy and Pillow drawing), `dataset_hash()` (SHA-256 over image bytes, labels and generator config).
- `model.py`: `build_model(pretrained, fallback)`, small CNN fallback, safetensors save and load.
- `train.py`: `train(config) -> TrainResult` (weights, metrics, confusion matrix, dataset hash), `evaluate()`.
- `export.py`: `export_onnx()` as a single file (verified by checking the graph has no external data references), `check_parity()` with onnxruntime.
- `inference.py`: `load_from_run(run_id)`, `predict()` shared by notebook 2, dashboard and edge.

The notebook and the flow call the same `train()` and `export_onnx()`. The flow differs only by `@step`, decorators, artifacts and the card.

### Flow, promotion, build

- `train_flow.py`: `start` (generate and hash data), `train`, `evaluate_and_export` (metrics, parity, card), `end`. Artifacts: `dataset_hash`, `metrics`, `confusion_matrix`, `weights_bytes` (safetensors), `onnx_bytes`, `config`, `model_version`.
- `scripts/compare.py`: tabulates the last N runs from the Client API (F1, accuracy, dataset hash, tags).
- `scripts/promote.py RUN=<id>`: refuses if the run is unsuccessful, if the F1 is below `CANDETECT_MIN_F1` (default 0.90), or if ONNX parity was not recorded as passed. On success it moves the `production` tag (`replace_tag`), adds `approved_by:<user>` and `approved_at`, then publishes the event. Approver comes from `CANDETECT_APPROVER`, falling back to the Metaflow username. The threshold logic is a pure function so it can be unit tested.
- `build_image_flow.py`: `@trigger` on the promotion event, fetches the run, re-validates, requests the GitHub build, and records the image reference.
- `edge/Dockerfile`: `ARG BASE_IMAGE`, installs onnxruntime and numpy and Pillow, copies `model.onnx` and `serve.py`, sets `LABEL` for run ID, model version, dataset hash.
- `dashboard/app.py`: Panel with panel-material-ui, hvPlot for metrics and drift. Drift is a population stability index plus mean-intensity shift on a deliberately shifted synthetic batch against the training distribution.

### CI

- `ci.yml` on pull requests: install, ruff, pytest, then deploy a branch copy with `--branch pr<N> argo-workflows create`, comment the workflow name.
- `deploy.yml` on push to main: deploy with `--production argo-workflows create` for `train_flow` and `build_image_flow`, and `outerbounds app deploy` for the dashboard.
- `build-image.yml`: `workflow_dispatch` with inputs `run_id`, `model_version`, `dataset_hash`, `base_image`. Fallback only.

### Config

One file, `config/demo.env.example`, documented in the README. Variables: `CANDETECT_PERIMETER_EXPERIMENT`, `CANDETECT_PERIMETER_PROD`, `CANDETECT_PROJECT`, `CANDETECT_CONDA_CHANNELS`, `CANDETECT_BASE_IMAGE`, `CANDETECT_REGISTRY`, `CANDETECT_MIN_F1`, `CANDETECT_APPROVER`, `CANDETECT_ENV` (`anaconda` default). The Makefile loads it.

### Tests

`data` determinism and hash stability, model forward shape, safetensors round trip, ONNX single-file check and parity within tolerance, promotion threshold and tag logic against a fake client, drift indicator.

## 4. Build order and verification gates

1. Repo scaffold, config, `PLAN.md` (this file). Gate: your approval.
2. `data.py`, `model.py`, `train.py`, `export.py`, tests, notebook 1, `make data train-local`. Gate: runs on CPU in a few minutes, tests pass.
3. `train_flow.py` and `make run-remote`. Gate: a real run on the platform with a card showing metrics, confusion matrix and dataset hash. First local `run`, then `--with kubernetes`.
4. Notebook 2 and `make load-local`. Gate: weights loaded from the remote run, predictions identical to ONNX output.
5. `compare.py`, `promote.py`, `make compare promote RUN=<id>`. Gate: tag visible, threshold refusal demonstrated.
6. `build_image_flow.py`, `edge/`, `build-image.yml`. Gate: event name matches, flow starts after promotion, image built locally with `docker build` as a check.
7. CI workflows and machine-user setup. Gate: needs the repo and machine users (open questions 2 and 4).
8. Dashboard and `make deploy-dashboard`. Gate: app reachable, Client API works from inside the app or the artifact fallback is in place.
9. README runbook with expected output and an "if this fails live" note per step, `make reset`.

## 5. Resolved decisions (after review)

1. Target deployment: gtm-01, `default` perimeter, `@project` branches stand in for two perimeters.
2. Conda channel: public `main` (`https://repo.anaconda.com/pkgs/main`), overridable with `CANDETECT_CONDA_CHANNELS`.
3. Image build start: `promote.py` runs `gh workflow run build-image.yml` with the presenter's own `gh` login. The event-triggered `build_image_flow` validates the model and records the image build request; it does not hold a GitHub token.
4. CI machine users: two, one restricted to pull requests and one to main. You create them in the UI.
5. GitHub owner and repo name were not answered. `CANDETECT_GITHUB_REPO` stays unset until you tell me; the registry defaults to `ghcr.io/<owner>`.
6. Local environment: a repo-local conda environment (`environment.yml`, Python 3.12). The shared `dev3.14` env cannot hold `onnx` because it needs protobuf 6.31 or newer while other packages there pin protobuf below 6.

## 5b. Original open questions (kept for the record)

1. Which Outerbounds deployment is this demo for? The only perimeter on this machine is `default` on `gtm-01`. Is that the target, or does the Ardagh demo have its own deployment, perhaps with two perimeters? Also, which conda channel URL should `@anaconda` use (public default or an org channel)?
2. GitHub owner and repository name, and whether GHCR is acceptable for the edge image registry.
3. How should the platform-side `build_image_flow` start the GitHub workflow? Options: (a) a fine-scoped GitHub token stored as an Outerbounds secret and used with the GitHub API `workflow_dispatch`, or (b) `promote.py` itself runs `gh workflow run` from the presenter's machine using their own `gh` login, with the event-triggered flow only validating and recording. (a) is fully automatic but needs a long-lived token, which conflicts with the "no personal tokens" intent for CI. (b) avoids that but makes step 6 depend on the presenter's laptop.
4. Machine users: you need to create them in the UI. One user with no branch restriction, or two (PR and main) to mimic the two-perimeter separation on a single perimeter?

## 6. Risks

- Pretrained weights download may be blocked from pods. Mitigation: fallback flag, tested on platform.
- Fast Bakery resolution of the conda environment (torch CPU) may take several minutes on the first run. Mitigation: pre-run once before the demo, and the README says so.
- Event name mismatch between `promote.py` and `@trigger` under `@project`. Mitigation: read the resolved name from the deployed template.
- Dashboard Client API access from inside the app is unverified.

## 7. Outcome and deviations (written after the build)

What changed from the plan above, and what the platform actually did.

### Confirmed on the platform

- `@anaconda_base` with `--environment=fast-bakery` resolves the pinned conda packages from the public `main` channel and bakes an image in about 30 seconds. Mixing `@pypi` onto it is accepted at bake time. Organisation channels were not tested.
- `@kubernetes` via `--with kubernetes`, `@resources`, and `@card(type="blank")` work as expected. Pretrained MobileNetV3 weights download from inside a pod, so the from-scratch fallback was not needed on this deployment.
- A tag cannot trigger anything. The promotion event `mfns.<project>.<branch>.<event>` published with `ArgoEvent` from a laptop starts the `@trigger` flow. The deployed name matches `promotion_event_name()`.
- `outerbounds app deploy` with `--package-src-path`, `--dep-from-requirements`, `--env`, `--project` and `--branch` works. App URL: `https://ui-c-<id>.gtm-01.outerbounds.xyz`, so the websocket origin `*.gtm-01.outerbounds.xyz` is right.

### Problems found and how they were resolved

- Dataset hash differed between arm64 and x86 with float32 rendering. The generator now uses float64, and the hash is identical on laptop and platform.
- onnxruntime failed to import in pods (`cannot enable executable stack`) with the conda build (1.30.0, 1.24.4, 1.22.2) and with a PyPI wheel. A custom base image did not help because `python:3.12-slim` is also Debian 13 now. A runtime ELF patch did not work. `GLIBC_TUNABLES=glibc.rtld.execstack=2` on the step works. Probes are in `manual_tests/ort_probe.py`.
- The event sensor needed several minutes after `argo-workflows create` before events triggered runs.
- `shifted` drift batches and same-distribution batches are separated reliably only with a 1,000 image reference and a 400 image batch. At 400 and 200 images a same-distribution batch scored PSI up to 0.14.
- The cluster was short of memory when the dashboard app first deployed, so the pod waited several minutes. The app now requests 1 CPU and 2Gi.

### Deviations from the plan

- Fast-bakery is the default environment (not `anaconda`), per the docs for platform runs.
- `candetect/inference.py` is torch-free so the dashboard image has no torch. `fetch` and `tag-image` live in `candetect/publish.py`.
- Added `candetect/promotion.py`, `drift.py`, `onnx_check.py`, `publish.py`, a shared setup action for the workflows, `manual_tests/` and `docs/recorded/`.
- The GitHub owner and repository name were not provided, so `CANDETECT_GITHUB_REPO` is a placeholder and `promote.py` skips the GitHub build until it is set.

### Not verified

GitHub workflows end to end, `gh workflow run`, the `docker build`, and a `--production` deploy, because no repository, machine users or Docker daemon were available. Whether `@anaconda_base` uses organisation channels automatically. The dashboard page rendered on the platform: the app uses Browser SSO so it could not be opened headless (its startup log does confirm Client API access from inside the container).
