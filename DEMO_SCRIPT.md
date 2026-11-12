# Demo script: dead plate detection on AI Workflow Orchestration

About 35 minutes plus questions. The story is one model followed from a laptop to production: train it locally, run the same code on the platform, promote it with an approval, build an edge image from the promotion, review a code change against production, and watch it in a dashboard.

Spoken lines are in quotes and commands are in code blocks. "You should see" is what the screen shows when it works, and every step ends with what to do if it does not.

## The night before and the hour before

Do not push code to `main` in the 90 minutes before the demo. A push redeploys production, and the event sensor that starts the image build takes 40 to 75 minutes to become active again. Events sent before that are lost. For documentation fixes use `[skip ci]` in the commit message.

Check these in order:

1. `pixi run setup` ends with "Validating metaflow token in config" and a tick.
2. `pixi run compare` shows exactly one run with `yes` in the production column, on branch `main`, and still lists the weak run `73006` as `refused` (the table shows the newest 10 runs, so `pixi run promote 73006` is the reliable way to show the refusal).
3. In the Outerbounds UI the project `can_detection_demo` shows branch `main` with two workflows and the `candetect-monitor` app.
4. Pull request 3 (`demo/fewer-epochs-v2`) has its comparison comment. Twenty minutes before you start, rerun its check so the comment is fresh: from that branch run `git commit --allow-empty -m "Rerun demo check" && git push`. It takes about 9 minutes and you will show it near the end.
5. `gh auth status` is logged in as you, and `config/demo.env` has `CANDETECT_GITHUB_REPO=philippjfr/can-demo`.
6. Open these tabs: the Outerbounds project page, the pull request, the dashboard (`https://ui-c-xwqk3b.gtm-01.outerbounds.xyz`, already logged in), the GitHub Actions page, and `docs/recorded/` in a file browser.
7. Terminal one in the repo root with `pixi run lab` running. Terminal two in the repo root for the platform commands.

Numbers to have in your head: 2,000 images at 128 by 128, dataset hash starts `ced936f2`, F1 0.993 at 6 epochs, 0.983 at 3 epochs, 0.766 for the weak model, promotion gate F1 0.90.

## Order of events and where the waiting hides

| Min | Step | Waiting you cover by talking |
| --- | --- | --- |
| 0 | Opening | |
| 2 | 1. Dataset and local training | Training takes 3 to 8 minutes. Start it, talk about the data, then show outputs. |
| 8 | 2. Same code as a flow | |
| 11 | 3. Run on the platform | Run takes about 6 minutes. Start it, then walk the flow code. |
| 17 | 4. Load the weights back | |
| 20 | 5. Compare, promote, image build | Image build takes about 4 minutes. Start it, then do step 6. |
| 25 | 6. Pull request | Already finished before you started. |
| 30 | 7. Dashboard | |
| 34 | Close | |

## Opening (2 minutes)

"A can line has a camera over the conveyor. When a can falls over it blocks the plate, and you want to know within seconds. Here the model is the easy part. What takes work is knowing which data trained which model, who approved it, how it gets to the edge, and how you notice when it goes stale."

"I'm using synthetic images so there is no customer data involved: cans from above are circles, a fallen can is an elongated shape lying across the plate, with noise, uneven lighting and blur. Everything runs on CPU and nothing here needs a GPU."

"Everything you see comes from one Git repository. I'll show it in the order a team would meet it."

## 1. Dataset and local training (6 minutes)

Do: in JupyterLab open `notebooks/01_train_locally.ipynb`. Start from the top and run the cells.

Say: "This is a normal notebook. The data generator is deterministic: the same seed gives the same 2,000 images on any machine. At the end of generation it computes a hash of the content. Keep that hash in mind: it starts `ced936f2`."

You should see: the preview image with normal plates on the top row and fallen cans below, then `dataset sha256: ced936f2033808af...`.

Say, while the training cell runs: "The model is MobileNetV3-Small, fine-tuned from ImageNet weights. If a plant has no internet access, one flag trains a small CNN from scratch instead, and the card says which one you got. Weights are saved with safetensors, not pickle, and the model is exported to a single ONNX file. The last cell checks that onnxruntime gives the same answers as PyTorch."

You should see: a test F1 around 0.99, a confusion matrix near `[[150, 0], [3, 147]]`, and `onnx parity: ... passed: True`.

If this fails live: do not retrain. Scroll through the committed notebook, which already holds the outputs of a real run, and say so. `docs/recorded/dataset_preview.png` has the preview.

## 2. The same code as a flow (3 minutes)

Do: put `notebooks/01_train_locally.ipynb` and `flows/train/flow.py` side by side.

Say: "Here is the same logic as a Metaflow flow. The flow calls `generate_dataset`, `make_splits`, `train`, `export_onnx` and `check_parity` from one shared module, `src/candetect`, and adds only steps, decorators and artifacts. There is no second copy of the training code to drift."

Point at: `@conda_env` (the environment is declared next to the code, resolved from Anaconda channels), `@resources(cpu=4, memory=8192)`, `ProjectFlow` at the top of the class, and the `self.weights = ...` line.

Say: "The weights are stored as plain bytes, and loading a model later never unpickles anything."

If this fails live: nothing runs in this step.

## 3. Run it on the platform (6 minutes, mostly overlap)

Do: in terminal two:

```
pixi run run-remote
```

Say: "This sends the same flow to the platform. The first thing it does is bake a container image from the environment declaration. That is cached, so it takes seconds, not minutes. Then a pod starts on Kubernetes, trains, and writes a card."

While it runs, open the project page in the UI. Say: "This is a project, with branches that mirror Git. My laptop runs show up under scratch, the pull request copies get their own branch, and production is `main`."

When it finishes you should see a line like `model 20261112-... f1=0.99` and a UI link. Open the run, then its `summary` card.

Say, pointing at the card: "Metrics and confusion matrix. The dataset hash is the same `ced936f2` as on my laptop, which makes this model traceable to exact data on any machine. ONNX parity is recorded. And here is a section comparing this run with whatever is in production right now."

If this fails live: open `docs/recorded/train_card.html` in a browser and say it is a saved card from the same flow. If a pod is slow to schedule, say so, keep talking through the card from an earlier run in the UI (any run of `TrainCanDetectorFlow`), and come back.

## 4. Load the weights back into a notebook (3 minutes)

Do: in JupyterLab open `notebooks/02_load_model_locally.ipynb`. It loads the newest successful run, which is the one from step 3 as long as the pull request check finished before you started. To pin a run, replace the `pathspec = ...` line with `pathspec = "TrainCanDetectorFlow/<run id>"`. Run all cells.

Say: "Now the other direction. A data scientist can take any run from the platform and use it locally through the Client API: weights, ONNX file, metadata. The notebook regenerates the dataset from the run's recorded config and asserts the hashes match. Then it runs inference with onnxruntime and PyTorch and compares them."

You should see: `identical data version`, probabilities that differ by about 1e-5, and local metrics equal to the recorded ones.

If this fails live: the committed notebook has outputs from a real run. Skip ahead.

## 5. Compare, promote, and the image build (6 minutes)

Do:

```
pixi run compare
```

Say: "Every run, the branch it ran on, its metrics, and which one is production. The weak one, 73006, is a small network trained from scratch for two epochs."

Do the refusal first:

```
pixi run promote 73006
```

You should see: `REFUSED: not promoted, no tags changed.` and `test F1 0.7657 is below the threshold 0.9000`.

Say: "The promotion script is the only way to tag a model production. It refuses anything below the evaluation threshold or without a passing ONNX parity check. Now the real run."

```
pixi run promote <run id from step 3>
```

You should see: `PROMOTED`, then tags `production`, `approved_by:<you>`, `approved_at:...`, `eval_f1:...`, and `retired production runs:` naming the previous one. Then a line saying the event was published and the GitHub workflow started.

Say: "Four things happened. The run got the production tag with who approved it and when. The old production run lost its tag. An event went out, and a GitHub workflow started."

Say, the honest part: "Since a tag cannot start anything on the platform, the script publishes an event and a flow listens for it. That flow checks provenance on the platform: it confirms the tag and approver, regenerates the dataset and compares the hash, and reproduces the F1 through the ONNX file. What the platform cannot do today is build a standalone container image with a model file and our labels inside it. So that step runs in GitHub Actions. I'd rather tell you that than hide it."

Show: the `BuildEdgeImageFlow` run card in the UI (build manifest), then the GitHub Actions run `build-image`. Point at the steps "Fetch the promoted model", "Check labels and run a prediction in the image", "Push", "Record the image on the model run". Go to step 6 while it runs, which takes 3 to 4 minutes, and come back.

When it is done, say: "The image carries the run id, the model version and the dataset hash as labels, and the same reference is now a tag on the run in the platform. From an edge device you can trace back to the data. The base image is a build argument, so a plant with its own hardened base can swap it."

If this fails live: if the event does not start the build flow, run `pixi run build-image`, which republishes it. If GitHub is slow or down, show `docs/recorded/promote.txt` and `docs/recorded/build_manifest_card.html`. If the promotion itself fails, `pixi run python scripts/promote.py <id> --dry-run` shows the decision without changing tags.

## 6. A code change against production (5 minutes)

Do: open pull request 3 in the browser. It changes `epochs` from 6 to 3 in `flows/train/flow.py`.

Say: "A teammate wants faster training and opens a pull request, but nothing in the diff says whether the model is still good."

Scroll to the comment and say: "CI deployed a copy of the project on its own branch, trained with the pull request's code, and compared the result with production. F1 went from 0.9933 to 0.9831. Recall dropped from 0.987 to 0.967, which for this problem means missed fallen cans. Same dataset hash, so the comparison is fair. It still passes the 0.90 gate, so this is a judgement call a reviewer can now make with numbers."

Open the Checks tab, then the `model-cards` artifact. Say: "Both full model cards are attached."

Say: "The machine that did this authenticates with GitHub's OIDC token. There is no personal token or secret stored in the repository. When the PR closes, the branch copy is torn down. When it merges to main, production is deployed from the same workflow."

If this fails live: open `docs/recorded/pr-preview/report.md` and its two card files, and say it is the same output from a rehearsal run. `pixi run pr-preview "--epochs 3"` produces the same thing from your laptop, but it takes about 6 minutes and needs a feature branch.

## 7. The monitoring dashboard (4 minutes)

Do: open the dashboard tab.

Say: "This app runs on the platform and reads the platform. Top left is the current production model with its tags and the edge image. Next to it, its held-out metrics. Below, metrics across runs with the promotion threshold as the dashed line: every run, from the weak one up to the ones that clear it."

Point at the recent predictions: "A sample of predictions on a fresh batch. Green frames are correct, red are wrong, with the confidence."

Point at the drift section, with the sidebar set to Shifted camera: "This batch comes from a camera that is darker, noisier and blurrier than the training data, which is what a dirty lens does. The drift indicator compares brightness, contrast and sharpness against the training distribution. It reads DRIFT with a PSI around 7, and accuracy on this batch is about 72 percent. The model is untouched and the only change is the camera."

Click Same as training. Say: "Same camera, and it goes back to OK. The indicator works without labels, which suits a plant where ground truth arrives days later."

If this fails live: show `docs/recorded/dashboard.png`. If the app is slow to open after a long idle, click Reload from platform once. The app can also run on your laptop with `pixi run dashboard`.

## Close (1 minute)

"One repository, one set of functions, from notebook to edge. Every model traces to its data by hash. Promotion records who approved it. A pull request shows its effect on production before merging. What changes for your plant is the dataset and the labels. Everything around it stays."

## Questions you will probably get

- "Is this running on GPUs?" No. Everything runs on CPU in minutes. GPU support exists but is optional, never required.
- "Why does GitHub build the image?" The platform bakes dependency images but cannot embed a model file and labels in a standalone image, and its images live in its own registry. The fallback is documented in the README.
- "Can we use our own channels and base image?" The base image is `CANDETECT_BASE_IMAGE` and the conda channels are `CANDETECT_CONDA_CHANNELS`. Whether the platform picks up organisation channels automatically is not tested here, so we pass them explicitly.
- "Is the data real?" No. It is generated, with a fixed seed.
- "Why one perimeter?" This deployment has a single perimeter, and project branches stand in for an experimentation and a production perimeter. With two, the workflows select one per job.
- "What are the licenses?" Only Apache, BSD or MIT, with no Ultralytics or AGPL code.

## If everything fails

Open `docs/recorded/` and walk through the saved material in order: `dataset_preview.png`, `train_card.html`, `compare.txt`, `promote.txt`, `build_manifest_card.html`, `pr-preview/report.md`, `dashboard.png`. Together they cover the whole demo with real output from earlier runs.

## After the demo

Promotion moved the production tag to your live run. To restore the baseline for the next run, promote the run that held it without publishing an event:

```
pixi run promote "argo-candetectiondemo.prod.traincandetectorflow-fxkmp --no-event"
```

`pixi run reset` clears all production and approval tags and the local outputs, which also removes the baseline the pull request compares against.
