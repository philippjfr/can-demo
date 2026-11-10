<!-- candetect-pr-report -->
## Model comparison: pull request against production

Branch deployment `test.prdemo` trained `TrainCanDetectorFlow/argo-candeondemo.test.prdemo.traincatorflow-dapqv-9mt94` with this PR's code.

| | This PR [`dapqv-9mt94`](https://ui.gtm-01.outerbounds.xyz/p/default/TrainCanDetectorFlow/argo-candeondemo.test.prdemo.traincatorflow-dapqv-9mt94) | Production [`73007`](https://ui.gtm-01.outerbounds.xyz/p/default/TrainCanDetectorFlow/73007) | Change |
| --- | --- | --- | --- |
| F1 | 0.9831 | 0.9933 | -0.0102 |
| Accuracy | 0.9833 | 0.9933 | -0.0100 |
| Precision | 1.0000 | 1.0000 | +0.0000 |
| Recall | 0.9667 | 0.9867 | -0.0200 |
| Architecture | mobilenet_v3_small | mobilenet_v3_small | |
| Epochs | 3 | 6 | |
| Images | 2000 | 2000 | |
| Dataset hash | `ced936f203` | `ced936f203` | |
| ONNX parity | passed | passed | |

- Promotion gate: **passes** (test F1 0.9831 against 0.90).
- Against production: **worse**, F1 -0.0102.

The full model cards of both runs are attached to the workflow run as the `model-cards` artifact and the run is at https://ui.gtm-01.outerbounds.xyz/p/default/TrainCanDetectorFlow/argo-candeondemo.test.prdemo.traincatorflow-dapqv-9mt94.