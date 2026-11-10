# Training data and attribution

This directory contains public research datasets used to train the local classifiers. Posts include hateful and offensive language.

- `hatexplain.json` and `hatexplain_splits.json`: [HateXplain](https://github.com/hate-alert/HateXplain), Mathew et al., *AAAI 2021*. The supplied train/validation/test IDs are used. Only posts with a majority label are included. The repository's license is copied in `HateXplain_LICENSE`; cite the paper when using the data.
- `dynahate.csv`: [Dynamically Generated Hate Speech Dataset v0.2.3](https://github.com/bvidgen/Dynamically-Generated-Hate-Speech-Dataset), Vidgen et al., *ACL 2021*. The authors release it under [CC BY 4.0](https://creativecommons.org/licenses/by/4.0/). The supplied train/dev/test split is used. The file is an unmodified copy; our training code selects only its text, label, and split columns.

The models are trained with `py -3.12 ml_pipeline/train.py`. The held-out HateXplain test split is excluded from fitting and threshold selection. Every model decision is learned from labeled data; the inference code contains no sentence-specific match or answer table.

SHA-256 checksums:

```text
hatexplain.json         63bb3340fee0ec469b09690d04cb68f7c187787dd8b83807f071892c084967fb
hatexplain_splits.json  c2fb0d89862e7897b11ea3e9380753f15a793482b4b70ad0532dfb1212212835
dynahate.csv            6e0aa7e598ab5b648cb1f0b48626a3e5f2e39a5abbd9fee998efa5a0b6cab4ca
```
