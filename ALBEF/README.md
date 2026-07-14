# ALBEF model code

The code in this directory (`models/`, `configs/`) is third-party code, **not**
covered by the MIT license in the repository root.

- **Origin:** [salesforce/ALBEF](https://github.com/salesforce/ALBEF).
- **License:** BSD-3-Clause (full text in [`LICENSE`](./LICENSE)), except
  `models/xbert.py` and `models/tokenization_bert.py`, which are HuggingFace-derived
  and retain their own Apache-2.0 headers.
- **Modifications:** `models/xbert.py` has been modified for MM-SHAP — a
  `tokenizer_class=_TOKENIZER_FOR_DOC` argument (removed in `transformers > 4.11`,
  docs-only effect) was stripped so the code imports on modern `transformers`.
  Other files are unmodified.

The model **checkpoints** (`checkpoints/*.pth`) are large and are *not* vendored;
download them with `scripts/setup_albef.py` (see the top-level README).
