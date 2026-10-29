# IT550 Lab 10 — Semantic IR Using LSA and Word2Vec (202303022)

Cranfield collection: a TF-IDF baseline, LSA retrieval with truncated SVD (k = 20, 50, 100, 200, 300), evaluation
with P@5, MAP and nDCG@10, and Word2Vec (CBOW / Skip-gram) compared with count-based word similarity.

| File / folder | Content |
|---|---|
| `202303022_lab10.ipynb` | Notebook with all code and visible outputs (Parts 1–4) |
| `202303022_lab10.html` | Executed notebook exported to HTML |
| `202303022_lab10_report.pdf` | Report with results and discussion |
| `outputs/rankings/` | Top-10 rankings per query (CSV + TREC format) for TF-IDF and every LSA k |
| `outputs/tables/`, `outputs/figures/` | Result tables and figures |
| `outputs/results.json` | All reported numbers |
| `data/` | `cran.all.1400`, `cran.qry`, `cranqrel` |
| `make_report.py` | Builds the PDF report from `outputs/` |

## Key results (222 evaluated queries)

| System | P@5 | MAP | nDCG@10 |
|---|---|---|---|
| TF-IDF | 0.2739 | 0.2912 | 0.3495 |
| LSA k=50 | 0.2279 | 0.2685 | 0.3067 |
| LSA k=100 | 0.2658 | 0.3016 | 0.3531 |
| **LSA k=200** | **0.2856** | **0.3187** | **0.3817** |

## Reproduce

```bash
pip install -r requirements.txt
jupyter nbconvert --to notebook --execute --inplace 202303022_lab10.ipynb
jupyter nbconvert --to html 202303022_lab10.ipynb
python make_report.py
```

Relevance: grade 1–3 = relevant; nDCG gain = 4 − grade. Queries in `cranqrel` are numbered 1–225 in the order they
appear in `cran.qry`.
