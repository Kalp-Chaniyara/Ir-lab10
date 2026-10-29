"""Builds the Lab 10 report (PDF) from the outputs written by 202303022_lab10.ipynb.

Run the notebook first, then:  python make_report.py
"""
import json
import os
from pathlib import Path

import matplotlib
import pandas as pd
from reportlab.lib import colors
from reportlab.lib.enums import TA_CENTER
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import (Image, KeepTogether, PageBreak, Paragraph, SimpleDocTemplate, Spacer, Table,
                                TableStyle)

OUT = Path("outputs")
R = json.load(open(OUT / "results.json"))
REPORT = Path("202303022_lab10_report.pdf")

# DejaVu Sans (bundled with matplotlib) covers symbols such as Δ, →, ≈, ρ
FONT_DIR = Path(matplotlib.__file__).parent / "mpl-data" / "fonts" / "ttf"
pdfmetrics.registerFont(TTFont("DV", str(FONT_DIR / "DejaVuSans.ttf")))
pdfmetrics.registerFont(TTFont("DV-B", str(FONT_DIR / "DejaVuSans-Bold.ttf")))
pdfmetrics.registerFont(TTFont("DV-I", str(FONT_DIR / "DejaVuSans-Oblique.ttf")))
pdfmetrics.registerFont(TTFont("DV-BI", str(FONT_DIR / "DejaVuSans-BoldOblique.ttf")))
pdfmetrics.registerFont(TTFont("DVM", str(FONT_DIR / "DejaVuSansMono.ttf")))
from reportlab.pdfbase.pdfmetrics import registerFontFamily
registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

ss = getSampleStyleSheet()
BLUE = colors.HexColor("#1f4e79")
body = ParagraphStyle("body", parent=ss["Normal"], fontName="DV", fontSize=9.2, leading=12.6, spaceAfter=5)
small = ParagraphStyle("small", parent=body, fontSize=7.6, leading=9.6, spaceAfter=0)
smallb = ParagraphStyle("smallb", parent=small, fontName="DV-B")
cap = ParagraphStyle("cap", parent=body, fontName="DV-I", fontSize=8, leading=10, textColor=colors.HexColor("#444444"),
                     alignment=TA_CENTER, spaceBefore=2, spaceAfter=8)
h1 = ParagraphStyle("h1", parent=body, fontName="DV-B", fontSize=13, leading=16, textColor=BLUE, spaceBefore=10,
                    spaceAfter=5)
h2 = ParagraphStyle("h2", parent=body, fontName="DV-B", fontSize=10.5, leading=13, textColor=BLUE, spaceBefore=7,
                    spaceAfter=3)
title = ParagraphStyle("title", parent=body, fontName="DV-B", fontSize=17, leading=21, alignment=TA_CENTER,
                       textColor=BLUE, spaceAfter=3)
sub = ParagraphStyle("sub", parent=body, alignment=TA_CENTER, fontSize=10, leading=13, spaceAfter=1)
bullet = ParagraphStyle("bullet", parent=body, leftIndent=12, bulletIndent=2, spaceAfter=2.5)

story = []
P = lambda t, st=body: story.append(Paragraph(t, st))
H1 = lambda t: story.append(Paragraph(t, h1))
H2 = lambda t: story.append(Paragraph(t, h2))
def bullets(items):
    for it in items:
        story.append(Paragraph(it, bullet, bulletText="•"))

def table(rows, widths, header_rows=1, zebra=True, font_size=7.6, align_right_from=None):
    st = ParagraphStyle("t", parent=small, fontSize=font_size, leading=font_size + 2)
    stb = ParagraphStyle("tb", parent=st, fontName="DV-B")
    data = [[Paragraph(str(c), stb if r < header_rows else st) for c in row] for r, row in enumerate(rows)]
    t = Table(data, colWidths=widths, repeatRows=header_rows)
    style = [("BACKGROUND", (0, 0), (-1, header_rows - 1), colors.HexColor("#dce6f1")),
             ("LINEBELOW", (0, header_rows - 1), (-1, header_rows - 1), 0.8, BLUE),
             ("LINEABOVE", (0, 0), (-1, 0), 0.8, BLUE), ("LINEBELOW", (0, -1), (-1, -1), 0.8, BLUE),
             ("VALIGN", (0, 0), (-1, -1), "TOP"), ("TOPPADDING", (0, 0), (-1, -1), 2),
             ("BOTTOMPADDING", (0, 0), (-1, -1), 2), ("LEFTPADDING", (0, 0), (-1, -1), 3),
             ("RIGHTPADDING", (0, 0), (-1, -1), 3)]
    if zebra:
        for r in range(header_rows, len(rows)):
            if (r - header_rows) % 2 == 1:
                style.append(("BACKGROUND", (0, r), (-1, r), colors.HexColor("#f4f7fb")))
    t.setStyle(TableStyle(style))
    story.append(t)

def figure(path, width_cm, caption):
    from reportlab.lib.utils import ImageReader
    iw, ih = ImageReader(str(path)).getSize()
    w = width_cm * cm
    story.append(KeepTogether([Image(str(path), width=w, height=w * ih / iw), Paragraph(caption, cap)]))

f3 = lambda x: f"{x:.3f}"
f4 = lambda x: f"{x:.4f}"
pct = lambda a, b: f"{100 * (a / b - 1):+.1f} %"

ds, tf = R["dataset"], R["tfidf"]
kt = pd.DataFrame(R["k_table"])
base = kt.iloc[0]
lsa = {int(r["k (dims)"]): r for _, r in kt.iloc[1:].iterrows()}
cmp_ = R["top10_comparison"]

# ---------------------------------------------------------------- title
P("Lab Assignment 10 — Semantic IR Using LSA and Word2Vec", title)
P("IT550: Information Retrieval · Instructor: Prof. Sandip Modha", sub)
P("<b>Student ID: 202303022</b>", sub)
story.append(Spacer(1, 8))

H1("1. Overview")
P("This report covers a <b>TF-IDF</b> vector-space baseline and a <b>Latent Semantic Analysis (LSA)</b> retrieval "
  "system built with truncated SVD on the Cranfield collection, evaluated with P@5, MAP and nDCG@10. It also compares "
  "<b>Word2Vec</b> (CBOW and Skip-gram) word similarities with count-based (TF-IDF and PPMI) similarities. All numbers "
  "come from the notebook <font name='DVM'>202303022_lab10.ipynb</font>, which was run from top to bottom. Its "
  "outputs (rankings, tables, figures, <font name='DVM'>results.json</font>) are in <font name='DVM'>outputs/</font>.")

H2("Dataset and evaluation protocol")
bullets([
    f"<b>Cranfield</b>: {ds['n_docs']:,} aerodynamics abstracts (<font name='DVM'>cran.all.1400</font>), "
    f"{ds['n_queries']} queries (<font name='DVM'>cran.qry</font>) and {ds['n_qrels']:,} judgments "
    "(<font name='DVM'>cranqrel</font>). The title (.T) and abstract (.W) of each document are indexed together.",
    "<font name='DVM'>cran.qry</font> uses sparse ids (001, 002, 004, …), while <font name='DVM'>cranqrel</font> "
    "numbers the queries 1–225 in file order. The <i>i</i>-th query of the file is therefore query <i>i</i> in the "
    "judgments. All results use this mapping.",
    "<b>Relevant</b> = grade 1, 2 or 3 (for P@5 and MAP). <b>nDCG gain = 4 − grade</b> (3, 2, 1 for grades 1, 2, 3; "
    "grade 4 gives 0, and grade −1 is non-relevant).",
    f"<b>{ds['n_eval_queries']} of {ds['n_queries']} queries</b> have at least one relevant document and are evaluated. "
    "Queries 31, 119 and 215 (original ids 56, 173, 339) have none and are skipped. Evaluated queries have "
    f"{ds['mean_rel_per_query']:.2f} relevant documents on average.",
    "<b>MAP</b> is computed over the full ranking of all 1,400 documents. <b>MAP@10</b> (AP over the saved top-10, "
    "divided by min(|R|, 10)) is also reported. nDCG@10 uses DCG = Σ gain<sub>i</sub>/log<sub>2</sub>(i+1), "
    "normalised by the ideal ordering of all judged relevant documents.",
])

# ---------------------------------------------------------------- Part 1
H1("2. Part 1 — Corpus preparation and TF-IDF baseline")
P("<b>Pipeline</b> (identical for documents and queries): lower-casing → tokenisation with the regex "
  "<font name='DVM'>[a-z]+</font> → removal of scikit-learn's 318 English stop words and of 1-character tokens → "
  "Porter stemming. Example: <i>“what similarity laws must be obeyed when constructing aeroelastic models of heated "
  "high speed aircraft”</i> → <font name='DVM'>similar law obey construct aeroelast model heat high speed aircraft</font>.")
P("<b>TF-IDF:</b> <font name='DVM'>TfidfVectorizer</font> with raw tf, smoothed idf = ln((1+N)/(1+df)) + 1 and "
  "L2-normalised rows, fitted on the documents only. Queries are transformed with the <b>same vocabulary and IDF "
  "weights</b>. Each query is compared with all documents by cosine similarity, and the top 10 are saved as the baseline "
  "(<font name='DVM'>outputs/rankings/tfidf_baseline_top10.csv</font> and <font name='DVM'>.trec</font>).")
oov = ", ".join(tf["query_oov_terms"].keys())
table([["Statistic", "Value"],
       ["Documents / queries", f"{ds['n_docs']:,} / {ds['n_queries']}"],
       ["Tokens after preprocessing", "131,725 (94.1 per document on average)"],
       ["Vocabulary size (stems)", f"{tf['vocab_size']:,}"],
       ["Document–term matrix shape", f"{tf['shape'][0]:,} × {tf['shape'][1]:,}"],
       ["Non-zero entries", f"{tf['nnz']:,} (55.4 per document)"],
       ["Sparsity of the document matrix", f"{100 * tf['sparsity']:.2f} %"],
       ["Query matrix shape / sparsity", f"{tf['query_shape'][0]} × {tf['query_shape'][1]} / 99.77 %"],
       ["Query stems not in the vocabulary (ignored)", oov]],
      [6.2 * cm, 10.5 * cm])
P("Table 1 — Corpus and TF-IDF matrix statistics.", cap)

# ---------------------------------------------------------------- Part 2
H1("3. Part 2 — Truncated SVD and the LSA representation")
lk = R["lsa_k50"]
P("The TF-IDF matrix X (documents × terms) is factorised with randomized truncated SVD, "
  "X ≈ U<sub>k</sub>Σ<sub>k</sub>V<sub>k</sub><sup>T</sup>, with <b>k = 50</b> (valid, because k ≪ min(1400, 4170)). "
  "Documents are represented by XV<sub>k</sub> = U<sub>k</sub>Σ<sub>k</sub>. A query vector q is folded into the "
  "<b>same space</b> as qV<sub>k</sub>. Documents are ranked by cosine similarity in the latent space, and the top 10 "
  "are saved (<font name='DVM'>lsa_k50_top10.csv</font>).")
table([["Quantity (k = 50)", "Value"],
       ["Explained variance ratio of the 50 retained components", f"{lk['explained_variance']:.4f} ({100 * lk['explained_variance']:.2f} %)"],
       ["Share of ‖X‖²<sub>F</sub> kept (Σσ<sub>i</sub>² / ‖X‖²<sub>F</sub>)", f"{lk['energy']:.4f}"],
       ["Document matrix in latent space", f"{lk['doc_shape'][0]:,} × {lk['doc_shape'][1]}"],
       ["Query matrix in latent space", f"{lk['query_shape'][0]} × {lk['query_shape'][1]}"],
       ["Term loadings V<sub>k</sub>", "4,170 × 50"]],
      [9.2 * cm, 7.5 * cm])
P("Table 2 — LSA representation with k = 50.", cap)
P("The variance is spread over many components. The first 50 components explain only 31 % of the variance, and even 300 "
  "explain 70 % (Figure 1), which is typical for a sparse, high-rank TF-IDF matrix of short abstracts. The first component "
  "has a lower ratio than the second because TruncatedSVD does not centre the data, so component 1 mostly captures "
  "the mean direction. The leading components are interpretable topics. Component 2 contrasts <i>wing/buckling/flutter</i> "
  "with <i>boundary layer/heat transfer</i>, and component 3 groups <i>buckling, shell, stress, cylinder, load</i>.")
figure(OUT / "figures" / "explained_variance.png", 16.5,
       "Figure 1 — Explained variance per component (left) and cumulative explained variance (right).")

# ---------------------------------------------------------------- Part 3
H1("4. Part 3 — Evaluation and dimensionality analysis")
H2("4.1 TF-IDF vs LSA (k = 50)")
l50 = lsa[50]
table([["System (222 queries)", "P@5", "MAP", "MAP@10", "nDCG@10"],
       ["TF-IDF baseline", f4(base["P@5"]), f4(base["MAP"]), f4(base["MAP@10"]), f4(base["nDCG@10"])],
       ["LSA, k = 50", f4(l50["P@5"]), f4(l50["MAP"]), f4(l50["MAP@10"]), f4(l50["nDCG@10"])],
       ["Relative change", pct(l50["P@5"], base["P@5"]), pct(l50["MAP"], base["MAP"]),
        pct(l50["MAP@10"], base["MAP@10"]), pct(l50["nDCG@10"], base["nDCG@10"])]],
      [5.2 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm, 2.8 * cm])
P("Table 3 — Retrieval quality of the baseline and of LSA with k = 50.", cap)
P(f"<b>Top-10 comparison.</b> The two systems share on average only <b>{cmp_['mean_overlap']:.2f}</b> of their 10 documents "
  f"(Jaccard {cmp_['mean_jaccard']:.3f}), so LSA does more than re-order the TF-IDF ranking. LSA's top-10 lists contain "
  f"<b>{cmp_['rel_only_lsa']}</b> relevant documents that TF-IDF misses, but they lose <b>{cmp_['rel_only_tfidf']}</b> "
  f"that TF-IDF finds. In total LSA has {cmp_['rel_in_top10_lsa']} relevant documents in its top-10 lists against "
  f"{cmp_['rel_in_top10_tfidf']} for TF-IDF, and {cmp_['zero_rel_lsa']} queries without any relevant document "
  f"against {cmp_['zero_rel_tfidf']}.")
pq = R["per_query"]
P(f"<b>Per query</b>, LSA (k = 50) has a higher AP on {pq['ΔMAP']['better']} queries and a lower AP on "
  f"{pq['ΔMAP']['worse']} ({pq['ΔMAP']['tied']} ties). nDCG@10 is higher on {pq['ΔnDCG@10']['better']} queries, lower "
  f"on {pq['ΔnDCG@10']['worse']}, with {pq['ΔnDCG@10']['tied']} ties (Figure 2).")
figure(OUT / "figures" / "per_query_delta_ndcg.png", 15.5,
       "Figure 2 — Per-query difference in nDCG@10 (LSA k = 50 minus TF-IDF), sorted.")

def q_rows(recs):
    rows = [["qid", "Query", "AP TF-IDF", "AP LSA", "nDCG TF-IDF", "nDCG LSA", "ΔnDCG@10"]]
    for r in recs[:6]:
        rows.append([r["qid"], r["query"], f3(r["MAP tfidf"]), f3(r["MAP lsa"]), f3(r["nDCG@10 tfidf"]),
                     f3(r["nDCG@10 lsa"]), f"{r['ΔnDCG@10']:+.3f}"])
    return rows
H2("4.2 Queries where LSA improves or worsens the ranking")
table(q_rows(R["best_queries"]), [0.9 * cm, 8.1 * cm, 1.5 * cm, 1.4 * cm, 1.7 * cm, 1.5 * cm, 1.6 * cm])
P("Table 4 — Largest improvements by LSA (k = 50).", cap)
table(q_rows(R["worst_queries"]), [0.9 * cm, 8.1 * cm, 1.5 * cm, 1.4 * cm, 1.7 * cm, 1.5 * cm, 1.6 * cm])
P("Table 5 — Largest degradations by LSA (k = 50).", cap)

P("<b>Improvement — qid 184 “work on small-oscillation re-entry motions”</b> (nDCG@10 0.00 → 0.70, AP 0.10 → 0.60). "
  "The relevant documents use different wording for the same topic: <i>“dynamic stability of vehicles traversing "
  "ascending or descending paths”</i> (doc 67), <i>“oscillatory motion of manned vehicles <b>entering</b> the earth's "
  "atmosphere”</i> (doc 716), <i>“oscillations of a vehicle <b>entering</b> a planetary atmosphere”</i> (doc 499). "
  "The query stem <font name='DVM'>entri</font> (from <i>re-entry</i>) never matches <font name='DVM'>enter</font>, and "
  "the generic words <i>work</i> and <i>small</i> add noise, so TF-IDF ranks these documents 13th–94th. In the latent "
  "space <i>missile, vehicle, descending, atmosphere, oscillation, motion</i> load on the same dimensions. LSA moves "
  "doc 67 from rank 57 to 1 and doc 716 from 20 to 2. This is the <b>synonymy / vocabulary-mismatch</b> problem that "
  "LSA is designed to solve.")
P("<b>Degradation — qid 136 “creep buckling of round tubes under external pressure”</b> (AP 1.00 → 0.13). The single "
  "relevant document (1021) shares five stems with the query, and TF-IDF ranks it first. The collection holds a dense "
  "cluster of <i>creep buckling</i> papers (950, 951, 1013, 1017, 1031, …). With 50 dimensions they collapse onto nearly the "
  "same latent vector, and the words that single out the relevant paper (<i>tubes, round, external pressure</i>) are "
  "averaged away, so doc 1021 drops to rank 8. Queries 148 (<i>sandwich</i> cylinders), 5 (<i>chemical kinetic</i> "
  "systems) and 9 (<i>slip flow</i> heat transfer) fail the same way: a rare, discriminative term is what matters, and "
  "the truncated SVD discards it. This is the <b>precision cost</b> of merging related topics.")
ob = R["overlap_buckets"]
table([["Share of query stems in the relevant document", "Pairs", "Median rank TF-IDF", "Median rank LSA",
        "LSA ranks it higher"]] +
      [[b["bucket"], b["pairs"], f"{b['median_rank_tfidf']:.0f}", f"{b['median_rank_lsa']:.1f}".rstrip("0").rstrip("."),
        f"{100 * b['lsa_ranks_higher']:.1f} %"] for b in ob],
      [6.0 * cm, 1.6 * cm, 3.0 * cm, 3.0 * cm, 3.1 * cm])
P("Table 6 — All (query, relevant document) pairs grouped by literal word overlap with the query.", cap)
P("Table 6 supports this explanation across all queries. When a relevant document shares at most 20 % of the "
  "query's stems, LSA ranks it higher than TF-IDF in 72.6 % of cases (median rank 397 → 188). When it shares more than 60 %, "
  "TF-IDF wins and LSA ranks it higher only 28.8 % of the time. <b>LSA helps on vocabulary mismatch and hurts where "
  "exact matching already works.</b>")

H2("4.3 Effect of the number of latent dimensions and runtime")
rows = [["System", "k", "Explained var.", "P@5", "MAP", "MAP@10", "nDCG@10", "Index / fit time (s)",
         "Query time (s, 225 q.)"]]
for _, r in kt.iterrows():
    is_base = r["system"] == "TF-IDF"
    rows.append(["TF-IDF" if is_base else "LSA", "4,170 (vocab.)" if is_base else int(r["k (dims)"]),
                 "—" if is_base else f"{100 * r['explained var.']:.1f} %", f4(r["P@5"]), f4(r["MAP"]),
                 f4(r["MAP@10"]), f4(r["nDCG@10"]), f"{r['fit time (s)']:.3f}", f"{r['query time (s)']:.3f}"])
table(rows, [1.3 * cm, 1.9 * cm, 1.8 * cm, 1.3 * cm, 1.3 * cm, 1.7 * cm, 1.9 * cm, 2.4 * cm, 2.6 * cm])
P("Table 7 — Retained dimensions, retrieval metrics and runtime (median of 3 runs). TF-IDF fit time includes "
  "preprocessing and building the matrix, which LSA also needs before its SVD.", cap)
figure(OUT / "figures" / "metrics_vs_k.png", 16.5, "Figure 3 — Retrieval quality of LSA as a function of k "
       "(dashed line: TF-IDF baseline).")
l100, l200, l300 = lsa[100], lsa[200], lsa[300]
sr = R["seed_robustness_k50"]["std"]
bullets([
    "Quality <b>rises with k and then falls slightly</b>. With k = 20 the space is too coarse (P@5 0.170). "
    f"k = 50 is below the baseline on every metric, and k = 100 roughly ties it (MAP {pct(l100['MAP'], base['MAP'])}, "
    f"nDCG@10 {pct(l100['nDCG@10'], base['nDCG@10'])}, P@5 {pct(l100['P@5'], base['P@5'])}).",
    f"<b>k = 200 is best</b>: P@5 {f4(l200['P@5'])} ({pct(l200['P@5'], base['P@5'])}), MAP {f4(l200['MAP'])} "
    f"({pct(l200['MAP'], base['MAP'])}), nDCG@10 {f4(l200['nDCG@10'])} ({pct(l200['nDCG@10'], base['nDCG@10'])}). "
    f"At k = 300 the scores drop again (MAP {f4(l300['MAP'])}) because LSA moves back toward TF-IDF as k approaches "
    "the rank of the matrix. For this small, technical collection the k = 50 suggested in the handout is too aggressive.",
    "Explained variance (19 % → 70 %) is a poor guide for choosing k. The best k keeps only 60 % of the variance, so k "
    "should be tuned on retrieval metrics.",
    f"The randomized SVD is stable: over 5 seeds the k = 50 metrics vary by at most ±{max(sr.values()):.3f}.",
    f"<b>Runtime:</b> the SVD (indexing) cost grows roughly linearly with k ({lsa[20]['fit time (s)']:.2f} s at k = 20, "
    f"{lsa[300]['fit time (s)']:.2f} s at k = 300, on top of building the TF-IDF matrix). Query processing (folding in "
    f"225 queries, scoring and sorting 1,400 documents) takes {1000 * kt['query time (s)'].min():.0f}–"
    f"{1000 * kt['query time (s)'].max():.0f} ms for every system, well under 1 ms per query. At this scale it is "
    "dominated by sorting and hardly depends on k.",
])

# ---------------------------------------------------------------- Part 4
H1("5. Part 4 — Word embeddings with Word2Vec")
w = R["w2v"]
P(f"<b>Training data:</b> the same 1,400 documents (title + abstract), split into {w['sentences']:,} sentences "
  f"({w['tokens']:,} tokens). The tokenizer and stop-word list are the same as in Part 1. Stemming is not applied here, so "
  "neighbours stay readable. <b>Settings</b> (identical except the architecture): vector size 100, window 5, "
  "min_count 5 (vocabulary of 2,697 words), negative sampling with 10 noise words, sub-sampling 1e-3, 30 epochs, "
  "one worker thread with a fixed seed and hash function for reproducibility. <b>CBOW</b> (sg = 0) predicts a word "
  "from the average of its context. <b>Skip-gram</b> (sg = 1) predicts each context word from the word.")
H2("5.1 Five nearest words — CBOW vs Skip-gram")
nn = R["w2v_neighbours"]
table([["Term (freq.)", "CBOW top-5 (cosine)", "Skip-gram top-5 (cosine)", "Shared"]] +
      [[f"<b>{r['term']}</b> ({r['freq']})", r["CBOW top-5"], r["Skip-gram top-5"], r["shared"]] for r in nn],
      [2.3 * cm, 6.75 * cm, 6.75 * cm, 1.4 * cm], font_size=7.2)
P("Table 8 — Five nearest neighbours by cosine similarity.", cap)
cc = R["w2v_config_compare"]
table([["Model", "Mean cos, 12 related pairs", "Mean cos, 500 random pairs", "Gap", "Training time (s)"],
       ["CBOW", f3(cc["CBOW"]["mean cos (related pairs)"]), f3(cc["CBOW"]["mean cos (random pairs)"]),
        f3(cc["CBOW"]["gap"]), f"{cc['CBOW']['train time (s)']:.1f}"],
       ["Skip-gram", f3(cc["Skip-gram"]["mean cos (related pairs)"]), f3(cc["Skip-gram"]["mean cos (random pairs)"]),
        f3(cc["Skip-gram"]["gap"]), f"{cc['Skip-gram']['train time (s)']:.1f}"]],
      [2.6 * cm, 4.2 * cm, 4.2 * cm, 2.2 * cm, 3.0 * cm])
P("Table 9 — Comparison of the two configurations.", cap)
bullets([
    "<b>Kind of neighbours.</b> CBOW returns words that can <b>replace</b> the target in a sentence: "
    "<i>supersonic → transonic, subsonic</i>; <i>buckling → deformations, wrinkling, buckle</i>; <i>aerofoil → airfoil, "
    "wings, delta</i>. Skip-gram returns more <b>topical associates</b>, including rare, specific words: <i>heat → transfer, "
    "shield, refractory, transpiration</i> (the heat-shield topic); <i>buckling → rings, unstiffened, stiffened, postbuckling</i>.",
    f"<b>Agreement</b> is low. On average only {100 * cc['jaccard_top10_cbow_vs_sg']:.0f} % of the top-10 neighbours are "
    f"shared (Jaccard {cc['jaccard_top10_cbow_vs_sg']:.3f}, 1,000 most frequent words).",
    "Both models separate related from random pairs. CBOW has the larger gap. Skip-gram gives even random pairs a cosine "
    "of about 0.2, so its scores must be read relative to that baseline. Skip-gram trains "
    f"{cc['Skip-gram']['train time (s)'] / cc['CBOW']['train time (s)']:.1f}× slower, because it makes "
    "one prediction per (word, context word) pair.",
])
H2("5.2 Comparison with count-based similarity")
P("Two count-based word representations are built on the same unstemmed vocabulary. <b>TF-IDF (document level)</b>: "
  "each word is a column of the document–term TF-IDF matrix, so words are similar when they occur in the <i>same "
  "documents</i>. <b>PPMI (window level)</b>: word × context counts within ±5 words, weighted by positive pointwise "
  "mutual information. All similarities are cosines.")
cn = R["count_neighbours"]
table([["Term", "TF-IDF (same documents)", "PPMI (±5-word window)", "Skip-gram"]] +
      [[f"<b>{r['term']}</b>", r["TF-IDF (same documents)"], r["PPMI (window co-occurrence)"], r["Skip-gram"]]
       for r in cn], [1.8 * cm, 5.1 * cm, 5.1 * cm, 5.2 * cm], font_size=7.0)
P("Table 10 — Five nearest neighbours under count-based and embedding similarity.", cap)
pt = R["pair_table"]
table([["Word pair", "df(a)", "df(b)", "Docs with both", "TF-IDF", "PPMI", "LSA k=100", "CBOW", "Skip-gram"]] +
      [[r["pair"], r["df(a)"], r["df(b)"], r["docs with both"], f3(r["TF-IDF cos"]), f3(r["PPMI cos"]),
        f3(r["LSA k=100 cos"]), f3(r["CBOW cos"]), f3(r["Skip-gram cos"])] for r in pt],
      [4.0 * cm, 1.2 * cm, 1.2 * cm, 1.8 * cm, 1.5 * cm, 1.5 * cm, 1.8 * cm, 1.5 * cm, 1.8 * cm])
P("Table 11 — Cosine similarity of selected word pairs (LSA uses stems, so <i>plate/plates</i> = 1.0).", cap)
corr = R["similarity_correlation"]
names = ["TF-IDF", "PPMI", "CBOW", "Skip-gram"]
table([["Spearman ρ"] + names] + [[a] + [f3(corr[a][b]) for b in names] for a in names],
      [3.0 * cm] + [2.6 * cm] * 4)
P("Table 12 — Rank correlation of pairwise similarities over all 44,850 pairs of the 300 most frequent words.", cap)
bullets([
    "Document-level <b>TF-IDF</b> neighbours are topical associates (<i>supersonic → flow, speeds, mach, theory, pressure</i>). "
    "A pair scores high only when the words occur in the same abstracts (<i>heat–transfer</i> 0.83, <i>shock–wave</i> 0.54).",
    "<b>PPMI</b> uses local context and is already closer to Word2Vec (<i>supersonic → subsonic</i>; <i>wing → wings, "
    "tail, chord, flaps</i>). Word2Vec correlates more with PPMI (ρ ≈ 0.52–0.54) than with TF-IDF (ρ ≈ 0.37–0.40). This "
    "is consistent with skip-gram with negative sampling implicitly factorising a shifted PMI matrix.",
    "Words that <b>substitute</b> for each other but rarely appear in the same abstract get almost zero TF-IDF "
    "similarity (<i>aerofoil–airfoil</i> 0.013, <i>wing–aerofoil</i> 0.035, <i>supersonic–hypersonic</i> 0.086). CBOW "
    "gives them 0.46–0.49. LSA term vectors behave like smoothed TF-IDF: they help when the documents themselves "
    "overlap (<i>viscous–inviscid</i> 0.27 → 0.56) but not for disjoint spelling variants.",
])

H2("5.3 A relationship that embeddings capture and word overlap misses: aerofoil / airfoil")
P("The abstracts use both the British spelling <b>aerofoil</b> (18 documents) and the American <b>airfoil</b> "
  "(53 documents) for the same object, and only <b>one</b> document contains both. Exact matching treats them as "
  "unrelated: the stems differ, so a query that says <i>airfoil</i> gets no credit from a document that says "
  "<i>aerofoil</i>. Their document-level TF-IDF cosine is <b>0.013</b>. LSA (k = 100) gives <b>−0.03</b>, because the two "
  "spellings occur in disjoint sets of documents.")
P("<b>CBOW ranks airfoil as the nearest neighbour of aerofoil</b> (rank 1 of 2,696 words, cosine 0.49). "
  "Word2Vec does not need the two words to co-occur. It needs them to occur in <b>similar contexts</b>, and both appear "
  "next to <i>theory, pressure, lift, distribution, section, mach, velocity, transonic, flow</i>. Because CBOW learns to "
  "predict a word from its context, two words that fill the same slot get similar vectors. This second-order "
  "(distributional) relation is invisible to first-order word overlap. A systematic search found more such pairs that "
  "never share a document but have Skip-gram cosine ≈ 0.55, for example <i>diameter–width</i> and <i>integration–solving</i>. "
  "In retrieval this affects queries 42 and 43 (“transonic flow around an <i>airfoil</i>”): relevant doc 468 says only "
  "<i>aerofoil</i> and is found (ranks 6 and 12) only through other shared words. Expanding queries with "
  "embedding neighbours would close this gap.")
P("Skip-gram ranks <i>airfoil</i> only 450th for <i>aerofoil</i>. With 35 occurrences, Skip-gram fits the "
  "word's specific co-occurring terms (<i>split, discs, inverted</i>) rather than its general slot, so CBOW's context averaging "
  "is more robust on this small corpus. Note that Word2Vec vectors are <i>static</i>, one per word type learnt from its contexts. "
  "Fully contextual models such as BERT would also distinguish senses of a word from sentence to sentence.")
figure(OUT / "figures" / "skipgram_pca.png", 11.5, "Figure 4 — 2-D PCA projection of Skip-gram vectors for five semantic "
       "groups. The groups form separate regions, and aerofoil/airfoil/wing lie together.")

# ---------------------------------------------------------------- conclusions
H1("6. Conclusions")
bullets([
    f"<b>TF-IDF baseline</b> (4,170 stems, 98.7 % sparse): P@5 {f4(base['P@5'])}, MAP {f4(base['MAP'])}, "
    f"nDCG@10 {f4(base['nDCG@10'])} on 222 queries.",
    f"<b>LSA with k = 50</b> keeps 31 % of the variance and is <b>worse</b> than TF-IDF on Cranfield (MAP {f4(l50['MAP'])}, "
    f"nDCG@10 {f4(l50['nDCG@10'])}). It recovers relevant documents written with different vocabulary (qid 184) but "
    "blurs precise queries whose key term is rare (qid 136).",
    f"<b>Dimensionality matters most</b>: k ≈ 200 gives the best system (MAP {f4(l200['MAP'])}, nDCG@10 "
    f"{f4(l200['nDCG@10'])}, about +9 % over TF-IDF). Small k under-fits, and very large k turns LSA back into TF-IDF. Indexing cost grows "
    "with k, while query cost stays well under 1 ms per query.",
    "<b>Word2Vec</b>: CBOW gives substitutable neighbours and Skip-gram gives topical associates. Both agree more with "
    "window-based PPMI than with document-level TF-IDF, and they capture distributional relations such as "
    "<i>aerofoil ≈ airfoil</i> that word-overlap similarity cannot see.",
])

H1("7. Submission contents and reproduction")
table([["File / folder", "Content"],
       ["202303022_lab10.ipynb", "Notebook with all code and visible outputs (Parts 1–4)"],
       ["202303022_lab10.html", "The executed notebook exported as HTML (easy to view)"],
       ["202303022_lab10_report.pdf", "This report"],
       ["outputs/rankings/", "Top-10 rankings for every query: TF-IDF baseline and LSA k = 20, 50, 100, 200, 300 (CSV and TREC format)"],
       ["outputs/tables/, outputs/figures/", "Per-query metrics, metrics by k, neighbour tables, word-pair similarities; all figures"],
       ["outputs/results.json", "All reported numbers in machine-readable form"],
       ["data/", "Cranfield files (cran.all.1400, cran.qry, cranqrel)"],
       ["make_report.py, requirements.txt, README.md", "Report generator, dependencies, instructions"]],
      [5.2 * cm, 11.5 * cm])
P("To reproduce: <font name='DVM'>pip install -r requirements.txt</font>, then "
  "<font name='DVM'>jupyter nbconvert --to notebook --execute --inplace 202303022_lab10.ipynb</font> and "
  "<font name='DVM'>python make_report.py</font>. All random components (randomized SVD, Word2Vec) use fixed seeds. "
  "Runtimes depend on the machine.")


def on_page(canvas, doc):
    canvas.saveState()
    canvas.setFont("DV", 7.5)
    canvas.setFillColor(colors.grey)
    canvas.drawString(2 * cm, 1.2 * cm, "IT550 Lab 10 — Semantic IR using LSA and Word2Vec — 202303022")
    canvas.drawRightString(A4[0] - 2 * cm, 1.2 * cm, f"page {doc.page}")
    canvas.restoreState()


doc = SimpleDocTemplate(str(REPORT), pagesize=A4, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm,
                        bottomMargin=1.8 * cm, title="Lab 10 — Semantic IR Using LSA and Word2Vec (202303022)",
                        author="202303022")
doc.build(story, onFirstPage=on_page, onLaterPages=on_page)
print("wrote", REPORT, f"({os.path.getsize(REPORT) / 1024:.0f} KB)")
