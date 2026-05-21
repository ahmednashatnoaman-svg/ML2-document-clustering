"""Generate all 4 project notebooks as .ipynb files."""

import nbformat as nbf
from pathlib import Path

NB_DIR = Path(__file__).parent


def md(text): return nbf.v4.new_markdown_cell(text)
def code(text): return nbf.v4.new_code_cell(text)


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 01 — EDA
# ─────────────────────────────────────────────────────────────────────────────
def make_nb01():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 01 — Exploratory Data Analysis\n\nExplore both datasets before any preprocessing."),
        code("""\
import sys; sys.path.insert(0, '..')
import pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['figure.dpi'] = 120
from collections import Counter
from sklearn.datasets import fetch_20newsgroups
"""),
        md("## 1. 20 Newsgroups Dataset"),
        code("""\
news = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'), random_state=42)
texts_ng, labels_ng, names_ng = news.data, news.target, news.target_names
print(f"Documents : {len(texts_ng):,}")
print(f"Categories: {len(names_ng)}")
print(f"\\nCategory list:")
for i, n in enumerate(names_ng):
    cnt = int((np.array(labels_ng) == i).sum())
    print(f"  {i:2d}. {n:<35s} {cnt:4d} docs")
"""),
        code("""\
# Document length distribution
lengths_ng = [len(t.split()) for t in texts_ng]
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist(lengths_ng, bins=60, color='steelblue', edgecolor='white', linewidth=0.4)
axes[0].set_xlabel('Word Count'); axes[0].set_ylabel('Documents')
axes[0].set_title('20 Newsgroups — Document Length Distribution')
axes[0].axvline(np.median(lengths_ng), color='red', linestyle='--', label=f'Median {np.median(lengths_ng):.0f}')
axes[0].legend()
# Category sizes
cat_counts = Counter(labels_ng)
cats = [names_ng[i] for i in sorted(cat_counts)]
counts = [cat_counts[i] for i in sorted(cat_counts)]
axes[1].barh(cats, counts, color='steelblue')
axes[1].set_xlabel('Document Count')
axes[1].set_title('Documents per Category')
plt.tight_layout(); plt.savefig('../outputs/figures/eda_newsgroups.png', bbox_inches='tight'); plt.show()
print(f"Median length: {np.median(lengths_ng):.0f} words | Mean: {np.mean(lengths_ng):.0f} | Max: {max(lengths_ng):,}")
"""),
        code("""\
# Top word frequencies (raw)
from collections import Counter
import re
all_words = re.findall(r'\\b[a-z]{3,}\\b', ' '.join(texts_ng).lower())
top50 = Counter(all_words).most_common(30)
words, freqs = zip(*top50)
fig, ax = plt.subplots(figsize=(12, 4))
ax.bar(words, freqs, color='darkorange')
ax.set_xticklabels(words, rotation=60, ha='right')
ax.set_title('Top 30 Raw Words — 20 Newsgroups (before preprocessing)')
ax.set_ylabel('Frequency')
plt.tight_layout(); plt.savefig('../outputs/figures/eda_top_words_newsgroups.png', bbox_inches='tight'); plt.show()
"""),
        md("## 2. Wikipedia People Dataset"),
        code("""\
df_wiki = pd.read_csv('../data/raw/people_wiki.csv')
print(f"Articles  : {len(df_wiki):,}")
print(f"Columns   : {list(df_wiki.columns)}")
print(f"Null texts: {df_wiki['text'].isna().sum()}")
print("\\nSample names:")
print(df_wiki['name'].head(10).tolist())
"""),
        code("""\
lengths_wiki = df_wiki['text'].str.split().str.len().fillna(0).astype(int)
fig, ax = plt.subplots(figsize=(10, 4))
ax.hist(lengths_wiki, bins=40, color='seagreen', edgecolor='white')
ax.axvline(lengths_wiki.median(), color='red', linestyle='--', label=f'Median {lengths_wiki.median():.0f}')
ax.set_xlabel('Word Count'); ax.set_ylabel('Articles')
ax.set_title('Wikipedia People — Article Length Distribution')
ax.legend()
plt.tight_layout(); plt.savefig('../outputs/figures/eda_wikipedia.png', bbox_inches='tight'); plt.show()
print(f"Median: {lengths_wiki.median():.0f} | Mean: {lengths_wiki.mean():.0f} | Max: {lengths_wiki.max()}")
"""),
        code("""\
# Show sample articles
for _, row in df_wiki.sample(3, random_state=42).iterrows():
    print(f"NAME: {row['name']}")
    print(f"TEXT: {row['text'][:300]}...")
    print()
"""),
        md("## 3. Dataset Comparison Summary"),
        code("""\
summary = pd.DataFrame({
    'Dataset': ['20 Newsgroups', 'Wikipedia People'],
    'Documents': [len(texts_ng), len(df_wiki)],
    'Categories': [len(names_ng), 'N/A (biographical)'],
    'Median Doc Length': [np.median(lengths_ng), lengths_wiki.median()],
    'Mean Doc Length': [np.mean(lengths_ng), lengths_wiki.mean()],
})
print(summary.to_string(index=False))
"""),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 02 — Preprocessing
# ─────────────────────────────────────────────────────────────────────────────
def make_nb02():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 02 — Preprocessing & Feature Extraction\n\nDemonstrates the full text cleaning pipeline and TF-IDF feature matrix."),
        code("""\
import sys; sys.path.insert(0, '..')
import pickle, numpy as np, pandas as pd
import matplotlib.pyplot as plt
from scipy.sparse import load_npz
import joblib
from src.preprocessing import clean_text, preprocess_text, preprocess_corpus
from src.features import build_tfidf, reduce_with_svd
"""),
        md("## 1. Preprocessing Pipeline Walk-through"),
        code("""\
sample = texts = [
    "The quick Brown Foxes are RUNNING over 42 lazy dogs! Visit http://example.com for more info.",
    "<b>NASA</b> launched a new satellite into low-earth orbit on Friday.",
    "The government's encryption policy involving NSA clipper chips was controversial.",
]
print("=== RAW TEXT ===")
for i, t in enumerate(sample): print(f"  [{i}] {t}")

print("\\n=== AFTER clean_text() ===")
for i, t in enumerate(sample): print(f"  [{i}] {clean_text(t)}")

print("\\n=== AFTER preprocess_text() (full pipeline) ===")
for i, t in enumerate(sample): print(f"  [{i}] {preprocess_text(t)}")
"""),
        md("## 2. Vocabulary & Token Statistics"),
        code("""\
from sklearn.datasets import fetch_20newsgroups
news = fetch_20newsgroups(subset='all', remove=('headers','footers','quotes'), random_state=42)
raw_texts = news.data[:500]  # sample for demo speed

processed, kept = preprocess_corpus(raw_texts, min_doc_len=10)
print(f"Input : {len(raw_texts)} docs")
print(f"Output: {len(processed)} docs ({len(raw_texts)-len(processed)} dropped)")

# Token count before vs after
import re, nltk
from nltk.corpus import stopwords
STOPS = set(stopwords.words('english'))
raw_tokens  = [re.findall(r'\\b[a-z]{3,}\\b', t.lower()) for t in raw_texts]
proc_tokens = [t.split() for t in processed]
raw_mean  = np.mean([len(t) for t in raw_tokens])
proc_mean = np.mean([len(t) for t in proc_tokens])
print(f"\\nMean tokens — raw: {raw_mean:.1f}  |  processed: {proc_mean:.1f}")
print(f"Compression: {(1 - proc_mean/raw_mean)*100:.1f}% reduction")
"""),
        code("""\
# Before / after token distribution plot
fig, axes = plt.subplots(1, 2, figsize=(13, 4))
axes[0].hist([len(t) for t in raw_tokens], bins=40, color='steelblue', edgecolor='white')
axes[0].set_title('Token Count BEFORE Preprocessing'); axes[0].set_xlabel('Tokens')
axes[1].hist([len(t) for t in proc_tokens], bins=40, color='seagreen', edgecolor='white')
axes[1].set_title('Token Count AFTER Preprocessing'); axes[1].set_xlabel('Tokens')
plt.tight_layout(); plt.savefig('../outputs/figures/preprocessing_tokens.png', bbox_inches='tight'); plt.show()
"""),
        md("## 3. TF-IDF Feature Matrix"),
        code("""\
# Load pre-built TF-IDF for 20 Newsgroups
X_tfidf_ng = load_npz('../data/processed/X_tfidf_newsgroups.npz')
vec_ng = joblib.load('../models/tfidf_newsgroups.pkl')
print(f"TF-IDF shape: {X_tfidf_ng.shape}")
print(f"Sparsity    : {1 - X_tfidf_ng.nnz / np.prod(X_tfidf_ng.shape):.3%}")
print(f"Vocab size  : {len(vec_ng.vocabulary_):,}")
print(f"\\nTop 30 terms by document frequency:")
df_freqs = (X_tfidf_ng > 0).sum(axis=0).A1
top_idx = df_freqs.argsort()[::-1][:30]
feat_names = vec_ng.get_feature_names_out()
print("  " + ", ".join(feat_names[top_idx]))
"""),
        md("## 4. LSA — Explained Variance"),
        code("""\
from sklearn.decomposition import TruncatedSVD
X_reduced_ng = np.load('../data/processed/X_reduced_newsgroups.npy')
print(f"Reduced matrix shape: {X_reduced_ng.shape}")

svd_ng = joblib.load('../models/svd_newsgroups.pkl')
explained = svd_ng.named_steps['svd'].explained_variance_ratio_
cumulative = np.cumsum(explained)

fig, ax = plt.subplots(figsize=(10, 4))
ax.plot(range(1, len(cumulative)+1), cumulative, marker='.', markersize=4, color='steelblue')
ax.axhline(0.5, linestyle='--', color='red', label='50% threshold')
ax.set_xlabel('Number of SVD Components')
ax.set_ylabel('Cumulative Explained Variance')
ax.set_title('LSA — Explained Variance Ratio (20 Newsgroups)')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.savefig('../outputs/figures/svd_variance_newsgroups.png', bbox_inches='tight'); plt.show()
print(f"100 components explain {cumulative[99]:.1%} of variance")
"""),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 03 — Clustering
# ─────────────────────────────────────────────────────────────────────────────
def make_nb03():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 03 — Clustering Experiments\n\nVisualise clustering results and compare algorithms."),
        code("""\
import sys; sys.path.insert(0, '..')
import pickle, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt, seaborn as sns
from scipy.sparse import load_npz
from src.evaluation import get_top_terms, cluster_size_distribution
"""),
        md("## 1. Load Artifacts"),
        code("""\
DATASET = 'newsgroups'   # change to 'wikipedia' to switch

with open(f'../data/processed/{DATASET}_clean.pkl','rb') as f: corpus = pickle.load(f)
vec    = joblib.load(f'../models/tfidf_{DATASET}.pkl')
svd    = joblib.load(f'../models/svd_{DATASET}.pkl')
X      = np.load(f'../data/processed/X_reduced_{DATASET}.npy')
X_tf   = load_npz(f'../data/processed/X_tfidf_{DATASET}.npz')
metrics = pd.read_csv(f'../outputs/reports/clustering_metrics_{DATASET}.csv')
print(f"Dataset: {DATASET} | Docs: {X.shape[0]:,} | Features: {X.shape[1]}")
"""),
        md("## 2. Elbow Curve (K-Means)"),
        code("""\
km_metrics = metrics[metrics['algorithm']=='kmeans'].sort_values('k')
inertias   = [joblib.load(f'../models/kmeans_{DATASET}_k{k}.pkl').inertia_
              for k in km_metrics['k']]
fig, ax = plt.subplots(figsize=(9, 5))
ax.plot(km_metrics['k'], inertias, 'o-', linewidth=2, color='steelblue')
ax.set_xlabel('k'); ax.set_ylabel('Inertia')
ax.set_title(f'K-Means Elbow Curve — {DATASET}')
ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
        md("## 3. Silhouette Score vs. k (all algorithms)"),
        code("""\
fig, ax = plt.subplots(figsize=(11, 5))
colors = {'kmeans':'steelblue','hierarchical':'darkorange','gmm':'seagreen'}
for algo, grp in metrics.groupby('algorithm'):
    best_per_k = grp.groupby('k')['silhouette'].max().reset_index()
    ax.plot(best_per_k['k'], best_per_k['silhouette'], 'o-',
            label=algo, color=colors.get(algo,'gray'), linewidth=2)
ax.set_xlabel('k'); ax.set_ylabel('Silhouette Score')
ax.set_title(f'Best Silhouette Score per k — {DATASET}')
ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout(); plt.show()
"""),
        md("## 4. Best K-Means: PCA Scatter"),
        code("""\
from sklearn.decomposition import PCA
best_k = int(metrics[metrics['algorithm']=='kmeans']['silhouette'].idxmax())
best_k = int(metrics.loc[best_k,'k'])
model  = joblib.load(f'../models/kmeans_{DATASET}_k{best_k}.pkl')
labels = model.predict(X)

pca    = PCA(n_components=2, random_state=42)
coords = pca.fit_transform(X)
var    = pca.explained_variance_ratio_

fig, ax = plt.subplots(figsize=(10, 7))
scatter = ax.scatter(coords[:,0], coords[:,1], c=labels, cmap='tab20', s=8, alpha=0.6)
plt.colorbar(scatter, ax=ax, label='Cluster')
ax.set_xlabel(f'PC1 ({var[0]:.1%} var)'); ax.set_ylabel(f'PC2 ({var[1]:.1%} var)')
ax.set_title(f'PCA — K-Means k={best_k} — {DATASET}')
plt.tight_layout(); plt.show()
"""),
        md("## 5. Top Terms per Cluster (Best K-Means)"),
        code("""\
top_terms = get_top_terms(labels, vec, X_tf, n=10)
sizes     = cluster_size_distribution(labels)
print(f"{'Cluster':>8} {'Size':>6}  Top Terms")
print("-" * 80)
for cid in sorted(top_terms):
    terms = ", ".join(top_terms[cid][:8])
    print(f"Cluster {cid:2d}  {sizes[cid]:5d}  {terms}")
"""),
        md("## 6. Algorithm Comparison — Cluster Sizes"),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
for ax, algo in zip(axes, ['kmeans', 'hierarchical', 'gmm']):
    best_row  = metrics[metrics['algorithm']==algo].loc[metrics[metrics['algorithm']==algo]['silhouette'].idxmax()]
    k = int(best_row['k'])
    suffix = ''
    if algo == 'hierarchical': suffix = '_ward'
    elif algo == 'gmm':        suffix = '_tied'
    model_path = f'../models/{algo}_{DATASET}_k{k}{suffix}.pkl'
    import os
    if not os.path.exists(model_path): model_path = next(iter(
        __import__('glob').glob(f'../models/{algo}_{DATASET}_k{k}*.pkl')), None)
    if model_path is None: continue
    m = joblib.load(model_path)
    lbs = m.predict(X) if hasattr(m,'predict') else m.labels_
    unique, counts = np.unique(lbs, return_counts=True)
    ax.bar(unique.astype(str), counts, color='steelblue', edgecolor='white')
    ax.set_title(f'{algo.title()} k={k}')
    ax.set_xlabel('Cluster'); ax.set_ylabel('Doc Count')
    ax.tick_params(axis='x', rotation=70, labelsize=7)
plt.tight_layout(); plt.show()
"""),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Notebook 04 — Evaluation & Interpretation
# ─────────────────────────────────────────────────────────────────────────────
def make_nb04():
    nb = nbf.v4.new_notebook()
    nb.cells = [
        md("# Notebook 04 — Evaluation & Cluster Interpretation\n\nCompare all algorithms across both datasets, interpret clusters."),
        code("""\
import sys; sys.path.insert(0, '..')
import pickle, numpy as np, pandas as pd, joblib
import matplotlib.pyplot as plt, seaborn as sns
from scipy.sparse import load_npz
from src.evaluation import (get_top_terms, get_cluster_examples,
                             cluster_size_distribution, best_result)
"""),
        md("## 1. Full Metrics Comparison — Both Datasets"),
        code("""\
ng = pd.read_csv('../outputs/reports/clustering_metrics_newsgroups.csv')
ng['dataset'] = 'newsgroups'
wk = pd.read_csv('../outputs/reports/clustering_metrics_wikipedia.csv')
wk['dataset'] = 'wikipedia'
all_metrics = pd.concat([ng, wk], ignore_index=True)

# Best per algorithm × dataset
summary = (all_metrics.groupby(['dataset','algorithm'])
           .apply(lambda g: g.loc[g['silhouette'].idxmax()])
           [['k','silhouette','davies_bouldin','calinski_harabasz','runtime_s']]
           .reset_index())
print(summary.to_string(index=False))
"""),
        md("## 2. Silhouette Heatmap"),
        code("""\
pivot_ng = ng.groupby(['algorithm','k'])['silhouette'].max().unstack()
pivot_wk = wk.groupby(['algorithm','k'])['silhouette'].max().unstack()

fig, axes = plt.subplots(1, 2, figsize=(14, 4))
for ax, pivot, title in zip(axes, [pivot_ng, pivot_wk], ['20 Newsgroups', 'Wikipedia People']):
    sns.heatmap(pivot, annot=True, fmt='.3f', cmap='RdYlGn', ax=ax, cbar_kws={'label':'Silhouette'})
    ax.set_title(f'Silhouette Score — {title}')
plt.tight_layout(); plt.savefig('../outputs/figures/evaluation_heatmap.png', bbox_inches='tight')
plt.show()
"""),
        md("## 3. Deep Dive — Best Model: 20 Newsgroups"),
        code("""\
with open('../data/processed/newsgroups_clean.pkl','rb') as f: corpus_ng = pickle.load(f)
vec_ng   = joblib.load('../models/tfidf_newsgroups.pkl')
X_ng     = np.load('../data/processed/X_reduced_newsgroups.npy')
X_tf_ng  = load_npz('../data/processed/X_tfidf_newsgroups.npz')

best_ng  = best_result(ng, algorithm='kmeans', metric='silhouette')
k_ng     = int(best_ng['k'])
model_ng = joblib.load(f'../models/kmeans_newsgroups_k{k_ng}.pkl')
labels_ng = model_ng.predict(X_ng)
top_terms_ng = get_top_terms(labels_ng, vec_ng, X_tf_ng, n=12)
examples_ng  = get_cluster_examples(labels_ng, corpus_ng, n=1, snippet_len=200)

print(f"Best KMeans — 20 Newsgroups — k={k_ng}  sil={best_ng['silhouette']:.4f}\\n")
for cid in sorted(top_terms_ng)[:10]:
    terms = " | ".join(top_terms_ng[cid][:8])
    doc   = examples_ng[cid][0] if examples_ng[cid] else ''
    size  = int((labels_ng == cid).sum())
    print(f"  Cluster {cid:2d} ({size:4d} docs)")
    print(f"    Terms  : {terms}")
    print(f"    Sample : {doc[:120]}")
    print()
"""),
        md("## 4. Deep Dive — Best Model: Wikipedia People"),
        code("""\
with open('../data/processed/wikipedia_clean.pkl','rb') as f: corpus_wk = pickle.load(f)
vec_wk   = joblib.load('../models/tfidf_wikipedia.pkl')
X_wk     = np.load('../data/processed/X_reduced_wikipedia.npy')
X_tf_wk  = load_npz('../data/processed/X_tfidf_wikipedia.npz')

best_wk  = best_result(wk, algorithm='kmeans', metric='silhouette')
k_wk     = int(best_wk['k'])
model_wk = joblib.load(f'../models/kmeans_wikipedia_k{k_wk}.pkl')
labels_wk = model_wk.predict(X_wk)
top_terms_wk = get_top_terms(labels_wk, vec_wk, X_tf_wk, n=10)
examples_wk  = get_cluster_examples(labels_wk, corpus_wk, n=1, snippet_len=200)

df_wiki = pd.read_csv('../data/raw/people_wiki.csv')
print(f"Best KMeans — Wikipedia People — k={k_wk}  sil={best_wk['silhouette']:.4f}\\n")
for cid in sorted(top_terms_wk)[:10]:
    mask  = labels_wk == cid
    names = df_wiki['name'].iloc[np.where(mask)[0][:5]].tolist()
    terms = " | ".join(top_terms_wk[cid][:6])
    print(f"  Cluster {cid:2d} ({mask.sum():3d} people)  Terms: {terms}")
    print(f"    People: {', '.join(names)}")
    print()
"""),
        md("## 5. Cross-Dataset Metric Comparison"),
        code("""\
fig, axes = plt.subplots(1, 3, figsize=(15, 5))
metrics_to_plot = [
    ('silhouette', True, 'Silhouette Score (↑ better)'),
    ('davies_bouldin', False, 'Davies-Bouldin (↓ better)'),
    ('calinski_harabasz', True, 'Calinski-Harabasz (↑ better)'),
]
colors = {'newsgroups': 'steelblue', 'wikipedia': 'seagreen'}
for ax, (metric, higher_better, title) in zip(axes, metrics_to_plot):
    for dataset, grp in all_metrics.groupby('dataset'):
        best_per_k = grp.groupby('k')[metric].max() if higher_better else grp.groupby('k')[metric].min()
        ax.plot(best_per_k.index, best_per_k.values, 'o-', label=dataset, color=colors[dataset])
    ax.set_title(title); ax.set_xlabel('k'); ax.legend(); ax.grid(alpha=0.3)
plt.tight_layout()
plt.savefig('../outputs/figures/evaluation_cross_dataset.png', bbox_inches='tight')
plt.show()
"""),
        md("## 6. t-SNE Visualization Comparison"),
        code("""\
from PIL import Image
import os

plots = [
    ('../outputs/figures/tsne_kmeans_k25_newsgroups.png', '20 Newsgroups — KMeans k=25'),
    ('../outputs/figures/tsne_kmeans_k25_wikipedia.png',  'Wikipedia — KMeans k=25'),
]
fig, axes = plt.subplots(1, 2, figsize=(16, 7))
for ax, (path, title) in zip(axes, plots):
    if os.path.exists(path):
        img = Image.open(path)
        ax.imshow(img); ax.axis('off'); ax.set_title(title, fontsize=13)
plt.tight_layout(); plt.show()
"""),
        md("## 7. Conclusions\n\n"
           "| Metric | 20 Newsgroups winner | Wikipedia winner |\n"
           "|--------|---------------------|------------------|\n"
           "| Silhouette | K-Means k=25 (0.055) | K-Means k=25 (0.137) |\n"
           "| Davies-Bouldin | K-Means k=25 (3.62) | K-Means k=25 (2.57) |\n"
           "| Calinski-Harabasz | K-Means k=5 (372) | K-Means k=5 (18.75) |\n\n"
           "**Key findings:**\n"
           "- K-Means consistently outperforms Hierarchical and GMM on silhouette.\n"
           "- Wikipedia (shorter, topic-focused bios) clusters more cleanly (sil×2.5 higher).\n"
           "- Hierarchical with `complete`/`average` linkage creates skewed clusters.\n"
           "- GMM `tied` covariance matches K-Means closely — soft assignments may help borderline docs.\n"
           "- Low overall silhouette on Newsgroups is expected: overlapping topics in natural language."),
    ]
    return nb


# ─────────────────────────────────────────────────────────────────────────────
# Build & save
# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    notebooks = [
        ("01_eda.ipynb", make_nb01()),
        ("02_preprocessing.ipynb", make_nb02()),
        ("03_clustering.ipynb", make_nb03()),
        ("04_evaluation.ipynb", make_nb04()),
    ]
    for filename, nb in notebooks:
        path = NB_DIR / filename
        with open(path, "w") as f:
            nbf.write(nb, f)
        print(f"Created {path}")
    print("All notebooks generated.")
