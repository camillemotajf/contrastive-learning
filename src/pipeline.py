"""Pipeline de features SEM data leakage para detecção de tráfego (bots vs unsafe).

Diferenças em relação ao src/datasets.py original:
  - Split estratificado treino/teste ANTES de qualquer ajuste.
  - TF-IDF, TruncatedSVD e o scaler Z-score são ajustados (fit) APENAS no treino
    e aplicados (transform) no teste. Sem isso, as métricas de teste vazam.
  - Toggles de ablação para auditar o risco de "atalhos":
      * use_manual ........ inclui as 4 features de tamanho/contagem
      * canonicalize ...... reordena as chaves dos headers e re-serializa,
                            removendo o artefato de ordem/serialização
  - Os rótulos vêm do campo `decision` do JSON (decisão de um sistema
    existente). Aprender essa tarefa = imitar esse sistema, não verdade absoluta.
"""
import json
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD


def _canonicalize(headers_str):
    """Reordena chaves e re-serializa de forma estável para anular o artefato
    de ordenação/espaçamento. Se o parse falhar, devolve a string original."""
    try:
        d = json.loads(headers_str)
        return json.dumps(d, sort_keys=True, separators=(",", ":"))
    except Exception:
        return headers_str


def load_raw(file_unsafe, file_bots):
    """Lê os JSONs e devolve listas paralelas de strings brutas e rótulos.

    label 0 = unsafe (humano), label 1 = bots.
    """
    headers, requests, labels = [], [], []

    def _load(path, label):
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        for item in data:
            headers.append(item.get("headers", "{}"))
            requests.append(item.get("request", "{}"))
            labels.append(label)

    _load(file_unsafe, 0)
    _load(file_bots, 1)
    return headers, requests, np.array(labels, dtype=np.int64)


def _manual_features(headers, requests):
    feats = np.array(
        [
            [len(h), len(r), h.count(":"), r.count(":")]
            for h, r in zip(headers, requests)
        ],
        dtype=np.float32,
    )
    return feats


def build_features(
    file_unsafe,
    file_bots,
    test_size=0.3,
    seed=42,
    svd_dim=64,
    use_manual=True,
    canonicalize=False,
):
    """Devolve X_train, X_test, y_train, y_test sem leakage.

    Todo o `fit` (TF-IDF, SVD, Z-score) ocorre apenas no conjunto de treino.
    """
    headers, requests, labels = load_raw(file_unsafe, file_bots)

    if canonicalize:
        headers = [_canonicalize(h) for h in headers]

    idx = np.arange(len(labels))
    idx_tr, idx_te = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=labels
    )

    def subset(lst, ids):
        return [lst[i] for i in ids]

    h_tr, h_te = subset(headers, idx_tr), subset(headers, idx_te)
    r_tr, r_te = subset(requests, idx_tr), subset(requests, idx_te)
    y_tr, y_te = labels[idx_tr], labels[idx_te]

    # --- TF-IDF de char n-grams: FIT só no treino ---
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), max_features=5000)
    Xtr_tfidf = vec.fit_transform(h_tr)
    Xte_tfidf = vec.transform(h_te)

    svd = TruncatedSVD(n_components=svd_dim, random_state=seed)
    Xtr = svd.fit_transform(Xtr_tfidf)
    Xte = svd.transform(Xte_tfidf)

    if use_manual:
        m_tr = _manual_features(h_tr, r_tr)
        m_te = _manual_features(h_te, r_te)
        Xtr = np.hstack([Xtr, m_tr])
        Xte = np.hstack([Xte, m_te])

    # --- Z-score: FIT só no treino ---
    mean = Xtr.mean(axis=0)
    std = Xtr.std(axis=0) + 1e-8
    Xtr = ((Xtr - mean) / std).astype(np.float32)
    Xte = ((Xte - mean) / std).astype(np.float32)

    return Xtr, Xte, y_tr, y_te
