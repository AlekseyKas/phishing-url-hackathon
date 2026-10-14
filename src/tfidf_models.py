"""Шаг 2. Ещё три линейные текстовые модели на разных частях URL.

host_char - символьные n-граммы только по хосту,
path_char - только по пути и строке запроса,
full_sgd  - широкие символьные n-граммы по всему URL, обучение через SGD.

Для каждой модели сохраняются oof_<имя>.npy и test_<имя>.npy.
Можно передать имена моделей аргументами, по умолчанию считаются все три.
"""
import sys, re, numpy as np, pandas as pd, scipy.sparse as sp
from urllib.parse import urlsplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

SEED = 42
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv'); y = tr.result.values


def split(u):
    # делим URL на хост и остаток (путь + запрос); если схемы нет, подставляем http://
    u = u.strip(); s = u if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', u) else 'http://' + u
    try: p = urlsplit(s); return (p.netloc or '').lower(), p.path + ('?' + p.query if p.query else '')
    except ValueError: return u.lower(), ''


H = [split(u) for u in pd.concat([tr.url, te.url])]
host = pd.Series([h for h, _ in H]); rest = pd.Series([r for _, r in H]); full = pd.concat([tr.url, te.url]).reset_index(drop=True)
n = len(tr)

# имя модели: (текст, параметры TfidfVectorizer, тип классификатора)
MODELS = {
    'host_char': (host, dict(analyzer='char_wb', ngram_range=(2, 6), min_df=2, sublinear_tf=True), 'lr'),
    'path_char': (rest, dict(analyzer='char', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=400_000), 'lr'),
    'full_sgd': (full, dict(analyzer='char', ngram_range=(1, 6), min_df=2, sublinear_tf=True, lowercase=False, max_features=1_500_000), 'sgd'),
}
names = sys.argv[1:] or list(MODELS)
for name in names:
    text, vp, kind = MODELS[name]
    # словарь n-грамм строится по train и test вместе: меток здесь нет, это не утечка
    X = TfidfVectorizer(**vp).fit_transform(text).tocsr()
    Xtr, Xte = X[:n], X[n:]
    if kind == 'lr':
        mk = lambda: LogisticRegression(C=8, max_iter=2000, solver='liblinear')
    else:
        mk = lambda: SGDClassifier(loss='modified_huber', alpha=2e-6, max_iter=50, tol=None, random_state=SEED)
    oof = np.zeros(n); pt = np.zeros(len(te))
    for f, (i, j) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y)):
        m = mk().fit(Xtr[i], y[i]); oof[j] = m.predict_proba(Xtr[j])[:, 1]; pt += m.predict_proba(Xte)[:, 1] / 5
    print(name, X.shape, 'OOF F1', round(f1_score(y, oof > .5), 4), flush=True)
    np.save(f'oof_{name}.npy', oof); np.save(f'test_{name}.npy', pt)
