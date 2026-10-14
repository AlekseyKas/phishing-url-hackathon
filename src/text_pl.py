"""Шаг 5. Текстовые модели первого уровня, переобученные с псевдоразметкой теста.

В обучающую часть каждого фолда добавляются строки теста, для которых метка
известна надёжно: либо она есть в размеченных внешних датасетах (x_lab_url),
либо стек предыдущего шага уверен в ней на 99 и более процентов.
Метки train при этом не трогаем, оценка качества идёт по исходным меткам.

Результат: oof_<имя>_pl.npy и test_<имя>_pl.npy для четырёх моделей.
Четыре модели считаются параллельно в отдельных процессах.
"""
import re, sys, numpy as np, pandas as pd, scipy.sparse as sp
from urllib.parse import urlsplit
from joblib import Parallel, delayed
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

SEED = 42
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv'); y = tr.result.values; n = len(tr)

# какие строки теста берём в псевдоразметку и с какой меткой
E = pd.read_csv('ext_feats.csv').iloc[n:].reset_index(drop=True); p4 = np.load('test_stack4.npy')
pl_mask = (E.x_lab_url >= 0).values | (p4 > .99) | (p4 < .01)
pl_y = np.where(E.x_lab_url >= 0, (E.x_lab_url > .5).astype(int), (p4 > .5).astype(int))[pl_mask]
print('pseudo-labelled test rows:', pl_mask.sum(), 'pos rate', pl_y.mean().round(3), flush=True)


def split(u):
    u = u.strip(); s = u if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', u) else 'http://' + u
    try: p = urlsplit(s); return (p.netloc or '').lower(), p.path + ('?' + p.query if p.query else '')
    except ValueError: return u.lower(), ''


allu = pd.concat([tr.url, te.url]).reset_index(drop=True)
H = [split(u) for u in allu]
TEXT = {'full': allu, 'host': pd.Series([h for h, _ in H]), 'path': pd.Series([r for _, r in H])}
# те же четыре модели, что в baseline_tfidf.py и tfidf_models.py
MODELS = {
    'tfidf': ('full', [dict(analyzer='char', ngram_range=(1, 5), min_df=2, sublinear_tf=True, max_features=500_000, lowercase=False),
                       dict(analyzer='word', token_pattern=r'[A-Za-z0-9]+', ngram_range=(1, 2), min_df=2, sublinear_tf=True)], 'lr'),
    'host_char': ('host', [dict(analyzer='char_wb', ngram_range=(2, 6), min_df=2, sublinear_tf=True)], 'lr'),
    'path_char': ('path', [dict(analyzer='char', ngram_range=(2, 5), min_df=3, sublinear_tf=True, max_features=400_000)], 'lr'),
    'full_sgd': ('full', [dict(analyzer='char', ngram_range=(1, 6), min_df=2, sublinear_tf=True, lowercase=False, max_features=1_500_000)], 'sgd'),
}


def run(name):
    key, vps, kind = MODELS[name]
    X = sp.hstack([TfidfVectorizer(**vp).fit_transform(TEXT[key]) for vp in vps]).tocsr()
    Xtr, Xte = X[:n], X[n:]; Xpl = Xte[pl_mask]
    if kind == 'lr':
        mk = lambda: LogisticRegression(C=10 if name == 'tfidf' else 8, max_iter=3000, solver='liblinear')
    else:
        mk = lambda: SGDClassifier(loss='modified_huber', alpha=2e-6, max_iter=50, tol=None, random_state=SEED)
    oof = np.zeros(n); pt = np.zeros(len(te))
    for i, j in StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y):
        # обучающая часть фолда + псевдоразмеченный тест; валидационная часть фолда остаётся чистой
        m = mk().fit(sp.vstack([Xtr[i], Xpl]), np.r_[y[i], pl_y])
        oof[j] = m.predict_proba(Xtr[j])[:, 1]; pt += m.predict_proba(Xte)[:, 1] / 5
    np.save(f'oof_{name}_pl.npy', oof); np.save(f'test_{name}_pl.npy', pt)
    return name, round(f1_score(y, oof > .5), 4)


for r in Parallel(n_jobs=4)(delayed(run)(nm) for nm in MODELS): print(r, flush=True)
