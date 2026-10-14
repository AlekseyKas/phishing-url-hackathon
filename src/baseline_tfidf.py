"""Шаг 1. Базовая текстовая модель: TF-IDF по символам и словам URL + логистическая регрессия.

Считает out-of-fold вероятности для train (5 фолдов) и вероятности для test,
сохраняет их в oof_tfidf.npy и test_tfidf.npy. Эти файлы потом идут на вход
второму уровню (stack_lgb.py).
"""
import numpy as np, pandas as pd, scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score

SEED = 42
tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv')
y = tr.result.values


def vecs():
    # два векторизатора: символьные n-граммы 1..5 (регистр сохраняем, он информативен)
    # и словесные n-граммы 1..2 по буквенно-цифровым токенам
    return [TfidfVectorizer(analyzer='char', ngram_range=(1, 5), min_df=2, sublinear_tf=True, max_features=500_000, lowercase=False),
            TfidfVectorizer(analyzer='word', token_pattern=r'[A-Za-z0-9]+', ngram_range=(1, 2), min_df=2, sublinear_tf=True)]


def fit_transform(a, b):
    # векторизаторы обучаются только на обучающей части фолда, чтобы не было утечки
    vs = vecs()
    return sp.hstack([v.fit_transform(a) for v in vs]).tocsr(), sp.hstack([v.transform(b) for v in vs]).tocsr()


# out-of-fold предсказания: каждая строка train получает вероятность от модели,
# которая эту строку не видела
oof = np.zeros(len(tr))
for f, (i, j) in enumerate(StratifiedKFold(5, shuffle=True, random_state=SEED).split(tr, y)):
    Xa, Xb = fit_transform(tr.url.iloc[i], tr.url.iloc[j])
    m = LogisticRegression(C=10, max_iter=3000, solver='liblinear').fit(Xa, y[i])
    oof[j] = m.predict_proba(Xb)[:, 1]
    print(f, f1_score(y[j], oof[j] > .5))

# подбор порога по F1 нужен только для первого сабмита, дальше порог 0.5
ths = np.arange(.2, .8, .01); fs = [f1_score(y, oof > t) for t in ths]
t = ths[int(np.argmax(fs))]
print('OOF F1@0.5', f1_score(y, oof > .5), 'best', max(fs), 'thr', t)
np.save('oof_tfidf.npy', oof)

# для теста модель обучается на всём train
Xa, Xb = fit_transform(tr.url, te.url)
p = LogisticRegression(C=10, max_iter=3000, solver='liblinear').fit(Xa, y).predict_proba(Xb)[:, 1]
np.save('test_tfidf.npy', p)
pd.DataFrame({'Id': np.arange(len(te)), 'Predicted': (p > t).astype(int)}).to_csv('submissions/sub_tfidf_lr.csv', index=False)
