"""Шаг 4 и шаг 7. Второй уровень: бустинг на признаках и вероятностях текстовых моделей.

Запуск:  python src/stack_lgb.py <имя> [флаги]
Имя задаёт файлы результата: oof_<имя>.npy, test_<имя>.npy, submissions/sub_<имя>.csv.

Флаги:
  --base=a,b,c   какие вероятности первого уровня брать (файлы oof_a.npy / test_a.npy),
                 по умолчанию tfidf,host_char,path_char,full_sgd
  --ext          добавить признаки из сторонних источников (ext_feats.csv)
  --dns          добавить DNS-признаки хоста (dns_feats.csv)
  --rdap         добавить данные о регистрации домена (rdap_feats.csv)
  --knn          добавить признаки ближайших соседей (knn_feats.csv)
  --cat          учить CatBoost вместо LightGBM
  --seeds=N      усреднить N запусков с разными seed модели (разбиение на фолды не меняется)

Всегда добавляется out-of-fold кодирование зарегистрированного домена:
доля фишинга среди других строк train с тем же доменом (со сглаживанием).

Стеки, вошедшие в финальные решения:
  stack9s   --ext --dns --rdap --seeds=5 --base=tfidf_pl,host_char_pl,path_char_pl,full_sgd_pl
  stack9cat --ext --dns --rdap --cat     --base=tfidf_pl,host_char_pl,path_char_pl,full_sgd_pl
  stack10   --ext --dns --rdap --knn     --base=tfidf_pl,host_char_pl,path_char_pl,full_sgd_pl
  stack11   --ext --dns --rdap           --base=tfidf_pl2,host_char_pl2,path_char_pl2,full_sgd_pl2
"""
import os, re, sys, numpy as np, pandas as pd, lightgbm as lgb, tldextract
from urllib.parse import urlsplit
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import f1_score
sys.path.insert(0, 'src'); from features import build_v2

SEED = 42
BASE = [a.split('=', 1)[1].split(',') for a in sys.argv if a.startswith('--base=')]
BASE = BASE[0] if BASE else ['tfidf', 'host_char', 'path_char', 'full_sgd']
NSEED = int([a.split('=')[1] for a in sys.argv if a.startswith('--seeds=')][0]) if any(a.startswith('--seeds=') for a in sys.argv) else 1

tr = pd.read_csv('data/train.csv'); te = pd.read_csv('data/test.csv'); y = tr.result.values; n = len(tr)

# лексические признаки и признаки брендов
Xtr, Xte, regs = build_v2(tr.url, te.url)
Xte['tld'] = pd.Categorical(Xte['tld'].astype(str), categories=Xtr['tld'].cat.categories)

# вероятности моделей первого уровня: для train это out-of-fold значения
for b in BASE:
    Xtr['p_' + b] = np.load(f'oof_{b}.npy'); Xte['p_' + b] = np.load(f'test_{b}.npy')


def _host(u):
    s = str(u).strip(); s = s if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', s) else 'http://' + s
    try: return (urlsplit(s).hostname or '').lower()
    except ValueError: return ''


_ext = tldextract.TLDExtract(suffix_list_urls=())

if '--ext' in sys.argv:
    E = pd.read_csv('ext_feats.csv')
    Xtr = pd.concat([Xtr, E.iloc[:n].reset_index(drop=True)], axis=1); Xte = pd.concat([Xte, E.iloc[n:].reset_index(drop=True)], axis=1)

if '--dns' in sys.argv:
    D = pd.read_csv('dns_feats.csv').set_index('host')
    for X, urls in ((Xtr, tr.url), (Xte, te.url)):
        h = urls.map(_host)
        for c in ['a_cnt', 'resolved', 'cname', 'mx']: X['dns_' + c] = h.map(D[c]).fillna(-3).values   # -3: хост не проверялся

if '--rdap' in sys.argv:
    Rd = pd.read_csv('rdap_feats.csv').drop_duplicates('reg').set_index('reg')
    ref = pd.Timestamp('2026-10-13')   # дата сбора данных, от неё считается возраст домена в днях
    for c in ['created', 'expires', 'changed']:
        Rd[c + '_d'] = (ref - pd.to_datetime(Rd[c], errors='coerce', utc=True).dt.tz_localize(None)).dt.days
    for X, urls in ((Xtr, tr.url), (Xte, te.url)):
        r = urls.map(_host).map(lambda h: _ext(h).top_domain_under_public_suffix or h)
        X['rdap_code'] = r.map(Rd['code']).fillna(-9).values   # -9: домен не запрашивался
        X['rdap_n_status'] = r.map(Rd['n_status']).fillna(-9).values
        for c in ['created_d', 'expires_d', 'changed_d']: X['rdap_' + c] = r.map(Rd[c]).values   # NaN, если дата неизвестна

if '--knn' in sys.argv:
    Kf = pd.read_csv('knn_feats.csv')
    Xtr = pd.concat([Xtr, Kf.iloc[:n].reset_index(drop=True)], axis=1); Xte = pd.concat([Xte, Kf.iloc[n:].reset_index(drop=True)], axis=1)

# разбиение на фолды одно и то же во всех скриптах
folds = list(StratifiedKFold(5, shuffle=True, random_state=SEED).split(Xtr, y))

# out-of-fold кодирование домена: для валидационного фолда статистика считается по остальным фолдам
rtr, rte = regs.iloc[:n].reset_index(drop=True), regs.iloc[n:].reset_index(drop=True)


def te_map(idx):
    return pd.DataFrame({'r': rtr.iloc[idx].values, 'y': y[idx]}).groupby('r').y.agg(['sum', 'count'])


prior, k = y.mean(), 5   # сглаживание к общей доле фишинга, вес 5 строк
te_oof = np.full(n, np.nan); cnt_oof = np.zeros(n)
for i, j in folds:
    g = te_map(i); s = rtr.iloc[j].map(g['sum']); c = rtr.iloc[j].map(g['count']).fillna(0)
    te_oof[j] = ((s.fillna(0) + prior * k) / (c + k)).values; cnt_oof[j] = c.values
g = te_map(np.arange(n)); c = rte.map(g['count']).fillna(0)
Xtr['dom_te'], Xtr['dom_cnt'] = te_oof, cnt_oof
Xte['dom_te'], Xte['dom_cnt'] = ((rte.map(g['sum']).fillna(0) + prior * k) / (c + k)).values, c.values

P = dict(objective='binary', learning_rate=0.02, num_leaves=31, min_child_samples=30, feature_fraction=0.6,
         bagging_fraction=0.8, bagging_freq=1, lambda_l2=3.0, cat_smooth=20, verbose=-1, seed=SEED, n_jobs=8)

oof = np.zeros(n); pt = np.zeros(len(te))
for sd in range(NSEED):
    P['seed'] = SEED + sd
    for f, (i, j) in enumerate(folds):
        if '--cat' in sys.argv:
            from catboost import CatBoostClassifier
            cats = [c for c in Xtr.columns if str(Xtr[c].dtype) == 'category']
            Xa, Xb, Xt = (X.copy() for X in (Xtr.iloc[i], Xtr.iloc[j], Xte))
            for X in (Xa, Xb, Xt):
                for c in cats: X[c] = X[c].astype(str)   # CatBoost ждёт категории строками
            m = CatBoostClassifier(iterations=5000, learning_rate=0.03, depth=7, l2_leaf_reg=3, random_seed=SEED + sd, thread_count=8,
                                   eval_metric='Logloss', od_type='Iter', od_wait=200, verbose=False, cat_features=cats)
            m.fit(Xa, y[i], eval_set=(Xb, y[j]))
            pj, ptf, bi = m.predict_proba(Xb)[:, 1], m.predict_proba(Xt)[:, 1], m.get_best_iteration()
        else:
            # ранняя остановка по валидационному фолду
            m = lgb.train(P, lgb.Dataset(Xtr.iloc[i], y[i]), 5000, valid_sets=[lgb.Dataset(Xtr.iloc[j], y[j])],
                          callbacks=[lgb.early_stopping(200, verbose=False)])
            pj, ptf, bi = m.predict(Xtr.iloc[j]), m.predict(Xte), m.best_iteration
        oof[j] += pj / NSEED; pt += ptf / 5 / NSEED   # усреднение по фолдам и по seed
        print(sd, f, bi, round(f1_score(y[j], pj > .5), 4), flush=True)

ths = np.arange(.2, .8, .01); fs = [f1_score(y, oof > t) for t in ths]; t = ths[int(np.argmax(fs))]
print('OOF F1@0.5', round(f1_score(y, oof > .5), 4), 'best', round(max(fs), 4), 'thr', round(t, 2))
tag = [a for a in sys.argv[1:] if not a.startswith('--')][0] if len(sys.argv) > 1 else 'stack'
np.save(f'oof_{tag}.npy', oof); np.save(f'test_{tag}.npy', pt)
pd.DataFrame({'Id': np.arange(len(te)), 'Predicted': (pt > t).astype(int)}).to_csv(f'submissions/sub_{tag}.csv', index=False)
if '--cat' not in sys.argv:
    print(pd.Series(m.feature_importance('gain'), Xtr.columns).sort_values(ascending=False).head(20).round(0))
