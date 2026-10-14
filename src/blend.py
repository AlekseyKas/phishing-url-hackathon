"""Шаг 8. Финальные файлы ответов: среднее вероятностей нескольких стеков, порог 0.5.

  python src/blend.py blend3 stack9s stack9cat stack10
  python src/blend.py blend4 stack9s stack9cat stack10 stack11

Первый аргумент - имя результата, дальше имена стеков (файлы test_<имя>.npy и oof_<имя>.npy).
Печатает F1 смеси на кросс-валидации и пишет submissions/sub_<имя>.csv.
"""
import sys, numpy as np, pandas as pd
from sklearn.metrics import f1_score

name, parts = sys.argv[1], sys.argv[2:]
y = pd.read_csv('data/train.csv').result.values
oof = np.mean([np.load(f'oof_{k}.npy') for k in parts], 0)
p = np.mean([np.load(f'test_{k}.npy') for k in parts], 0)
print(name, 'из', parts, 'OOF F1', round(f1_score(y, oof > .5), 4))
np.save(f'oof_{name}.npy', oof); np.save(f'test_{name}.npy', p)
pd.DataFrame({'Id': np.arange(len(p)), 'Predicted': (p > .5).astype(int)}).to_csv(f'submissions/sub_{name}.csv', index=False)
