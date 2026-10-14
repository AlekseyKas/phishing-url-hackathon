#!/usr/bin/env bash
# Полный прогон решения. Запускать из корня проекта, где лежат папки data/ и src/.
# Нужны data/train.csv и data/test.csv с платформы. Около 2 часов на 16 ядрах.
#
# Шаги 3 и 6 (сторонние данные, DNS, RDAP) по умолчанию пропускаются: их результаты
# ext_feats.csv, dns_feats.csv, rdap_feats.csv приложены к пакету и меняются со временем.
# Чтобы пересчитать их заново, запустите с переменной REBUILD_EXT=1.
set -e
PY=${PY:-.venv/bin/python}
mkdir -p submissions

$PY src/baseline_tfidf.py                                   # шаг 1
$PY src/tfidf_models.py                                     # шаг 2
if [ "${REBUILD_EXT:-0}" = "1" ]; then
  $PY src/ext_feats.py                                      # шаг 3, нужна папка ext/ с данными
fi
$PY src/stack_lgb.py stack4 --ext                           # шаг 4
$PY src/text_pl.py                                          # шаг 5
if [ "${REBUILD_EXT:-0}" = "1" ]; then
  $PY src/dns_feats.py                                      # шаг 6, живые проверки
  $PY src/rdap_feats.py
fi
$PY src/knn_feats.py                                        # шаг 6, соседи

B=tfidf_pl,host_char_pl,path_char_pl,full_sgd_pl            # шаг 7
$PY src/stack_lgb.py stack9s  --ext --dns --rdap --seeds=5 --base=$B
$PY src/stack_lgb.py stack9cat --ext --dns --rdap --cat --base=$B
$PY src/stack_lgb.py stack10  --ext --dns --rdap --knn --base=$B
$PY src/text_pl2.py test_stack9s.npy
$PY src/stack_lgb.py stack11  --ext --dns --rdap --base=tfidf_pl2,host_char_pl2,path_char_pl2,full_sgd_pl2

$PY src/blend.py blend3 stack9s stack9cat stack10           # шаг 8
$PY src/blend.py blend4 stack9s stack9cat stack10 stack11
echo "готово: submissions/sub_blend4.csv"
