"""Лексические признаки URL, считаются только по самой строке адреса.

build()    - около 90 признаков: длины, счётчики символов, энтропия, ключевые слова,
             бесплатные хостинги, сокращатели ссылок, IP вместо домена, punycode и т.п.
build_v2() - то же плюс признаки брендов (тайпсквоттинг, бренд в поддомене или пути)
             и частота домена/хоста по train и test вместе.
Модуль импортируется из stack_lgb.py.
"""
import math, re
from collections import Counter
from urllib.parse import urlsplit
import numpy as np, pandas as pd
import tldextract

# список публичных суффиксов берём из поставки tldextract, без обращения в сеть
_ext = tldextract.TLDExtract(suffix_list_urls=())

# слова, которые часто встречаются в фишинговых адресах
KW = ['login', 'signin', 'sign-in', 'logon', 'verify', 'verif', 'account', 'update', 'secure', 'security', 'bank', 'confirm',
      'password', 'webscr', 'paypal', 'apple', 'microsoft', 'office', 'outlook', 'amazon', 'netflix', 'wallet', 'support',
      'billing', 'invoice', 'unlock', 'suspend', 'auth', 'wp-', 'admin', 'includes', 'index.php', '.php', '.html', 'free',
      'bonus', 'lucky', 'gift', 'ebay', 'facebook', 'instagram', 'whatsapp', 'dhl', 'post', 'recover', 'session', 'token', 'mail']
# бесплатные хостинги и конструкторы сайтов, где фишинг размещают чаще всего
FREE_HOSTS = ['000webhost', 'weebly', 'web.app', 'firebaseapp', 'wix', 'blogspot', 'github.io', 'netlify', 'vercel', 'glitch',
              'herokuapp', 'repl.co', 'ngrok', 'duckdns', 'sharepoint', 'godaddysites', 'webcindario', 'workers.dev', 'pages.dev',
              'square.site', 'ipfs', 'r2.dev', 'firebasestorage', 'appspot', 'azurewebsites', 'cloudfront', 'mybluehost', 'flywheelsites']
# сокращатели ссылок
SHORT = ['bit.ly', 'tinyurl', 'goo.gl', 't.co', 'ow.ly', 'is.gd', 'cutt.ly', 'rb.gy', 'shorturl', 'tiny.cc', 'qrco.de', 'linktr.ee']


def entropy(s):
    # энтропия Шеннона по символам: случайные строки вроде хэшей дают высокое значение
    if not s: return 0.0
    c = Counter(s); n = len(s)
    return -sum(v / n * math.log2(v / n) for v in c.values())


def _safe_port(sp):
    # urlsplit бросает ValueError на некорректном порте, такие адреса тоже бывают
    try: sp.port; return True
    except ValueError: return False


def feats(url):
    u = url.strip()
    has_scheme = bool(re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', u))
    sp = urlsplit(u if has_scheme else 'http://' + u)
    host = (sp.hostname or '').lower()
    path, query = sp.path, sp.query
    e = _ext(host)
    dom, sub, suf = e.domain, e.subdomain, e.suffix   # домен второго уровня, поддомен, публичный суффикс
    low = u.lower()
    d = {
        'len': len(u), 'host_len': len(host), 'path_len': len(path), 'query_len': len(query), 'frag_len': len(sp.fragment),
        'https': int(low.startswith('https')), 'has_scheme': int(has_scheme), 'www': int(host.startswith('www.')),
        'n_dots_host': host.count('.'), 'n_hyph_host': host.count('-'), 'n_digit_host': sum(c.isdigit() for c in host),
        'n_sub': len(sub.split('.')) if sub else 0, 'sub_len': len(sub), 'dom_len': len(dom), 'suf_len': len(suf),
        'is_ip': int(bool(re.fullmatch(r'[\d.]+', host)) or host.startswith('[')), 'port': int(sp.port is not None) if _safe_port(sp) else 1,
        'n_slash': path.count('/'), 'n_qparams': query.count('&') + (1 if query else 0), 'n_eq': u.count('='),
        'n_at': u.count('@'), 'n_pct': u.count('%'), 'n_tilde': u.count('~'), 'n_dslash': u[8:].count('//'),
        'n_digits': sum(c.isdigit() for c in u), 'n_upper': sum(c.isupper() for c in u), 'non_ascii': sum(ord(c) > 127 for c in u),
        'punycode': int('xn--' in host), 'ent_url': entropy(u), 'ent_host': entropy(host), 'ent_dom': entropy(dom),
        'digit_ratio': sum(c.isdigit() for c in u) / max(len(u), 1), 'trailing_slash': int(u.endswith('/')),
        'path_depth': len([p for p in path.split('/') if p]), 'ends_ellipsis': int(u.endswith('...')),
        'free_host': int(any(h in host for h in FREE_HOSTS)), 'shortener': int(any(h in host for h in SHORT)),
        'http_in_path': int('http' in (path + query).lower()), 'amp_esc': int('&amp;' in u),
        'longest_tok': max((len(t) for t in re.split(r'[^A-Za-z0-9]', u)), default=0),
        'max_dom_digits_run': max((len(t) for t in re.findall(r'\d+', dom)), default=0),
        'ext_php': int(path.lower().endswith('.php')), 'ext_html': int(path.lower().endswith(('.html', '.htm'))),
        'kw_count': sum(k in low for k in KW),
    }
    for k in KW: d['kw_' + k] = int(k in low)   # по одному бинарному признаку на ключевое слово
    return d, suf, dom


def build(urls):
    rows, sufs, doms = [], [], []
    for u in urls:
        d, s, dm = feats(u); rows.append(d); sufs.append(s); doms.append(dm)
    X = pd.DataFrame(rows)
    X['tld'] = pd.Categorical(sufs)   # публичный суффикс как категориальный признак
    return X, pd.Series(doms)


# бренды, под которые чаще всего маскируется фишинг
BRANDS = ['paypal', 'apple', 'icloud', 'microsoft', 'office365', 'outlook', 'live', 'hotmail', 'amazon', 'netflix', 'facebook',
          'instagram', 'whatsapp', 'google', 'gmail', 'yahoo', 'ebay', 'chase', 'wellsfargo', 'bankofamerica', 'citibank', 'hsbc',
          'barclays', 'lloyds', 'santander', 'natwest', 'halifax', 'dhl', 'fedex', 'usps', 'ups', 'dropbox', 'docusign', 'adobe',
          'linkedin', 'twitter', 'steam', 'steamcommunity', 'roblox', 'binance', 'coinbase', 'metamask', 'blockchain', 'americanexpress',
          'amex', 'visa', 'mastercard', 'sberbank', 'tinkoff', 'alibaba', 'aliexpress', 'rakuten', 'mercari', 'aeon', 'smbc', 'mufg',
          'orange', 'ovh', 'att', 'verizon', 'comcast', 'xfinity', 'irs', 'gov', 'telegram', 'discord', 'spotify', 'wetransfer', 'onedrive',
          'sharepoint', 'yandex', 'mail', 'vk', 'avito', 'ozon', 'wildberries', 'gosuslugi', 'postbank', 'credit-agricole', 'itau',
          'bradesco', 'caixa', 'nubank', 'davivienda', 'bancolombia', 'interac', 'desjardins', 'tangerine', 'rbc', 'scotiabank']


def lev(a, b):
    # расстояние Левенштейна; если длины отличаются больше чем на 2, считать не нужно
    if a == b: return 0
    if abs(len(a) - len(b)) > 2: return 9
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        cur = [i]
        for j, cb in enumerate(b, 1):
            cur.append(min(prev[j] + 1, cur[j - 1] + 1, prev[j - 1] + (ca != cb)))
        prev = cur
    return prev[-1]


def brand_feats(url, dom, sub):
    low = url.lower(); d = dom.lower()
    in_url = [b for b in BRANDS if b in low]
    dists = [lev(d, b) for b in BRANDS if len(b) >= 4]
    md = min(dists) if dists else 9
    return {'brand_any': len(in_url),                       # сколько брендов упомянуто где угодно в URL
            'brand_is_domain': int(d in BRANDS),            # сам домен и есть бренд (скорее всего, легитимный сайт)
            'brand_not_domain': int(bool(in_url) and d not in BRANDS and not any(b == d for b in in_url)),  # бренд есть, но домен чужой
            'brand_in_sub': int(any(b in sub.lower() for b in BRANDS if len(b) >= 4)),       # бренд в поддомене
            'brand_in_dom_substr': int(any(b in d and b != d for b in BRANDS if len(b) >= 4)),  # бренд как часть домена
            'brand_lev_min': md, 'brand_typo': int(0 < md <= 2)}   # опечатка в 1-2 символа от бренда


def build_v2(urls_train, urls_test):
    """Признаки для train и test: лексика, бренды, частота домена и хоста по обеим выборкам."""
    allu = pd.concat([urls_train, urls_test]).reset_index(drop=True)
    X, doms = build(allu)
    subs, regs, hosts = [], [], []
    for u in allu:
        s = u.strip(); s = s if re.match(r'^[a-zA-Z][a-zA-Z0-9+.-]*://', s) else 'http://' + s
        try: h = (urlsplit(s).hostname or '').lower()
        except ValueError: h = ''
        e = _ext(h); subs.append(e.subdomain); regs.append(e.registered_domain or h); hosts.append(h)
    B = pd.DataFrame([brand_feats(u, d, s) for u, d, s in zip(allu, doms, subs)])
    X = pd.concat([X, B], axis=1)
    regs = pd.Series(regs); hosts = pd.Series(hosts)
    # сколько раз домен и хост встречаются в train и test вместе (меток не требует)
    X['reg_freq'] = regs.map(regs.value_counts()).values
    X['host_freq'] = hosts.map(hosts.value_counts()).values
    n = len(urls_train)
    return X.iloc[:n].reset_index(drop=True), X.iloc[n:].reset_index(drop=True), regs
