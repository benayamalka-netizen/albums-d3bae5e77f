"""Incremental face grouping for the albums site.

Decrypts catalog.bin with ALBUM_PASSWORD, finds photos not yet scanned, downloads only those,
detects + embeds faces, re-clusters everything, keeps each person's id stable, and re-encrypts
people.bin (what the site shows) and faces.bin (the stored embeddings).

Usage: ALBUM_PASSWORD=... python tools/update_faces.py [site_dir]
"""
import os, sys, json, gzip, hashlib, base64, ssl, urllib.request, concurrent.futures as cf
import numpy as np, cv2
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

SITE = os.path.abspath(sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), '..'))
MODELS = os.path.join(SITE, 'tools', 'models')
pw = os.environ['ALBUM_PASSWORD'].strip()
salt = open(os.path.join(SITE, 'salt.bin'), 'rb').read()
KEY = hashlib.pbkdf2_hmac('sha256', pw.encode('utf8'), salt, 200000, 32)

def dec(name, gz=False):
    p = os.path.join(SITE, name)
    if not os.path.exists(p): return None
    b = open(p, 'rb').read(); d = AESGCM(KEY).decrypt(b[:12], b[12:], None)
    return json.loads(gzip.decompress(d) if gz else d)

def enc(obj, name, gz=False):
    d = json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf8')
    if gz: d = gzip.compress(d, 9)
    iv = os.urandom(12)
    open(os.path.join(SITE, name), 'wb').write(iv + AESGCM(KEY).encrypt(iv, d, None))

A = dec('catalog.bin')
store = dec('faces.bin', gz=True) or {'scanned': [], 'faces': []}
scanned = set(store['scanned'])
photo_ids = [f[0] for a in A for f in a['f'] if not f[1]]
live = set(photo_ids)
new = [i for i in photo_ids if i not in scanned]
print(f'{len(photo_ids)} photos, {len(new)} new')

# ---- scan new photos ----
ctx = ssl.create_default_context()
def fetch(i):
    for u in (f'https://drive.google.com/thumbnail?id={i}&sz=w1000', f'https://lh3.googleusercontent.com/d/{i}=w1000'):
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30, context=ctx).read()
            im = cv2.imdecode(np.frombuffer(r, np.uint8), 1)
            if im is not None and im.shape[0] > 50: return i, im
        except Exception: pass
    return i, None

det = cv2.FaceDetectorYN.create(os.path.join(MODELS, 'yunet.onnx'), '', (320, 320), 0.85)
rec = cv2.FaceRecognizerSF.create(os.path.join(MODELS, 'sface.onnx'), '')
crops = {}  # photo id -> image, kept only while needed for avatars
with cf.ThreadPoolExecutor(8) as ex:
    for n, (i, im) in enumerate(ex.map(fetch, new)):
        if im is None: continue  # retried next run
        h, w = im.shape[:2]; det.setInputSize((w, h)); _, fs = det.detect(im)
        for f in (fs if fs is not None else []):
            if min(f[2], f[3]) < 36: continue
            e = rec.feature(rec.alignCrop(im, f)).flatten(); e /= np.linalg.norm(e)
            store['faces'].append({'id': i, 'b': [round(float(f[0]) / w, 4), round(float(f[1]) / h, 4), round(float(f[2]) / w, 4), round(float(f[3]) / h, 4)],
                                   's': round(float(f[14]), 3), 'e': [round(float(x), 5) for x in e]})
        scanned.add(i)
        if n % 200 == 0: print('scanned', n, flush=True)

store['faces'] = [f for f in store['faces'] if f['id'] in live]  # drop removed photos
store['scanned'] = sorted(scanned & live)
if not new and os.path.exists(os.path.join(SITE, 'people.bin')):
    print('nothing new'); sys.exit(0)

# ---- cluster ----
from sklearn.cluster import AgglomerativeClustering
F = store['faces']; E = np.array([f['e'] for f in F], np.float32)
lab = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.6).fit_predict(E)
groups = {}
for f, l in zip(F, lab): groups.setdefault(int(l), []).append(f)

old = dec('people.bin') or []
old_of = {}  # (photo id, face box) -> old person key
for p in old:
    for pid, b in p['p']: old_of[(pid, tuple(b))] = p.get('k', old.index(p))
next_k = max([p.get('k', n) for n, p in enumerate(old)] + [-1]) + 1
old_av = {p.get('k', n): p for n, p in enumerate(old)}
used = set(); P = []
for g in sorted(groups.values(), key=len, reverse=True):
    photos = {}
    for f in g: photos.setdefault(f['id'], f)
    if len(photos) < 4: continue
    votes = {}
    for pid, f in photos.items():
        ok = old_of.get((pid, tuple(f['b'])))
        if ok is not None: votes[ok] = votes.get(ok, 0) + 1
    k = max(votes, key=votes.get) if votes else None
    if k is None or k in used or votes[k] < len(photos) * 0.3: k = next_k; next_k += 1
    used.add(k)
    c = np.mean([f['e'] for f in g], 0); c /= np.linalg.norm(c)
    best = max(g, key=lambda f: float(np.dot(f['e'], c)) * min(1, f['b'][2] * 6) * f['s'])
    o = old_av.get(k)
    if o and o.get('a') and o['c'] in photos:  # keep the familiar face picture
        P.append({'k': k, 'c': o['c'], 'b': o['b'], 'a': o['a'], 'p': [[pid, v['b']] for pid, v in photos.items()]}); continue
    P.append({'k': k, 'c': best['id'], 'b': best['b'], 'p': [[pid, v['b']] for pid, v in photos.items()]})

# ---- avatars (small face crops embedded in the encrypted file) ----
def avatar(pid, b):
    _, im = fetch(pid)
    if im is None: return None
    h, w = im.shape[:2]; fw, fh = b[2] * w, b[3] * h; s = max(fw, fh) * 1.7; S = int(s)
    x0, y0 = int(b[0] * w + fw / 2 - s / 2), int(b[1] * h + fh / 2 - s / 2)
    pad = cv2.copyMakeBorder(im, S, S, S, S, cv2.BORDER_REPLICATE)
    c = cv2.resize(pad[y0 + S:y0 + 2 * S, x0 + S:x0 + 2 * S], (120, 120), interpolation=cv2.INTER_AREA)
    return 'data:image/jpeg;base64,' + base64.b64encode(cv2.imencode('.jpg', c, [cv2.IMWRITE_JPEG_QUALITY, 72])[1]).decode()
need = []
for p in P:
    if not p.get('a'): need.append(p)
with cf.ThreadPoolExecutor(8) as ex:
    for p, a in zip(need, ex.map(lambda p: avatar(p['c'], p['b']), need)): p['a'] = a or ''
print(f'{len(P)} people, {len(need)} new avatars')

if P != old: enc(P, 'people.bin')
if store != (dec('faces.bin', gz=True) or {}): enc(store, 'faces.bin', gz=True)  # random IV: write only on real change
