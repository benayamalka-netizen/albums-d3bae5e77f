"""Download album thumbnails, detect faces (YuNet), embed (SFace), cluster per person."""
import json, os, sys, urllib.request, ssl, concurrent.futures as cf
import numpy as np, cv2

D = os.path.dirname(os.path.abspath(__file__))
IMG = os.path.join(D, 'img'); os.makedirs(IMG, exist_ok=True)
A = json.load(open(os.path.join(D, '..', 'albums_catalog.json'), encoding='utf8'))
ids = [f[0] for a in A for f in a['f'] if not f[1]]
ctx = ssl.create_default_context()

def dl(i):
    p = os.path.join(IMG, i + '.jpg')
    if os.path.exists(p) and os.path.getsize(p) > 1000: return
    for u in (f'https://drive.google.com/thumbnail?id={i}&sz=w1000', f'https://lh3.googleusercontent.com/d/{i}=w1000'):
        try:
            r = urllib.request.urlopen(urllib.request.Request(u, headers={'User-Agent': 'Mozilla/5.0'}), timeout=30, context=ctx).read()
            if len(r) > 1000: open(p, 'wb').write(r); return
        except Exception: pass

if 'dl' in sys.argv:
    with cf.ThreadPoolExecutor(16) as ex:
        for n, _ in enumerate(ex.map(dl, ids)):
            if n % 500 == 0: print('dl', n, flush=True)
    print('downloaded', len(os.listdir(IMG)))

if 'embed' in sys.argv:
    det = cv2.FaceDetectorYN.create(os.path.join(D, 'models/yunet.onnx'), '', (320, 320), 0.85)
    rec = cv2.FaceRecognizerSF.create(os.path.join(D, 'models/sface.onnx'), '')
    out = []
    for n, i in enumerate(ids):
        p = os.path.join(IMG, i + '.jpg')
        im = cv2.imdecode(np.fromfile(p, np.uint8), 1) if os.path.exists(p) else None
        if im is None: continue
        h, w = im.shape[:2]
        det.setInputSize((w, h)); _, fs = det.detect(im)
        for f in (fs if fs is not None else []):
            fw, fh = f[2], f[3]
            if min(fw, fh) < 36: continue  # too small to recognise reliably
            e = rec.feature(rec.alignCrop(im, f)).flatten()
            out.append({'id': i, 'b': [round(float(f[0]) / w, 4), round(float(f[1]) / h, 4), round(float(fw) / w, 4), round(float(fh) / h, 4)], 's': float(f[14]), 'e': (e / np.linalg.norm(e)).tolist()})
        if n % 500 == 0: print('embed', n, len(out), flush=True)
    json.dump(out, open(os.path.join(D, 'faces.json'), 'w'))
    print('faces', len(out))

if 'cluster' in sys.argv:
    from sklearn.cluster import AgglomerativeClustering
    F = json.load(open(os.path.join(D, 'faces.json')))
    E = np.array([f['e'] for f in F], np.float32)
    lab = AgglomerativeClustering(n_clusters=None, metric='cosine', linkage='average', distance_threshold=0.6).fit_predict(E)
    groups = {}
    for f, l in zip(F, lab): groups.setdefault(int(l), []).append(f)
    P = []
    for g in groups.values():
        photos = {}
        for f in g: photos.setdefault(f['id'], f)
        if len(photos) < 4: continue  # only recurring people
        c = np.mean([f['e'] for f in g], 0); c /= np.linalg.norm(c)
        best = max(g, key=lambda f: float(np.dot(f['e'], c)) * min(1, f['b'][2] * 6) * f['s'])
        P.append({'c': best['id'], 'b': best['b'], 'p': [[k, v['b']] for k, v in photos.items()]})
    P.sort(key=lambda p: -len(p['p']))
    json.dump(P, open(os.path.join(D, 'people.json'), 'w', encoding='utf8'))
    print('people', len(P), [len(p['p']) for p in P[:20]])
