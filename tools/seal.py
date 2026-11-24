"""Encrypt catalog + people with a key derived from a password. Usage: python seal.py <site_dir> <password>"""
import sys, os, json, hashlib
from cryptography.hazmat.primitives.ciphers.aead import AESGCM
site, pw = sys.argv[1], sys.argv[2]
D = os.path.dirname(os.path.abspath(__file__))
sp = os.path.join(site, 'salt.bin')
# keep the salt so re-sealing with the same password does not log visitors out; pass --new to force
salt = open(sp, 'rb').read() if os.path.exists(sp) and '--new' not in sys.argv else os.urandom(16)
key = hashlib.pbkdf2_hmac('sha256', pw.strip().encode('utf8'), salt, 200000, 32)
def seal(obj, name):
    iv = os.urandom(12)
    open(os.path.join(site, name), 'wb').write(iv + AESGCM(key).encrypt(iv, json.dumps(obj, ensure_ascii=False, separators=(',', ':')).encode('utf8'), None))
open(os.path.join(site, 'salt.bin'), 'wb').write(salt)
seal(json.load(open(os.path.join(D, '..', 'albums_catalog.json'), encoding='utf8')), 'catalog.bin')
pp = os.path.join(D, 'people.json')
if os.path.exists(pp): seal(json.load(open(pp, encoding='utf8')), 'people.bin')
print('sealed', os.listdir(site))
