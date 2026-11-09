#!/usr/bin/env python3
"""Rebuild the encrypted catalog.bin, leaving out hidden albums.

Usage:
  python3 tools/build_catalog.py SOURCE.json HIDDEN.json OUT_DIR

SOURCE.json  full plaintext catalog (keep it OUTSIDE the repo)
HIDDEN.json  list of album ids to hide, e.g. ["1-FBy...", "..."]
OUT_DIR      where catalog.bin is written (the repo root)

Prints the new share link hash. Old links stop working because the key changes.
"""
import base64, json, os, sys
from cryptography.hazmat.primitives.ciphers.aead import AESGCM

src, hidden_path, out_dir = sys.argv[1:4]
albums = json.load(open(src, encoding="utf-8"))
hidden = set(json.load(open(hidden_path, encoding="utf-8")))
visible = [a for a in albums if a["id"] not in hidden]

key = AESGCM.generate_key(bit_length=128)
iv = os.urandom(12)
ct = AESGCM(key).encrypt(iv, json.dumps(visible, ensure_ascii=False).encode("utf-8"), None)
open(os.path.join(out_dir, "catalog.bin"), "wb").write(iv + ct)

print(f"albums total={len(albums)} hidden={len(albums) - len(visible)} visible={len(visible)}")
print("share hash:", base64.urlsafe_b64encode(key).decode().rstrip("="))
