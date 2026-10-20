# -*- coding: utf-8 -*-
"""raw_state.py — 코어가 보내는 상태를 통째로 찍는다. 진단용."""

import json
import socket
import sys

port = int(sys.argv[1]) if len(sys.argv) > 1 else 9977
s = socket.create_connection(("127.0.0.1", port), timeout=8)
f = s.makefile("rwb")
while True:
    m = json.loads(f.readline())
    if "ok" in m:
        continue
    m.pop("jpeg", None)
    for k in sorted(m):
        print("  %-12s %s" % (k, m[k]))
    break
s.close()
