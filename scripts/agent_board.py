#!/usr/bin/env python3
"""Minimal machine interface for AGENT_BOARD.jsonl."""
import argparse
import json
import re
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path

BOARD = Path(__file__).resolve().parents[1] / "AGENT_BOARD.jsonl"
TYPES = {"MSG","ASK","ACK","CLM","DONE","BLK","DEC","CORR"}
AGENTS = {"F","N"}
KEYS = {"v","id","ts","from","to","type","ref","scope","msg"}
TS_RE = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}Z$")
ID_RE = re.compile(r"^[FN]-\d{8}T\d{6}Z-(?:[0-9a-f]{8}|\d{2})$")  # accepts migrated v0 IDs


def load():
    records=[]; ids=set()
    for n,line in enumerate(BOARD.read_text().splitlines(),1):
        try: r=json.loads(line)
        except Exception as e: raise SystemExit(f"line {n}: invalid JSON: {e}")
        if set(r) != KEYS: raise SystemExit(f"line {n}: keys={sorted(r)}")
        if r["v"] != 1: raise SystemExit(f"line {n}: unsupported v")
        if not ID_RE.match(r["id"]): raise SystemExit(f"line {n}: bad id")
        if not TS_RE.match(r["ts"]): raise SystemExit(f"line {n}: bad ts")
        if r["from"] not in AGENTS or r["to"] not in AGENTS|{"*"}: raise SystemExit(f"line {n}: bad agent")
        if r["type"] not in TYPES: raise SystemExit(f"line {n}: bad type")
        if not isinstance(r["scope"],str) or not r["scope"]: raise SystemExit(f"line {n}: bad scope")
        if not isinstance(r["msg"],str) or "\n" in r["msg"]: raise SystemExit(f"line {n}: bad msg")
        if r["id"] in ids: raise SystemExit(f"line {n}: duplicate id")
        if r["ref"] is not None and r["ref"] not in ids: raise SystemExit(f"line {n}: ref must point backward")
        if r["type"] == "CORR" and r["ref"] is None: raise SystemExit(f"line {n}: CORR requires ref")
        ids.add(r["id"]); records.append(r)
    return records


def append(args):
    records=load()
    if args.ref and args.ref not in {r['id'] for r in records}: raise SystemExit("ref not found")
    now=datetime.now(timezone.utc).replace(microsecond=0)
    compact=now.strftime("%Y%m%dT%H%M%SZ")
    r={"v":1,"id":f"{args.sender}-{compact}-{uuid.uuid4().hex[:8]}","ts":now.strftime("%Y-%m-%dT%H:%M:%SZ"),"from":args.sender,"to":args.to,"type":args.type,"ref":args.ref,"scope":args.scope,"msg":args.msg}
    if r['type']=='CORR' and not r['ref']: raise SystemExit("CORR requires --ref")
    with BOARD.open('a', encoding='utf-8') as f: f.write(json.dumps(r,separators=(',',':'),ensure_ascii=False)+'\n')
    print(r['id'])


def main():
    ap=argparse.ArgumentParser()
    sp=ap.add_subparsers(dest='cmd',required=True)
    sp.add_parser('validate')
    t=sp.add_parser('tail'); t.add_argument('-n',type=int,default=10); t.add_argument('--agent',choices=['F','N']); t.add_argument('--scope')
    a=sp.add_parser('append'); a.add_argument('--from',dest='sender',choices=['F','N'],required=True); a.add_argument('--to',choices=['F','N','*'],required=True); a.add_argument('--type',choices=sorted(TYPES),required=True); a.add_argument('--ref'); a.add_argument('--scope',default='repo'); a.add_argument('--msg',required=True)
    args=ap.parse_args()
    if args.cmd=='validate': print(f"OK {len(load())}")
    elif args.cmd=='tail':
        rs=load()
        if args.agent: rs=[r for r in rs if r['from']==args.agent or r['to']==args.agent or r['to']=='*']
        if args.scope: rs=[r for r in rs if r['scope']==args.scope]
        for r in rs[-args.n:]: print(json.dumps(r,separators=(',',':'),ensure_ascii=False))
    else: append(args)

if __name__=='__main__': main()
