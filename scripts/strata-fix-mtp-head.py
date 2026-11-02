#!/usr/bin/env python3
# Copyright (c) llamaMan. Licensed under the Elastic License 2.0 - see LICENSE.

"""Make a Qwen3.6-35B-A3B GGUF whose expert weights Strata's kernels refuse
loadable, by rewriting only those expert tensors as Q8_0.

The case it was written for: peculiar-ragdoll's Tiel-Coder / Cyber-Tiel GGUFs
graft an MTP head onto block 40 and quantize its experts to Q3_K in every
tier. Strata's native expert kernels take Q4_K, Q5_K, Q6_K and Q8_0 only, so
the engine dies at load with "blk.40.experts are Q3_K/Q3_K, which the native
expert kernels do not take".

Every other byte is copied unchanged: the metadata, every other tensor, and
their order. Each refused tensor is dequantized and re-quantized to Q8_0,
which keeps its values to within Q8_0's rounding (it can't add back what Q3_K
lost). The file grows by roughly 0.5 GB.

    pip install numpy gguf
    python3 strata-fix-mtp-head.py Tiel-Coder-35B-A3B-MTP-UD-Q4_K_XL.gguf

writes Tiel-Coder-35B-A3B-MTP-Strata-UD-Q4_K_XL.gguf next to it. The name
still ends in the size, so llamaMan offers it for Strata's UD-Q4_K_XL slot.
Add --verify to read the result back and check it.
"""

import argparse
import os
import re
import struct
import sys

import numpy as np

try:
    from gguf import GGML_QUANT_SIZES, GGMLQuantizationType
    from gguf.quants import dequantize, quantize
except ImportError:
    sys.exit("needs the gguf package: pip install numpy gguf")

ALLOWED = {GGMLQuantizationType.Q4_K, GGMLQuantizationType.Q5_K,
           GGMLQuantizationType.Q6_K, GGMLQuantizationType.Q8_0}
TARGET = GGMLQuantizationType.Q8_0
EXPERT_RE = re.compile(r"^blk\.\d+\.ffn_(?:gate|up|down)_exps\.weight$")
COPY_CHUNK = 64 << 20          # bytes per read when copying a tensor as is
ROWS_PER_STEP = 2048           # rows per dequantize/quantize step

# GGUF value types -> fixed widths (8 = string, 9 = array).
WIDTHS = {0: 1, 1: 1, 2: 2, 3: 2, 4: 4, 5: 4, 6: 4, 7: 1, 10: 8, 11: 8, 12: 8}


def read(f, fmt):
    return struct.unpack("<" + fmt, f.read(struct.calcsize("<" + fmt)))[0]


def read_str(f):
    return f.read(read(f, "Q")).decode("utf-8", "replace")


def skip_value(f, vtype):
    if vtype in WIDTHS:
        f.seek(WIDTHS[vtype], 1)
    elif vtype == 8:
        f.seek(read(f, "Q"), 1)
    elif vtype == 9:
        etype, count = read(f, "I"), read(f, "Q")
        if etype in WIDTHS:
            f.seek(WIDTHS[etype] * count, 1)
        else:
            for _ in range(count):
                skip_value(f, etype)
    else:
        raise ValueError(f"unknown GGUF value type {vtype}")


def nbytes(ttype, dims):
    block, size = GGML_QUANT_SIZES[ttype]
    n = int(np.prod(dims))
    if n % block:
        raise ValueError(f"{n} elements is not a whole number of {ttype.name} blocks")
    return n // block * size


def align(n, a):
    return (n + a - 1) // a * a


def parse(path):
    """The header bytes up to the tensor directory, the alignment, the tensor
    directory, and where the tensor data starts."""
    with open(path, "rb") as f:
        if f.read(4) != b"GGUF":
            sys.exit(f"{path} is not a GGUF file")
        version = read(f, "I")
        if version < 2:
            sys.exit(f"GGUF version {version} is too old")
        n_tensors, n_kv = read(f, "Q"), read(f, "Q")
        alignment = 32
        for _ in range(n_kv):
            key = read_str(f)
            vtype = read(f, "I")
            if key == "general.alignment" and vtype == 4:
                alignment = read(f, "I")
            else:
                skip_value(f, vtype)
        kv_end = f.tell()
        tensors = []
        for _ in range(n_tensors):
            name = read_str(f)
            dims = [read(f, "Q") for _ in range(read(f, "I"))]
            ttype = GGMLQuantizationType(read(f, "I"))
            offset = read(f, "Q")
            tensors.append({"name": name, "dims": dims, "type": ttype, "offset": offset})
        data_start = align(f.tell(), alignment)
        f.seek(0)
        head = f.read(kv_end)
    return head, alignment, tensors, data_start


def tensor_info(t, ttype, offset):
    name = t["name"].encode()
    return (struct.pack("<Q", len(name)) + name + struct.pack("<I", len(t["dims"]))
            + b"".join(struct.pack("<Q", d) for d in t["dims"])
            + struct.pack("<IQ", int(ttype), offset))


def convert_rows(src, t, dst, progress):
    """Dequantize tensor `t` (at src's current position) and write it as TARGET."""
    ne0 = t["dims"][0]
    rows = int(np.prod(t["dims"][1:]))
    row_in = nbytes(t["type"], [ne0])
    done = 0
    while done < rows:
        n = min(ROWS_PER_STEP, rows - done)
        raw = np.frombuffer(src.read(row_in * n), dtype=np.uint8).reshape(n, row_in)
        values = dequantize(raw, t["type"]).astype(np.float32, copy=False)
        if not np.isfinite(values).all():
            sys.exit(f"{t['name']} holds non-finite values; refusing to convert it")
        dst.write(quantize(values, TARGET).tobytes())
        done += n
        progress(n * row_in)


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("src", help="the GGUF to fix")
    ap.add_argument("dst", nargs="?", help="where to write it (default: '-Strata' before the size)")
    ap.add_argument("--verify", action="store_true",
                    help="read the result back: every copied tensor byte-identical, converted ones close")
    ap.add_argument("--force", action="store_true", help="overwrite dst if it exists")
    args = ap.parse_args()

    head, alignment, tensors, data_start = parse(args.src)
    bad = [t for t in tensors if EXPERT_RE.match(t["name"]) and t["type"] not in ALLOWED]
    if not bad:
        print("every expert tensor is already a type Strata takes; nothing to do")
        return 0
    unsupported = []
    for t in bad:
        try:
            dequantize(np.zeros((1, nbytes(t["type"], [t["dims"][0]])), dtype=np.uint8), t["type"])
        except NotImplementedError:
            unsupported.append(t)
    if unsupported:
        sys.exit("gguf-py can't dequantize " + ", ".join(sorted({t['type'].name for t in unsupported})))

    dst = args.dst
    if not dst:
        m = re.search(r"((?:UD-)?I?Q\d+(?:_[A-Za-z0-9]+)*)\.gguf$", os.path.basename(args.src), re.I)
        if not m:
            sys.exit("can't find the size at the end of the file name; pass dst")
        base = args.src[:-len(m.group(0))].rstrip("-_.")
        dst = f"{base}-Strata-{m.group(1)}.gguf"
    if os.path.exists(dst) and not args.force:
        sys.exit(f"{dst} exists; pass --force to overwrite it")

    for t in bad:
        print(f"  {t['name']}: {t['type'].name} -> {TARGET.name}")

    # New directory: same order, converted tensors resized, offsets re-laid.
    convert = {t["name"] for t in bad}
    layout, offset = [], 0
    for t in tensors:
        ttype = TARGET if t["name"] in convert else t["type"]
        offset = align(offset, alignment)
        layout.append((t, ttype, offset))
        offset += nbytes(ttype, t["dims"])
    total = sum(nbytes(t["type"], t["dims"]) for t in tensors)

    written = [0]

    def progress(n):
        written[0] += n
        print(f"\r  {written[0] / total:6.1%}", end="", flush=True)

    tmp = dst + ".part"
    with open(args.src, "rb") as src, open(tmp, "wb") as out:
        out.write(head)
        for t, ttype, off in layout:
            out.write(tensor_info(t, ttype, off))
        out.write(b"\0" * (align(out.tell(), alignment) - out.tell()))
        out_data = out.tell()
        for t, ttype, off in layout:
            out.write(b"\0" * (out_data + off - out.tell()))
            src.seek(data_start + t["offset"])
            if t["name"] in convert:
                convert_rows(src, t, out, progress)
            else:
                left = nbytes(t["type"], t["dims"])
                while left:
                    chunk = src.read(min(COPY_CHUNK, left))
                    if not chunk:
                        sys.exit(f"{args.src} ends inside {t['name']}: the file is incomplete")
                    out.write(chunk)
                    left -= len(chunk)
                    progress(len(chunk))
    os.replace(tmp, dst)
    print(f"\r  wrote {dst} ({os.path.getsize(dst):,} bytes, was {os.path.getsize(args.src):,})")

    if args.verify:
        return verify(args.src, dst, convert)
    return 0


def verify(src_path, dst_path, convert):
    """Copied tensors byte-identical; converted ones within Q8_0's rounding."""
    print("  verifying ...")
    _, _, src_t, src_start = parse(src_path)
    _, _, dst_t, dst_start = parse(dst_path)
    if [t["name"] for t in src_t] != [t["name"] for t in dst_t]:
        sys.exit("verify: the tensor lists differ")
    worst = 0.0
    with open(src_path, "rb") as a, open(dst_path, "rb") as b:
        for s, d in zip(src_t, dst_t):
            a.seek(src_start + s["offset"])
            b.seek(dst_start + d["offset"])
            if s["name"] not in convert:
                left = nbytes(s["type"], s["dims"])
                while left:
                    n = min(COPY_CHUNK, left)
                    if a.read(n) != b.read(n):
                        sys.exit(f"verify: {s['name']} changed")
                    left -= n
                continue
            if d["type"] != TARGET:
                sys.exit(f"verify: {d['name']} is {d['type'].name}, not {TARGET.name}")
            ne0, rows = s["dims"][0], int(np.prod(s["dims"][1:]))
            rin, rout = nbytes(s["type"], [ne0]), nbytes(TARGET, [ne0])
            for done in range(0, rows, ROWS_PER_STEP):
                n = min(ROWS_PER_STEP, rows - done)
                x = dequantize(np.frombuffer(a.read(rin * n), np.uint8).reshape(n, rin), s["type"])
                y = dequantize(np.frombuffer(b.read(rout * n), np.uint8).reshape(n, rout), TARGET)
                scale = np.abs(x).max() or 1.0
                worst = max(worst, float(np.abs(x - y).max() / scale))
    print(f"  ok: copied tensors identical; converted ones off by at most {worst:.2%} of their largest value")
    if worst > 0.01:
        sys.exit("verify: the converted tensors drifted more than Q8_0's rounding should allow")
    return 0


if __name__ == "__main__":
    sys.exit(main())
