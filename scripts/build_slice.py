"""Build a tractable slice of the Community Notes public data.

The public dump lives at
https://ton.twimg.com/birdwatch-public-data/YYYY/MM/DD/<table>/<table>-NNNNN.zip
with tables notes, noteRatings (files named ratings-NNNNN.zip),
noteStatusHistory and userEnrollment. Only the last few days are retained.

Notes and status history are small and get downloaded in full. Ratings are
tens of GB uncompressed, so each part is downloaded, streamed through a zip
decoder, filtered down to notes created inside the candidate window, and
deleted before the next part is touched. At most two raw parts (about 3 GB)
are on disk at any moment. The final window is then cut by note creation day
so the ratings land inside the requested row range.

Run: .venv/bin/python scripts/build_slice.py --date 2026-09-02
"""

import argparse
import datetime as dt
import glob
import io
import json
import logging
import os
import shutil
import struct
import sys
import threading
import time
import zlib

import numpy as np
import pandas as pd
import requests

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
RAW = os.path.join(DATA, "raw")
BASE = "https://ton.twimg.com/birdwatch-public-data"
GB = 1 << 30

log = logging.getLogger("build_slice")

HELPFULNESS = {b"NOT_HELPFUL": 0.0, b"SOMEWHAT_HELPFUL": 0.5, b"HELPFUL": 1.0}

# suggestion (free text) is dropped, it is large and unused.
RATING_KEEP = [
    "noteId", "raterParticipantId", "createdAtMillis", "version",
    "helpfulnessLevel", "helpful", "notHelpful", "ratedOnTweetId",
    "ratingSourceBucketed",
]
NOTE_KEEP = [
    "noteId", "noteAuthorParticipantId", "createdAtMillis", "tweetId",
    "classification", "summary", "isMediaNote", "isCollaborativeNote",
]


def free_gb(path=DATA):
    return shutil.disk_usage(path).free / GB


def log_disk(label):
    log.info("disk free at %s: %.1f GB", label, free_gb())


def check_disk(need_bytes, floor_gb):
    free = free_gb()
    if free - need_bytes / GB < floor_gb:
        raise RuntimeError(
            f"refusing to write {need_bytes / GB:.1f} GB with {free:.1f} GB free "
            f"(floor {floor_gb} GB)")


def url(date, table, fname):
    return f"{BASE}/{date.strftime('%Y/%m/%d')}/{table}/{fname}"


def head_size(u, timeout=180):
    r = requests.head(u, timeout=timeout)
    if r.status_code == 404:
        return None
    r.raise_for_status()
    return int(r.headers.get("content-length", 0))


def find_date(start, max_back=14):
    """Walk back from start until notes-00000.zip exists."""
    for i in range(max_back):
        d = start - dt.timedelta(days=i)
        if head_size(url(d, "notes", "notes-00000.zip")) is not None:
            return d
    raise RuntimeError(f"no dump found in the {max_back} days before {start}")


def list_parts(date, table, prefix):
    """Enumerate <prefix>-NNNNN.zip until the first 404. Returns [(name, bytes)]."""
    parts = []
    i = 0
    while True:
        name = f"{prefix}-{i:05d}.zip"
        size = head_size(url(date, table, name))
        if size is None:
            break
        parts.append((name, size))
        i += 1
    return parts


def download(u, dest, size, floor_gb):
    if os.path.exists(dest) and os.path.getsize(dest) == size:
        log.info("already have %s", os.path.basename(dest))
        return
    check_disk(size, floor_gb)
    tmp = dest + ".part"
    t0 = time.time()
    with requests.get(u, stream=True, timeout=300) as r:
        r.raise_for_status()
        with open(tmp, "wb") as f:
            for chunk in r.iter_content(chunk_size=4 << 20):
                f.write(chunk)
    os.replace(tmp, dest)
    got = os.path.getsize(dest)
    log.info("downloaded %s: %.2f GB in %.0fs (%.1f MB/s)",
             os.path.basename(dest), got / GB, time.time() - t0,
             got / max(time.time() - t0, 1e-9) / 1e6)
    if size and got != size:
        raise RuntimeError(f"{dest}: expected {size} bytes, got {got}")


def zip_member_stream(path, chunk=4 << 20):
    """Yield decompressed bytes of the first member of a zip without seeking.

    The dump zips hold exactly one deflated TSV. Reading the local header and
    inflating as we go means the uncompressed TSV never touches the disk.
    """
    with open(path, "rb") as f:
        hdr = f.read(30)
        sig, _ver, _flags, method, _t, _d, _crc, csize, usize, fnlen, exlen = \
            struct.unpack("<IHHHHHIIIHH", hdr)
        if sig != 0x04034B50:
            raise RuntimeError(f"{path}: not a zip local header")
        f.read(fnlen + exlen)
        if method == 8:
            d = zlib.decompressobj(-15)
            while not d.eof:
                c = f.read(chunk)
                if not c:
                    break
                out = d.decompress(c)
                if out:
                    yield out
        elif method == 0:
            if usize == 0xFFFFFFFF:
                raise RuntimeError(f"{path}: stored zip64 member is not supported")
            left = usize
            while left > 0:
                c = f.read(min(chunk, left))
                if not c:
                    break
                left -= len(c)
                yield c
        else:
            raise RuntimeError(f"{path}: unsupported zip method {method}")


def filter_ratings_zip(path, note_ids):
    """Stream one ratings part and keep rows whose noteId is in note_ids.

    Returns (header_line, kept_lines, total_lines, total_bytes).
    """
    keep = []
    header = None
    tail = b""
    n = 0
    nbytes = 0
    for out in zip_member_stream(path):
        nbytes += len(out)
        buf = tail + out
        lines = buf.split(b"\n")
        tail = lines.pop()
        if header is None and lines:
            header = lines.pop(0)
        n += len(lines)
        for line in lines:
            if line[:line.find(b"\t")] in note_ids:
                keep.append(line)
    if tail:
        n += 1
        if tail[:tail.find(b"\t")] in note_ids:
            keep.append(tail)
    return header, keep, n, nbytes


def ratings_frame(header, lines):
    """Parse kept TSV lines into a compact DataFrame."""
    cols = header.decode().split("\t")
    tag_cols = [c for c in cols if c.startswith(("helpful", "notHelpful"))
                and c not in ("helpful", "notHelpful", "helpfulnessLevel")]
    usecols = [c for c in RATING_KEEP + tag_cols if c in cols]
    dtype = {c: "Int8" for c in tag_cols}
    dtype.update({
        "noteId": "int64", "raterParticipantId": "string",
        "createdAtMillis": "int64", "version": "Int8",
        "helpfulnessLevel": "string", "helpful": "Int8", "notHelpful": "Int8",
        "ratedOnTweetId": "Int64", "ratingSourceBucketed": "string",
    })
    dtype = {k: v for k, v in dtype.items() if k in usecols}
    if not lines:
        return pd.DataFrame({c: pd.Series(dtype=dtype.get(c, "object")) for c in usecols})
    buf = io.BytesIO(b"\n".join([header] + lines))
    df = pd.read_csv(buf, sep="\t", usecols=usecols, dtype=dtype,
                     keep_default_na=False, na_values=[""], quoting=3)
    return df[usecols]


def add_helpfulness(df):
    """Map helpfulnessLevel to the float the official scorer uses.

    Old v1 ratings only carry the binary helpful/notHelpful flags, so those
    are backfilled the same way the official preprocessing does.
    """
    lvl = df["helpfulnessLevel"].astype("string")
    if "helpful" in df and "notHelpful" in df:
        lvl = lvl.mask(lvl.isna() & (df["helpful"] == 1), "HELPFUL")
        lvl = lvl.mask(lvl.isna() & (df["notHelpful"] == 1), "NOT_HELPFUL")
    m = {k.decode(): v for k, v in HELPFULNESS.items()}
    df["helpfulnessLevel"] = lvl
    df["helpfulness"] = lvl.map(m).astype("float32")
    return df


def load_notes(date, parts, floor_gb, keep_raw):
    frames = []
    for name, size in parts:
        dest = os.path.join(RAW, name)
        download(url(date, "notes", name), dest, size, floor_gb)
        df = pd.read_csv(dest, sep="\t", dtype={
            "noteId": "int64", "noteAuthorParticipantId": "string",
            "createdAtMillis": "int64", "tweetId": "int64",
            "classification": "string", "summary": "string",
            "isMediaNote": "Int8", "isCollaborativeNote": "Int8",
            "believable": "string", "harmful": "string",
            "validationDifficulty": "string",
        }, keep_default_na=False, na_values=[""], quoting=3)
        log.info("%s: %d rows, columns: %s", name, len(df), list(df.columns))
        frames.append(df)
        if not keep_raw:
            os.remove(dest)
    notes = pd.concat(frames, ignore_index=True)
    return notes


def load_status(date, floor_gb, keep_raw):
    name = "noteStatusHistory-00000.zip"
    dest = os.path.join(RAW, name)
    size = head_size(url(date, "noteStatusHistory", name))
    download(url(date, "noteStatusHistory", name), dest, size, floor_gb)
    df = pd.read_csv(dest, sep="\t", dtype="string", keep_default_na=False,
                     na_values=[""], quoting=3)
    for c in df.columns:
        if c == "noteId" or c.startswith("timestamp") or c == "createdAtMillis":
            df[c] = pd.to_numeric(df[c], errors="coerce").astype("Int64")
    log.info("noteStatusHistory: %d rows, columns: %s", len(df), list(df.columns))
    if not keep_raw:
        os.remove(dest)
    return df


def build_ratings(date, parts, note_ids, floor_gb, keep_raw, prefetch=True):
    """Download, filter and delete each ratings part. Returns a DataFrame."""
    ids = {str(i).encode() for i in note_ids}
    frames = []
    total_rows = 0
    total_bytes = 0
    header = None

    dests = [os.path.join(RAW, name) for name, _ in parts]
    threads = {}

    def fetch(i):
        name, size = parts[i]
        download(url(date, "noteRatings", name), dests[i], size, floor_gb)

    def start_fetch(i):
        if i < len(parts) and i not in threads:
            t = threading.Thread(target=fetch, args=(i,), daemon=True)
            t.start()
            threads[i] = t

    start_fetch(0)
    for i, (name, size) in enumerate(parts):
        threads[i].join()
        if not os.path.exists(dests[i]):
            raise RuntimeError(f"download of {name} failed")
        if prefetch:
            start_fetch(i + 1)
        log_disk(f"before filtering {name}")
        t0 = time.time()
        hdr, kept, n, nbytes = filter_ratings_zip(dests[i], ids)
        header = header or hdr
        total_rows += n
        total_bytes += nbytes
        log.info("%s: %d rows, %.2f GB uncompressed, kept %d (%.0fs)",
                 name, n, nbytes / GB, len(kept), time.time() - t0)
        frames.append(ratings_frame(hdr, kept))
        del kept
        if not keep_raw:
            os.remove(dests[i])
        log_disk(f"after {name}")
    log.info("ratings total: %d rows, %.1f GB uncompressed across %d parts",
             total_rows, total_bytes / GB, len(parts))
    df = pd.concat(frames, ignore_index=True)
    return add_helpfulness(df), total_rows, total_bytes


def choose_window(notes, ratings, start, end, min_rows, max_rows, target_rows):
    """Cut the candidate window by note creation day so ratings hit the target.

    Days are added from `start` while the cumulative rating count stays at or
    under max_rows. The cut closest to target_rows wins.
    """
    per_note = ratings.groupby("noteId").size()
    day = pd.to_datetime(notes["createdAtMillis"], unit="ms").dt.floor("D")
    per_day = per_note.reindex(notes["noteId"]).fillna(0).groupby(day.values).sum()
    per_day = per_day.sort_index()
    cum = per_day.cumsum()
    ok = cum[cum <= max_rows]
    if ok.empty:
        cut = per_day.index[0]
        rows = cum.iloc[0]
        log.warning("first day alone has %d ratings, over max_rows %d", rows, max_rows)
    else:
        best = (ok - target_rows).abs().idxmin()
        cut = best
        rows = cum[best]
    cut_end = (cut + pd.Timedelta(days=1)).to_pydatetime()
    if rows < min_rows:
        log.warning("window %s..%s only has %d ratings, under min_rows %d",
                    start, cut_end, rows, min_rows)
    return cut_end, int(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--date", default=None,
                    help="dump date YYYY-MM-DD (default: newest available)")
    ap.add_argument("--window-start", default="2026-06-01",
                    help="notes created on or after this UTC date are candidates")
    ap.add_argument("--window-end", default="2026-07-01",
                    help="candidate window end (exclusive)")
    ap.add_argument("--target-rows", type=int, default=2_000_000)
    ap.add_argument("--min-rows", type=int, default=1_000_000)
    ap.add_argument("--max-rows", type=int, default=3_000_000)
    ap.add_argument("--disk-floor-gb", type=float, default=8.0,
                    help="never let free space drop under this")
    ap.add_argument("--keep-raw", action="store_true",
                    help="keep raw zips in data/raw instead of deleting them")
    ap.add_argument("--no-prefetch", action="store_true",
                    help="do not download the next ratings part while filtering")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, stream=sys.stdout,
                        format="%(asctime)s %(levelname)s %(message)s")
    os.makedirs(RAW, exist_ok=True)
    log_disk("start")

    if args.date:
        date = find_date(dt.date.fromisoformat(args.date))
    else:
        date = find_date(dt.date.today())
    log.info("using dump date %s", date)

    start = dt.datetime.fromisoformat(args.window_start)
    end = dt.datetime.fromisoformat(args.window_end)
    start_ms = int(start.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    end_ms = int(end.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)

    note_parts = list_parts(date, "notes", "notes")
    log.info("notes parts: %s", note_parts)
    notes_all = load_notes(date, note_parts, args.disk_floor_gb, args.keep_raw)
    log.info("notes total: %d", len(notes_all))
    cand = notes_all[(notes_all.createdAtMillis >= start_ms)
                     & (notes_all.createdAtMillis < end_ms)].copy()
    log.info("candidate notes in [%s, %s): %d", start, end, len(cand))

    status_all = load_status(date, args.disk_floor_gb, args.keep_raw)
    log_disk("after notes and status")

    rating_parts = list_parts(date, "noteRatings", "ratings")
    log.info("ratings parts: %d, %.1f GB zipped", len(rating_parts),
             sum(s for _, s in rating_parts) / GB)
    ratings, total_rows, total_bytes = build_ratings(
        date, rating_parts, cand.noteId.values, args.disk_floor_gb,
        args.keep_raw, prefetch=not args.no_prefetch)
    log.info("candidate ratings: %d rows on %d notes", len(ratings),
             ratings.noteId.nunique())
    cand_path = os.path.join(RAW, f"ratings_{args.window_start}_{args.window_end}.parquet")
    ratings.to_parquet(cand_path, index=False)
    log.info("wrote candidate ratings to %s", cand_path)

    cut_end, rows = choose_window(cand, ratings, start, end, args.min_rows,
                                  args.max_rows, args.target_rows)
    cut_ms = int(cut_end.replace(tzinfo=dt.timezone.utc).timestamp() * 1000)
    notes = cand[cand.createdAtMillis < cut_ms][NOTE_KEEP].reset_index(drop=True)
    ratings = ratings[ratings.noteId.isin(notes.noteId)].reset_index(drop=True)
    status = status_all[status_all.noteId.isin(notes.noteId)].reset_index(drop=True)
    log.info("final window [%s, %s): %d notes, %d ratings, %d raters, %d status rows",
             start, cut_end, len(notes), len(ratings),
             ratings.raterParticipantId.nunique(), len(status))

    ratings.to_parquet(os.path.join(DATA, "ratings.parquet"), index=False)
    notes.to_parquet(os.path.join(DATA, "notes.parquet"), index=False)
    status.to_parquet(os.path.join(DATA, "status.parquet"), index=False)
    log.info("status columns: %s", list(status.columns))
    log.info("ratings columns: %s", list(ratings.columns))

    meta = {
        "dump_date": date.isoformat(),
        "candidate_window": [args.window_start, args.window_end],
        "window": [start.date().isoformat(), cut_end.date().isoformat()],
        "notes_total_in_dump": int(len(notes_all)),
        "status_total_in_dump": int(len(status_all)),
        "ratings_total_in_dump": int(total_rows),
        "ratings_uncompressed_gb": round(total_bytes / GB, 2),
        "ratings_zipped_gb": round(sum(s for _, s in rating_parts) / GB, 2),
        "notes": int(len(notes)),
        "ratings": int(len(ratings)),
        "raters": int(ratings.raterParticipantId.nunique()),
        "status_rows": int(len(status)),
        "status_columns": list(status.columns),
        "ratings_columns": list(ratings.columns),
    }
    with open(os.path.join(DATA, "slice.json"), "w") as f:
        json.dump(meta, f, indent=2)
    log.info("wrote data/slice.json")
    log_disk("end")


if __name__ == "__main__":
    main()
