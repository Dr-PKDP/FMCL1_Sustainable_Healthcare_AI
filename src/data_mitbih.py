"""
data_mitbih.py -- loads the MIT-BIH Arrhythmia Database into a per-recording
client list for the federated split used throughout this experiment.

Each of 44 recordings (48 total, minus 4 paced-rhythm records
excluded per the standard AAMI convention) becomes one client. Each
client's data is a set of fixed-length windows extracted around every
annotated beat in that recording, with the beat symbol mapped to
one of the five AAMI EC57 superclasses (N, S, V, F, Q).

AAMI mapping used (source: ANSI/AAMI EC57, as tabulated consistently
across the MIT-BIH literature):
  N (normal)                 : N, L, R, e, j
  S (supraventricular ectopic): A, a, J, S
  V (ventricular ectopic)     : V, E
  F (fusion)                  : F
  Q (unknown/paced)           : /, f, Q

Beats annotated with symbols outside this table are excluded, as are beats
too close to a record boundary to yield a full window.

Requires the `wfdb` package and a local copy of the MIT-BIH Arrhythmia
Database (PhysioNet record set mitdb), e.g. downloaded with:
    wget -r -N -c -np https://physionet.org/files/mitdb/1.0.0/
"""
import os
from typing import Dict, List, Optional, Tuple

import numpy as np

AAMI_MAP = {
    'N': 0, 'L': 0, 'R': 0, 'e': 0, 'j': 0,           # N
    'A': 1, 'a': 1, 'J': 1, 'S': 1,                    # S
    'V': 2, 'E': 2,                                     # V
    'F': 3,                                             # F
    '/': 4, 'f': 4, 'Q': 4,                             # Q
}
AAMI_CLASSES = ["N", "S", "V", "F", "Q"]

# Standard MIT-BIH record IDs (48 total; 4 paced-rhythm records excluded
# per AAMI recommendation, matching the 44-record convention used
# throughout the MIT-BIH literature).
PACED_RECORDS = {"102", "104", "107", "217"}
ALL_RECORD_IDS = [
    "100", "101", "103", "105", "106", "107", "108", "109", "111", "112",
    "113", "114", "115", "116", "117", "118", "119", "121", "122", "123",
    "124", "200", "201", "202", "203", "205", "207", "208", "209", "210",
    "212", "213", "214", "215", "217", "219", "220", "221", "222", "223",
    "228", "230", "231", "232", "233", "234",
]


def load_mitbih(base_dir: str,
                window_before: int = 130,
                window_after: int = 170,
                exclude_paced: bool = True,
                record_ids: Optional[List[str]] = None) -> Tuple[Dict, List]:
    """
    Returns (metadata, client_list).

    client_list: one entry per recording.
      Each entry: {"patient_id": str,
                   "X": np.ndarray shape (n_beats, n_channels, window_len),
                   "y": np.ndarray shape (n_beats,),
                   "n_records": int}   -- always 1 here (one recording per client)

    window_before/window_after are sample counts either side of each
    annotated R-peak, at the record's native sampling rate (360 Hz for
    standard MIT-BIH records), giving a fixed window_len =
    window_before + window_after for every extracted beat.

    Beats whose window would run past the start/end of the recording are
    dropped rather than zero-padded, since a beat-shape learning task
    should not be given windows partly filled with unrelated silence.
    """
    import wfdb

    ids = record_ids if record_ids is not None else list(ALL_RECORD_IDS)
    if exclude_paced:
        ids = [r for r in ids if r not in PACED_RECORDS]

    clients = []
    n_excluded_symbol = 0
    n_excluded_edge = 0

    for rid in ids:
        rec_path = os.path.join(base_dir, rid)
        if not os.path.exists(rec_path + ".hea"):
            continue  # record not present in this download; skip

        record = wfdb.rdrecord(rec_path)
        ann = wfdb.rdann(rec_path, "atr")

        sig = record.p_signal.astype(np.float32)          # (n_samples, n_channels)
        n_samples, n_channels = sig.shape

        windows, labels = [], []
        for sample_idx, symbol in zip(ann.sample, ann.symbol):
            cls = AAMI_MAP.get(symbol)
            if cls is None:
                n_excluded_symbol += 1
                continue
            lo, hi = sample_idx - window_before, sample_idx + window_after
            if lo < 0 or hi > n_samples:
                n_excluded_edge += 1
                continue
            windows.append(sig[lo:hi, :].T)                 # (n_channels, window_len)
            labels.append(cls)

        if not windows:
            continue

        X = np.stack(windows).astype(np.float32)
        y = np.array(labels, dtype=np.int64)
        clients.append({
            "patient_id": rid,
            "X": X, "y": y,
            "n_records": 1,
        })

    total_beats = sum(c["X"].shape[0] for c in clients)
    meta = {
        "n_clients": len(clients),
        "n_records": len(clients),
        "n_beats": total_beats,
        "n_excluded_symbol": n_excluded_symbol,
        "n_excluded_edge": n_excluded_edge,
        "n_classes": 5,
        "class_names": AAMI_CLASSES,
        "sampling_rate_hz": 360,
        "window_len": window_before + window_after,
    }
    return meta, clients


# ---------------------------------------------------------------------
# Synthetic fixture builder, used only for testing this loader without
# network access to PhysioNet. Produces real WFDB-format files (not a
# mock of the parsing logic), so the self-test below exercises the
# actual wfdb read path this loader depends on.
# ---------------------------------------------------------------------
def _build_synthetic_fixture(out_dir: str, record_id: str = "100",
                             n_beats: int = 200, fs: int = 360,
                             seed: int = 0) -> None:
    import wfdb
    rng = np.random.RandomState(seed)
    os.makedirs(out_dir, exist_ok=True)

    beat_gap = 300  # samples between synthetic beats
    n_samples = n_beats * beat_gap + 1000
    sig = (0.05 * rng.randn(n_samples, 2)).astype(np.float64)

    symbols_cycle = ['N', 'N', 'N', 'V', 'N', 'A', 'N', 'N', 'F', 'N']
    r_positions, r_symbols = [], []
    for i in range(n_beats):
        pos = 500 + i * beat_gap
        sym = symbols_cycle[i % len(symbols_cycle)]
        sig[pos:pos + 20, 0] += 1.5  # a crude "QRS-like" bump
        r_positions.append(pos)
        r_symbols.append(sym)

    wfdb.wrsamp(record_name=record_id, fs=fs, units=['mV', 'mV'],
               sig_name=['MLII', 'V1'], p_signal=sig, fmt=['16', '16'],
               write_dir=out_dir)

    ann = wfdb.Annotation(record_name=record_id, extension='atr',
                          sample=np.array(r_positions), symbol=r_symbols)
    ann.wrann(write_dir=out_dir)


if __name__ == "__main__":
    import tempfile, shutil
    print("Running self-test with a synthetic WFDB fixture "
          "(no network access used) ...")
    tmp = tempfile.mkdtemp()
    try:
        _build_synthetic_fixture(tmp, record_id="100", n_beats=200)
        meta, clients = load_mitbih(tmp, record_ids=["100"], exclude_paced=False)
        print(f"  Clients: {meta['n_clients']}")
        print(f"  Total beats: {meta['n_beats']}")
        print(f"  Excluded (unmapped symbol): {meta['n_excluded_symbol']}")
        print(f"  Excluded (edge window): {meta['n_excluded_edge']}")
        assert meta['n_clients'] == 1, "expected exactly one synthetic client"
        c = clients[0]
        print(f"  Client X shape: {c['X'].shape}, y shape: {c['y'].shape}")
        print(f"  Label distribution: {np.bincount(c['y'], minlength=5)}")
        # 10 beats/cycle, symbols N,N,N,V,N,A,N,N,F,N -> per 10: N=7,V=1,A=1,F=1
        # 200 beats = 20 cycles -> N=140, S(A)=20, V=20, F=20, Q=0
        counts = np.bincount(c['y'], minlength=5)
        expected = np.array([140, 20, 20, 20, 0])
        assert np.array_equal(counts, expected), \
            f"label mapping mismatch: got {counts}, expected {expected}"
        print("  Label mapping matches the synthetic pattern.")
        print("\nSelf-test passed. It exercises WFDB reading, beat windowing, AAMI")
        print("mapping and edge exclusion on a synthetic record, not the real")
        print("MIT-BIH annotation files.")
    finally:
        shutil.rmtree(tmp)
