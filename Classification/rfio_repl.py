#!/usr/bin/env python3
import socket
import struct
from pathlib import Path

import numpy as np


HOST = "127.0.0.1"
PORT = 9090
DATA_PATH = Path(__file__).resolve().parent / "dataset" / "radioml_2016.10a.npz"

MODS = [
    "8PSK",
    "AM-DSB",
    "AM-SSB",
    "BPSK",
    "CPFSK",
    "GFSK",
    "PAM4",
    "QAM16",
    "QAM64",
    "QPSK",
    "WBFM",
]


def recv_exact(sock, nbytes):
    buf = bytearray(nbytes)
    view = memoryview(buf)
    while nbytes:
        n = sock.recv_into(view, nbytes)
        if n == 0:
            raise ConnectionError("socket closed")
        view = view[n:]
        nbytes -= n
    return bytes(buf)


def fft_log_power(x):
    win = np.hamming(128)
    iq = (x[:, 0, :].astype(np.float64) + 1j * x[:, 1, :].astype(np.float64)) * win
    spec = np.fft.fft(iq, axis=-1)
    power = (spec.real * spec.real + spec.imag * spec.imag).astype(np.float32)
    return np.log1p(power).astype(np.float32)


def make_sample(x, mean, std):
    x = x[None, :, :]
    freq = fft_log_power(x)
    if std > 0:
        freq = (freq - mean) / std
    sample = np.stack(
        [
            x[:, 0, :].astype(np.float32),
            x[:, 1, :].astype(np.float32),
            freq.astype(np.float32),
        ],
        axis=1,
    )
    return np.ascontiguousarray(sample, dtype=np.float32)


def send_one(sock, sample):
    sock.sendall(struct.pack("!I", 1))
    sock.sendall(sample.tobytes())

    # RFIO TX is word-wide. main.cpp sends one class byte, padded into a u32.
    return recv_exact(sock, 4)[0]


def main():
    data = np.load(DATA_PATH)
    X = data["X"]
    y = data["y"]
    snrs = data["snrs"]

    print("computing FFT normalization stats...")
    freq_all = fft_log_power(X)
    freq_mean = float(freq_all.mean())
    freq_std = float(freq_all.std())

    rng = np.random.default_rng()
    print(f"connecting to {HOST}:{PORT}")

    with socket.create_connection((HOST, PORT)) as sock:
        print("type a modulation name to send one random sample:")
        print("  " + " ".join(MODS))
        print("type q to quit")

        while True:
            name = input("mod> ").strip()
            if name.lower() in {"q", "quit", "exit"}:
                break
            if name not in MODS:
                print("unknown modulation")
                continue

            true_label = MODS.index(name)
            choices = np.flatnonzero(y == true_label)
            idx = int(rng.choice(choices))

            sample = make_sample(X[idx], freq_mean, freq_std)
            pred = send_one(sock, sample)

            print(
                f"sent idx={idx} true={name} snr={int(snrs[idx])} "
                f"pred={pred}:{MODS[pred] if pred < len(MODS) else 'unknown'}"
            )


if __name__ == "__main__":
    main()
