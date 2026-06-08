"""
MUSDB18-HQ Data Loader and Mono Downmixer

This module loads and processes the MUSDB18-HQ dataset 
(Rafii et al., 2019). It includes custom logic to downmix 
stereo tracks to mono audio to fit the Ascend NPU memory 
constraints for the TDANet architecture.

For the full dataset citation, please see the README.
"""

import os
from typing import List, Tuple

import soundfile as sf
import librosa
import numpy as np
import torch
from torch.utils.data import Dataset


class MusdbDataset(Dataset):
    DEFAULT_STEMS = ("vocals", "drums", "bass", "other")

    def __init__(self, musdb_root: str, subset: str = "train", sr: int = 16000, stems: Tuple[str, ...] = DEFAULT_STEMS, segment_length: float = None):
        """If `segment_length` is provided (seconds), training samples will be fixed-size segments.
        For evaluation use `segment_length=None` to return full tracks (the evaluator will chunk)."""
        self.musdb_root = musdb_root
        self.subset = subset
        self.sr = sr
        self.stems = list(stems)
        self.segment_length = segment_length
        self.segment_samples = None if segment_length is None else int(round(segment_length * sr))

        self.tracks = self._find_tracks()
        if len(self.tracks) == 0:
            raise RuntimeError(f"No MUSDB tracks found under {musdb_root} (subset={subset})")

    def _find_tracks(self) -> List[str]:
        base = os.path.join(self.musdb_root, self.subset)
        if not os.path.isdir(base):
            base = self.musdb_root
        tracks = []
        for entry in os.listdir(base):
            p = os.path.join(base, entry)
            if not os.path.isdir(p):
                continue
            ok = True
            for s in self.stems:
                stem_path = os.path.join(p, f"{s}.wav")
                if not os.path.exists(stem_path):
                    ok = False
                    break
            if ok:
                tracks.append(p)
        return sorted(tracks)

    def __len__(self) -> int:
        return len(self.tracks)

    def _load_audio(self, path: str) -> np.ndarray:
        wav, sr0 = sf.read(path)
        if wav.ndim == 2:
            wav = wav.mean(axis=1)
        if sr0 != self.sr:
            wav = librosa.resample(wav.astype(np.float32), orig_sr=sr0, target_sr=self.sr)
        return wav.astype(np.float32)

    def __getitem__(self, idx: int):
        track_dir = self.tracks[idx]
        stems = []
        for s in self.stems:
            p = os.path.join(track_dir, f"{s}.wav")
            wav = self._load_audio(p)
            stems.append(wav)

        min_len = min([w.shape[0] for w in stems])
        stems = [w[:min_len] for w in stems]

        sources = np.stack(stems, axis=0)
        mixture = sources.sum(axis=0)

        # If segment_samples is set, return a fixed-length segment (randomly cropped)
        if self.segment_samples is not None:
            seg = self.segment_samples
            total = mixture.shape[0]
            if total >= seg:
                start = np.random.randint(0, total - seg + 1)
                end = start + seg
                sources = sources[:, start:end]
                mixture = mixture[start:end]
            else:
                # pad to required length
                pad_len = seg - total
                sources = np.pad(sources, ((0, 0), (0, pad_len)), mode='constant')
                mixture = np.pad(mixture, (0, pad_len), mode='constant')

        return torch.from_numpy(mixture).float(), torch.from_numpy(sources).float()
