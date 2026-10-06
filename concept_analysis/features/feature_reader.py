"""Explicit access to existing consolidated feature caches."""

import math
from pathlib import Path
import tomllib
from zipfile import BadZipFile

import numpy as np


class NpzFeatureReader:
    """Read layer NPZ files from a cache directory with a manifest.

    Reads require a layer filename stem. IDs are exact archive keys.
    No inference or layout fallback is performed.
    """


    def __init__(self, cache_dir):
        self.cache_dir = Path(cache_dir)
        self._manifest_path = self.cache_dir / "manifest.toml"
        try:
            with self._manifest_path.open("rb") as stream:
                self._manifest = tomllib.load(stream)
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing cache manifest: {self._manifest_path}") from exc
        except (OSError, tomllib.TOMLDecodeError) as exc:
            raise ValueError(f"Cannot read cache manifest {self._manifest_path}: {exc}") from exc


    def feature_frame_rate_for_layer(
        self, layer: int | str, legacy: bool = False,
    ) -> float | dict[str, int | float]:
        """Return the layer rate, or legacy sampling arguments when legacy=True.

        Each field falls back to the model-wide value. Keep the legacy pair
        intact so conversions preserve multiply-then-divide rounding.
        """
        layers = self._manifest.get("layers", {})
        layer_settings = layers.get(str(layer), {})
        if legacy:
            feature_frame_rate_legacy_params = {}
            for name, key in (("sr", "audio_sampling_rate"),
                              ("subsampling_factor", "feature_subsampling_factor")):
                value = layer_settings.get(key, self._manifest.get(key))
                if (isinstance(value, bool)
                        or not isinstance(value, (int, float))
                        or not math.isfinite(value) or value <= 0):
                    raise ValueError(f"Missing or invalid {key} for layer {layer} in {self._manifest_path}")
                feature_frame_rate_legacy_params[name] = value
            return feature_frame_rate_legacy_params

        feature_frame_rate = layer_settings.get(
            "feature_frame_rate", self._manifest.get("feature_frame_rate")
        )

        if (isinstance(feature_frame_rate, bool)
                or not isinstance(feature_frame_rate, (int, float))
                or not math.isfinite(feature_frame_rate)
                or feature_frame_rate <= 0):
            raise ValueError(f"Missing or invalid feature_frame_rate for layer {layer} in {self._manifest_path}")
        return float(feature_frame_rate)


    def read(self, ids: str | list[str], layer: int | str) -> list[np.ndarray]:
        """Return independent arrays in requested order, including duplicates.

        Stored shapes and dtypes are preserved. A string selects one ID.
        Empty selections still check that the archive can be opened.
        Supply the layer filename stem for the requested cache file.
        """
        if layer is None:
            raise ValueError("A layer is required when reading a cache_dir")
        if ids is None:
            raise TypeError("Utterance IDs must be strings")
        path = self.cache_dir / f"{layer}.npz"
        return self._read_consolidated_layer(path, ids)


    def read_layer(self, ids: str | list[str], layer: int | str) -> list[np.ndarray]:
        """Read a layer; equivalent to read(ids, layer)."""
        return self.read(ids, layer)


    def list_ids(self, layer: int | str) -> list[str]:
        """Return every utterance key in archive order without loading arrays."""
        if layer is None:
            raise ValueError("A layer is required when reading a cache_dir")
        path = self.cache_dir / f"{layer}.npz"
        return self._read_consolidated_layer(path, None)


    def _read_consolidated_layer(self, path, ids):
        if ids is not None:
            ids = [ids] if isinstance(ids, str) else list(ids)
            if any(not isinstance(utterance_id, str) for utterance_id in ids):
                raise TypeError("Utterance IDs must be strings")
        try:
            with path.open("rb") as stream:
                archive = np.load(stream, allow_pickle=False)
                if not isinstance(archive, np.lib.npyio.NpzFile):
                    raise ValueError("Expected an NPZ archive")
                with archive:
                    if ids is None:
                        return list(archive.files)
                    missing = [utterance_id for utterance_id in ids
                               if utterance_id not in archive.files]
                    if missing:
                        raise KeyError(f"Missing utterance IDs in {path}: {missing!r}")
                    return [np.array(archive[utterance_id]) for utterance_id in ids]
        except FileNotFoundError as exc:
            raise FileNotFoundError(f"Missing consolidated layer cache: {path}") from exc
        except (ValueError, EOFError, BadZipFile, OSError) as exc:
            raise ValueError(f"Cannot read consolidated layer cache {path}: {exc}") from exc
