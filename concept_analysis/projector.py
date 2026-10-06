"""Projection and rejection operators for concept subspaces."""

from __future__ import annotations

from abc import ABC
from pathlib import Path
import pickle
import numpy as np
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.preprocessing import OneHotEncoder
from concept_analysis.utils import get_mean_agg
from concept_analysis.subspaces.leace_np import LeaceEraser, LeaceFitterCCA


class BaseProjection(ABC):
    """Abstract base for all concept-space projectors.

    Subclasses expose operations on feature matrices X of shape (N, D):
        - transform(X)  → project X into the concept subspace (low-dim)
        - inverse_transform(Z) → reconstruct the projected features in D dimensions
        - project(X)    → project X onto the concept subspace (stays in D-dim)
        - reject(X)     → project X onto the orthogonal complement
    """
    name: str
    n_components: int | None
    components: np.ndarray | None
    mean: np.ndarray | float | None
    projection_subtract_mean: bool
    _transform_matrices: dict[int, np.ndarray]
    _projection_matrices: dict[int, np.ndarray]
    _inverse_transform_matrices: dict[int, np.ndarray]

    def transform(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        n = self.n_components if n_components is None else n_components
        if n not in self._transform_matrices:
            U, S, Vh = np.linalg.svd(self.components, full_matrices=False)
            self._transform_matrices[n] = U[:n, :n] @ (S[:n, None] * Vh[:n])
        return (X - self.mean) @ self._transform_matrices[n].T

    def inverse_transform(
        self,
        Z: np.ndarray,
        subtract_mean: bool | None = None,
    ) -> np.ndarray:
        """Reconstruct in feature space using the component count in ``Z``."""
        n = Z.shape[1]
        if n not in self._inverse_transform_matrices:
            U, S, Vh = np.linalg.svd(self.components, full_matrices=False)
            matrix = U[:n, :n] @ (S[:n, None] * Vh[:n])
            self._inverse_transform_matrices[n] = np.linalg.pinv(matrix.T)
        reconstructed = Z @ self._inverse_transform_matrices[n]
        if subtract_mean is None:
            subtract_mean = self.projection_subtract_mean
        if not subtract_mean:
            centered_mean = np.broadcast_to(self.mean, (self.components.shape[1],))
            projected_mean = self.project(centered_mean, n_components=n, subtract_mean=False)
            reconstructed += projected_mean
        return reconstructed

    def project(
        self,
        X: np.ndarray,
        n_components: int | None = None,
        subtract_mean: bool | None = None,
    ) -> np.ndarray:
        n = self.n_components if n_components is None else n_components
        if subtract_mean is None:
            subtract_mean = self.projection_subtract_mean
        if subtract_mean:
            X = X - self.mean

        if n not in self._projection_matrices:
            _, _, Vh = np.linalg.svd(self.components, full_matrices=False)
            basis = Vh[:n].T
            self._projection_matrices[n] = basis @ basis.T
        return X @ self._projection_matrices[n]

    def reject(
        self,
        X: np.ndarray,
        n_components: int | None = None,
        subtract_mean: bool | None = None,
    ) -> np.ndarray:
        return X - self.project(
            X,
            n_components=n_components,
            subtract_mean=subtract_mean,
        )

    def collapse(self, *args, **kwargs) -> np.ndarray:
        return self.reject(*args, **kwargs)



    def save(self, path: str | Path, extra_state: dict = {}) -> None:
        """Save named projector state, including any fitted sklearn model."""

        projector_type = type(self)
        state = {"name": self.name, "n_components": self.n_components,
            "projection_subtract_mean": self.projection_subtract_mean,
            "components": getattr(self, "components", None),
            "mean": getattr(self, "mean", None),
        }
        state.update(**extra_state)
        payload = {
            "format_version": 1,
            "projector_type": projector_type.__name__,
            "state": state,
        }

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as stream:
            pickle.dump(payload, stream)


    @classmethod
    def load(
        cls, path: str | Path, extra_state_keys: tuple[str, ...] = (),
    ) -> BaseProjection:
        """Load a versioned projector state file from a trusted source."""
        with Path(path).open("rb") as stream:
            payload = pickle.load(stream)
        if not isinstance(payload, dict) or payload.get("format_version") != 1:
            raise ValueError("Unsupported projector file format; refit legacy projectors")
        if payload.get("projector_type") != cls.__name__:
            raise ValueError(
                f"Saved projector type {payload.get('projector_type')!r} "
                f"does not match {cls.__name__}"
            )

        state = payload.get("state")
        if not isinstance(state, dict):
            raise ValueError("Projector file is missing its state")

        projector = cls()
        for key in (
            "name", "n_components", "projection_subtract_mean", "components", "mean",
            *extra_state_keys,
        ):
            if key in state:
                setattr(projector, key, state[key])
        projector._transform_matrices = {}
        projector._projection_matrices = {}
        projector._inverse_transform_matrices = {}
        return projector


class PCAProjection(BaseProjection):
    def __init__(
        self,
        *,
        n_components: int | None = None,
        whiten: bool = False,
        name: str = "PCA",
        projection_subtract_mean: bool = True,
    ) -> None:
        self.name = name
        self.n_components = n_components
        self.projection_subtract_mean = projection_subtract_mean
        self.whiten = whiten
        self._transform_matrices = {}
        self._projection_matrices = {}
        self._inverse_transform_matrices = {}

    def fit(self, X: np.ndarray) -> PCAProjection:
        """Fit ordinary PCA over the rows of ``X``."""
        self._transform_matrices = {}
        self._projection_matrices = {}
        self._inverse_transform_matrices = {}
        model = PCA(whiten=self.whiten).fit(X)
        self.model_ = model
        self.components = model.components_
        self.mean = model.mean_
        return self

    def transform(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        n = self.n_components if n_components is None else n_components
        if n not in self._transform_matrices:
            matrix = self.components[:n]
            if self.whiten:
                scale = np.sqrt(self.model_.explained_variance_)
                scale[scale < np.finfo(scale.dtype).eps] = np.finfo(scale.dtype).eps
                matrix = matrix / scale[:n, None]
            self._transform_matrices[n] = matrix
        matrix = self._transform_matrices[n]
        return np.asarray(X) @ matrix.T - self.mean @ matrix.T

    def inverse_transform(
        self,
        Z: np.ndarray,
        subtract_mean: bool | None = None,
    ) -> np.ndarray:
        """Reconstruct the same feature-space values returned by ``project``."""
        n = Z.shape[1]
        if n not in self._inverse_transform_matrices:
            matrix = self.components[:n]
            if self.whiten:
                matrix = matrix * np.sqrt(self.model_.explained_variance_[:n, None])
            self._inverse_transform_matrices[n] = matrix
        reconstructed = Z @ self._inverse_transform_matrices[n]
        if subtract_mean is None:
            subtract_mean = self.projection_subtract_mean
        if not subtract_mean:
            reconstructed += self.project(self.mean, n_components=n, subtract_mean=False)
        return reconstructed

    def project(
        self,
        X: np.ndarray,
        n_components: int | None = None,
        subtract_mean: bool | None = None,
    ) -> np.ndarray:
        n = self.n_components if n_components is None else n_components
        if subtract_mean is None:
            subtract_mean = self.projection_subtract_mean
        if subtract_mean:
            X = X - self.mean
        if n not in self._projection_matrices:
            basis = self.components[:n].T
            self._projection_matrices[n] = basis @ basis.T
        return X @ self._projection_matrices[n]

    def save(self, path: str | Path, extra_state: dict= {}) -> None:
        state = {
            "whiten": self.whiten,
            "model_": self.model_
        }
        state.update(**extra_state)
        super().save(path, extra_state=state)

    @classmethod
    def load(
        cls, path: str | Path, extra_state_keys: tuple[str, ...] = (),
    ) -> PCAProjection:
        return super().load(path, ("whiten", "model_", *extra_state_keys))


class CPCAProjection(PCAProjection):
    """PCA fitted over one centroid per target class."""

    def __init__(
        self,
        *,
        n_components: int | None = None,
        whiten: bool = False,
        name: str = "CPCA",
        projection_subtract_mean: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components,
            whiten=whiten,
            name=name,
            projection_subtract_mean=projection_subtract_mean,
        )

    def fit(self, X: np.ndarray, y: np.ndarray) -> CPCAProjection:
        """Fit PCA over the class centroids induced by ``y``."""
        self._transform_matrices = {}
        self._projection_matrices = {}
        self._inverse_transform_matrices = {}
        X_array = np.asarray(X)
        y_array = np.asarray(y)
        if y_array.ndim != 1:
            raise ValueError("y must be a one-dimensional array")
        if X_array.ndim < 1 or X_array.shape[0] != y_array.shape[0]:
            raise ValueError("X and y must contain the same number of samples")
        centroids, classes = get_mean_agg(X_array, y_array)
        if len(classes) < 2:
            raise ValueError("CPCA requires at least two target classes")
        super().fit(centroids)
        self.classes_ = classes
        return self

    def save(self, path: str | Path, extra_state: dict = {}):

        state = dict()
        if hasattr(self, "classes_"):
            state = {"classes_": self.classes_}

        state.update(**extra_state)
        super().save(path, extra_state=state)

    @classmethod
    def load(
        cls, path: str | Path, extra_state_keys: tuple[str, ...] = (),
    ) -> CPCAProjection:
        return super().load(path, ("classes_", *extra_state_keys))


class LEACEProjection(BaseProjection):
    """An oblique LEACE concept projector fitted from one target array."""

    def __init__(
        self,
        *,
        n_components: int | None = None,
        name: str = "LEACE",
        projection_subtract_mean: bool = True,
        shrinkage: bool =True,
    ) -> None:
        self.name = name
        self.n_components = n_components
        self.projection_subtract_mean = projection_subtract_mean
        self.shrinkage = shrinkage
        self._transform_matrices = {}
        self._projection_matrices = {}
        self._inverse_transform_matrices = {}


    def fit(self, X: np.ndarray, y: np.ndarray) -> LEACEProjection:
        if X.ndim != 2:
            raise ValueError("X must have shape (samples, features)")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with one label per sample")

        onehot_enc = OneHotEncoder(sparse_output=False)
        y_onehot = onehot_enc.fit_transform(y.reshape(-1, 1))

        leace_eraser = LeaceFitterCCA.fit(
            X, y_onehot,
            method=self._fit_method(), svd_tol=1e-7, shrinkage=self.shrinkage,
        ).eraser

        return self._set_fitted_eraser(leace_eraser, onehot_enc.categories_[0])


    def _set_fitted_eraser(self, eraser: LeaceEraser, classes: np.ndarray) -> LEACEProjection:
        self.model_ = eraser
        self.mean = eraser.bias if eraser.bias is not None else 0
        self.proj_right = eraser.proj_right.conj().T
        self.proj_left = eraser.proj_left.conj().T
        self.components = self.proj_right.T
        self.classes_ = classes
        self._transform_matrices = {}
        self._projection_matrices = {}
        self._inverse_transform_matrices = {}
        return self

    def _fit_method(self) -> str:
        return "leace"

    def transform(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        N = self.n_components if n_components is None else n_components
        return (np.asarray(X) - self.mean) @ self.proj_right[:, :N]

    def inverse_transform(
        self,
        Z: np.ndarray,
        subtract_mean: bool | None = None,
    ) -> np.ndarray:
        """Reconstruct with LEACE's left factor, preserving its oblique map."""
        N = Z.shape[-1]
        reconstructed = Z @ self.proj_left[:N]
        if subtract_mean is None:
            subtract_mean = self.projection_subtract_mean
        if not subtract_mean:
            reconstructed += self.mean @ self.proj_right[:, :N] @ self.proj_left[:N]
        return reconstructed

    def project(
        self,
        X: np.ndarray,
        n_components: int | None = None,
        subtract_mean: bool | None = None,
    ) -> np.ndarray:
        N = self.n_components if n_components is None else n_components
        if subtract_mean is None:
            subtract_mean = self.projection_subtract_mean

        delta = X - self.mean if subtract_mean else X
        if N not in self._projection_matrices:
            self._projection_matrices[N] = (
                self.proj_right[:, :N] @ self.proj_left[:N]
            )
        return delta @ self._projection_matrices[N]

    def save(self, path: str | Path, extra_state: dict = {}) -> None:
        state = {
            "shrinkage": self.shrinkage,
            "model_": self.model_,
            "proj_right": self.proj_right,
            "proj_left": self.proj_left,
            "classes_": self.classes_,
        }
        state.update(**extra_state)
        super().save(path, extra_state=state)

    @classmethod
    def load(
        cls, path: str | Path, extra_state_keys: tuple[str, ...] = (),
    ) -> LEACEProjection:
        return super().load(
            path,
            ("shrinkage", "model_", "proj_right", "proj_left", "classes_", *extra_state_keys),
        )


class COVProjection(LEACEProjection):
    """Source COV estimator using the orthogonal LEACE fitter method."""

    def __init__(
        self,
        *,
        n_components: int | None = None,
        name: str = "COV",
        projection_subtract_mean: bool = True,
        shrinkage: bool = True,
    ) -> None:
        super().__init__(
            n_components=n_components, name=name,
            projection_subtract_mean=projection_subtract_mean, shrinkage=shrinkage,
        )

    def _fit_method(self) -> str:
        return "orth"


class RandomProjection(BaseProjection):
    """Immediately configured Gaussian random basis, without a fit step."""

    def __init__(
        self,
        feature_count: int | None = None,
        *,
        n_components: int = 40,
        name: str = "RANDOM",
    ) -> None:
        self.name = name
        self.n_components = n_components
        self.mean = 0
        self.projection_subtract_mean = False
        self.components = (
            None if feature_count is None else
            np.random.normal(0, 1 / np.sqrt(n_components), size=(n_components, feature_count))
        )
        self._transform_matrices = {}
        self._projection_matrices = {}
        self._inverse_transform_matrices = {}


class LDAProjection(BaseProjection):
    """Linear discriminant projector fitted to one target array."""

    def __init__(
        self,
        *,
        n_components: int | None = None,
        name: str = "LDA",
        projection_subtract_mean: bool = True,
    ) -> None:
        self.name = name
        self.n_components = n_components
        self.projection_subtract_mean = projection_subtract_mean
        self._transform_matrices = {}
        self._projection_matrices = {}
        self._inverse_transform_matrices = {}

    def fit(self, X: np.ndarray, y: np.ndarray) -> LDAProjection:
        if X.ndim != 2:
            raise ValueError("X must have shape (samples, features)")
        if y.ndim != 1 or len(y) != len(X):
            raise ValueError("y must be one-dimensional with one label per sample")
        self.model_ = LinearDiscriminantAnalysis(store_covariance=True).fit(
            X, y,
        )
        self.components = self.model_.scalings_.T
        self.mean = self.model_.xbar_
        self.classes_ = self.model_.classes_
        self._transform_matrices = {}
        self._projection_matrices = {}
        self._inverse_transform_matrices = {}
        return self

    def transform(self, X: np.ndarray, n_components: int | None = None) -> np.ndarray:
        N = self.n_components if n_components is None else n_components
        return self.model_.transform(X)[:, :N]

    def save(self, path: str | Path, extra_state: dict = {}) -> None:
        state = {"model_": self.model_, "classes_": self.classes_}
        state.update(**extra_state)
        super().save(path, extra_state=state)

    @classmethod
    def load(
        cls, path: str | Path, extra_state_keys: tuple[str, ...] = (),
    ) -> LDAProjection:
        return super().load(path, ("model_", "classes_", *extra_state_keys))


class IdentityProjection(BaseProjection):
    """Pass-through projector (no-op)."""

    def __init__(self, name: str = "Identity", n_components: int | None = None) -> None:
        self.name = name
        self.n_components = n_components
        self.components = None
        self.mean = None

    def transform(self, X: np.ndarray, **_) -> np.ndarray:
        return X

    def inverse_transform(self, Z: np.ndarray, **_) -> np.ndarray:
        return Z

    def project(self, X: np.ndarray, **_) -> np.ndarray:
        return X

    def reject(self, X: np.ndarray, **_) -> np.ndarray:
        return np.zeros_like(X)

    def save(self, *args, **kwargs):
        raise NotImplementedError("IdentityProjection has no parameters.")

    @classmethod
    def load(cls, *args, **kwargs):
        raise NotImplementedError("IdentityProjection has no parameters.")
