"""
src/features/text_features.py
==============================
Text Feature Pipeline — lightweight TF-IDF + SVD transformation.

Used at training time (fit_transform on train split only) and at
inference time (transform on new user-supplied problem descriptions).

The training-time analogue of the user's problem_description is the
historical 'reason' field in the merged dataset.  At inference time,
the user provides their own text which is projected through the same
vocabulary + SVD components that were fit on training data.

Public API
----------
  ReportedIssueTextTransformer
      .fit(texts)                  — fit TF-IDF + SVD on training texts
      .transform(texts)            — project texts to 24 SVD components
      .fit_transform(texts)        — fit + transform in one pass
      .get_feature_names()         — list of output column names

Leakage prevention
------------------
The TF-IDF vocabulary and SVD components are fit ONLY on training-period
data.  The transformer must be saved after fit() and loaded at inference
time — never re-fit on test/validation data.

Memory note
-----------
The vocabulary and SVD matrix are compact (<1 MB after serialisation).
They do not require the full dataset to be in memory at inference time.
"""

from __future__ import annotations

import logging
from typing import Optional

import numpy as np
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

log = logging.getLogger(__name__)

# Number of SVD components — must match the production model's feature schema
N_SVD_COMPONENTS = 24

# TF-IDF vocabulary size cap — keeps the intermediate matrix manageable
TFIDF_MAX_FEATURES = 5000

# Column name prefix for SVD output features
SVD_PREFIX = "reported_issue_svd_"


class ReportedIssueTextTransformer:
    """
    Transforms raw text (reason / problem_description / device_description)
    into a fixed-length numeric vector using TF-IDF + truncated SVD (LSA).

    The output has exactly N_SVD_COMPONENTS columns, named:
        reported_issue_svd_00, reported_issue_svd_01, ..., reported_issue_svd_23

    Parameters
    ----------
    n_components : int
        Number of SVD latent dimensions (default 24).
    max_features : int
        Maximum TF-IDF vocabulary size (default 5000).
    random_state : int
        Random seed for TruncatedSVD reproducibility.
    """

    def __init__(
        self,
        n_components: int = N_SVD_COMPONENTS,
        max_features: int = TFIDF_MAX_FEATURES,
        random_state: int = 42,
    ) -> None:
        self.n_components = n_components
        self.max_features = max_features
        self.random_state = random_state

        self.vectorizer = TfidfVectorizer(
            max_features=max_features,
            stop_words="english",
            ngram_range=(1, 2),
            min_df=2,
            max_df=0.95,
            sublinear_tf=True,
            dtype=np.float32,
        )
        self.svd = TruncatedSVD(
            n_components=n_components,
            random_state=random_state,
        )

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _prepare_texts(texts) -> list[str]:
        """Convert any text input to a list of clean strings."""
        if hasattr(texts, "fillna"):
            # pandas Series
            return texts.fillna("").astype(str).tolist()
        if isinstance(texts, np.ndarray):
            return [str(t) if t is not None else "" for t in texts]
        return [str(t) if t is not None else "" for t in texts]

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, texts) -> "ReportedIssueTextTransformer":
        """
        Fit the TF-IDF vocabulary and SVD components on training texts.

        Parameters
        ----------
        texts : array-like of str
            The 'reason' field from training events (or concatenated
            device_description + reason).

        Returns
        -------
        self
        """
        clean = self._prepare_texts(texts)
        log.info(
            "ReportedIssueTextTransformer.fit: %d texts, max_features=%d, n_components=%d",
            len(clean),
            self.max_features,
            self.n_components,
        )
        tfidf_matrix = self.vectorizer.fit_transform(clean)
        log.info(
            "  TF-IDF matrix: %d × %d", tfidf_matrix.shape[0], tfidf_matrix.shape[1]
        )
        self.svd.fit(tfidf_matrix)
        log.info(
            "  SVD explained variance ratio sum: %.4f",
            self.svd.explained_variance_ratio_.sum(),
        )
        self._is_fitted = True
        return self

    def transform(self, texts) -> np.ndarray:
        """
        Project texts into the latent SVD space.

        Parameters
        ----------
        texts : array-like of str
            Raw text strings.

        Returns
        -------
        np.ndarray of shape (n_samples, n_components), float32.
        """
        if not self._is_fitted:
            raise RuntimeError(
                "ReportedIssueTextTransformer must be fitted before transform(). "
                "Call fit() or fit_transform() first, or load a saved transformer."
            )
        clean = self._prepare_texts(texts)
        tfidf_matrix = self.vectorizer.transform(clean)
        svd_matrix = self.svd.transform(tfidf_matrix)
        return svd_matrix.astype(np.float32)

    def fit_transform(self, texts) -> np.ndarray:
        """Fit on and transform the same texts (training path only)."""
        self.fit(texts)
        return self.transform(texts)

    def get_feature_names(self) -> list[str]:
        """Return the list of output column names."""
        return [f"{SVD_PREFIX}{i:02d}" for i in range(self.n_components)]

    @property
    def _is_fitted(self) -> bool:
        """True if the vectorizer has a vocabulary (i.e., has been fit)."""
        return hasattr(self.vectorizer, "vocabulary_")

    @_is_fitted.setter
    def _is_fitted(self, value: bool) -> None:
        # Accept assignments but ignore — fitted state is derived from vectorizer
        pass

    def __setstate__(self, state: dict) -> None:
        """
        Custom unpickling: the old pkl only saves vectorizer + svd.
        Restore default hyperparameters for any missing attributes.
        """
        self.__dict__.update(state)
        # Ensure __init__ params are present (may be absent in old pkls)
        if not hasattr(self, "n_components"):
            self.n_components = (
                self.svd.n_components if hasattr(self, "svd") else N_SVD_COMPONENTS
            )
        if not hasattr(self, "max_features"):
            voc = getattr(getattr(self, "vectorizer", None), "vocabulary_", None)
            self.max_features = len(voc) if voc is not None else TFIDF_MAX_FEATURES
        if not hasattr(self, "random_state"):
            self.random_state = 42

    def __repr__(self) -> str:
        status = "fitted" if self._is_fitted else "not fitted"
        return (
            f"ReportedIssueTextTransformer("
            f"n_components={self.n_components}, "
            f"max_features={self.max_features}, "
            f"status={status})"
        )
