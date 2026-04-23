from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA


COMPONENTS = ["diff_compression", "yen_vol", "jgb_pressure", "vix"]


def zscore(frame: pd.DataFrame) -> pd.DataFrame:
    return (frame - frame.mean()) / frame.std(ddof=0)


def add_fragility_indexes(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    spread = out["us_2y"] - out["jpy_2y"]
    out["diff_compression"] = -spread.diff() * 100.0
    out["yen_vol"] = out["fx_ret"].rolling(20).std()
    out["jgb_pressure"] = out["jpy_10y_change"].abs().rolling(20).mean()

    z = zscore(out[COMPONENTS])
    valid = z.dropna()
    pca = PCA(n_components=1)
    pc = pd.Series(pca.fit_transform(valid).ravel(), index=valid.index)
    equal = valid.mean(axis=1)
    if pc.corr(equal) < 0:
        pc = -pc
    out["fragility_pca"] = pc
    out["fragility_equal"] = equal
    out["fragility_no_vix"] = zscore(out[["diff_compression", "yen_vol", "jgb_pressure"]]).mean(axis=1)
    out["fragility_japan_only"] = zscore(out[["yen_vol", "jgb_pressure"]]).mean(axis=1)
    out.attrs["pca_explained_variance"] = float(pca.explained_variance_ratio_[0])
    out.attrs["pca_loadings"] = dict(zip(COMPONENTS, pca.components_[0]))
    return out
