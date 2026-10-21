"""Portable NPZ/JSON prediction checkpoints; no pickle or hidden executable state."""
import json
from pathlib import Path
import numpy as np
from .core import GreenModel, Nystrom
from .baselines import IndependentModel, LiftedGP


def save_model(model, directory):
    path = Path(directory); path.mkdir(parents=True, exist_ok=True)
    if isinstance(model, IndependentModel):
        meta = {"type": "independent", "groups": len(model.models)}
        for g, child in enumerate(model.models): save_model(child, path/("group_%d" % g))
    elif isinstance(model, LiftedGP):
        meta = {"type": "lifted_gp", "groups": len(model.models)}
        arrays = {}
        for g, (z, scale, beta) in enumerate(model.models):
            arrays.update({"z_%d" % g: z, "scale_%d" % g: scale, "beta_%d" % g: beta})
        np.savez_compressed(path/"arrays.npz", **arrays)
    elif isinstance(model, GreenModel):
        f = model.features
        meta = {"type": "green", "order": model.order, "groups": model.groups,
                "feature_center": f.center, "feature_scale": f.scale, "rank": f.rank,
                "scope": "prediction checkpoint; fit diagnostics stored separately"}
        np.savez_compressed(path/"arrays.npz", omega=model.omega, B=model.B, z=f.z, whiten=f.whiten)
    else:
        raise TypeError("unsupported checkpoint model")
    (path/"metadata.json").write_text(json.dumps(meta, indent=2))


def load_model(directory):
    path = Path(directory); meta = json.loads((path/"metadata.json").read_text())
    if meta["type"] == "independent":
        model = IndependentModel()
        model.models = [load_model(path/("group_%d" % g)) for g in range(meta["groups"])]
        model.omega = np.vstack([m.omega for m in model.models])
        return model
    with np.load(path/"arrays.npz", allow_pickle=False) as data:
        if meta["type"] == "lifted_gp":
            model = LiftedGP()
            model.models = [(data["z_%d" % g].copy(), data["scale_%d" % g].copy(), data["beta_%d" % g].copy())
                            for g in range(meta["groups"])]
            return model
        if meta["type"] != "green": raise ValueError("unsupported checkpoint type")
        model = GreenModel(order=meta["order"])
        model.groups, model.omega, model.B = meta["groups"], data["omega"].copy(), data["B"].copy()
        model.features = Nystrom.__new__(Nystrom)
        model.features.center, model.features.scale, model.features.rank = meta["feature_center"], meta["feature_scale"], meta["rank"]
        model.features.z, model.features.whiten = data["z"].copy(), data["whiten"].copy()
        model.mean_beta = model.B.mean(axis=0); model.deviation_beta = model.B-model.mean_beta
        return model
