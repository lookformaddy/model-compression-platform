"""Optional FastAPI surface (skipped entirely when FastAPI is absent).

Serves the baseline model plus any registered compressed variant for a
side-by-side accuracy/size comparison on caller-supplied rows. The core
library never imports this module, so the platform works without it.
"""
from __future__ import annotations

try:
    from fastapi import FastAPI
    from pydantic import BaseModel
except ImportError:  # pragma: no cover - exercised by skip tests
    FastAPI = None  # type: ignore


def create_app(models: dict | None = None):
    if FastAPI is None:
        raise RuntimeError("FastAPI is not installed")
    app = FastAPI(title="model-compression-platform")
    store: dict = models or {}

    class PredictRequest(BaseModel):
        features: list[float]
        variant: str = "original"

    @app.get("/health")
    def health():
        return {"status": "ok", "variants": sorted(store.keys())}

    @app.post("/predict")
    def predict(req: PredictRequest):
        model = store.get(req.variant)
        if model is None:
            return {"error": f"unknown variant: {req.variant}"}
        return {"variant": req.variant, "prediction": int(model.predict(req.features))}

    return app


app = create_app() if FastAPI is not None else None
