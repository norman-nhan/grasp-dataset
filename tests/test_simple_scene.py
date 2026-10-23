import numpy as np

from grasp_dataset.simple_scene import DEFAULT_MODELS_ROOT, build_model, catalog, resolve_model


def test_catalog_and_resolve_converted_models():
    models = catalog(DEFAULT_MODELS_ROOT)
    assert models
    mug = resolve_model(DEFAULT_MODELS_ROOT, "025_mug")
    assert mug == DEFAULT_MODELS_ROOT / "025_mug" / "model.xml"


def test_combined_openarm_and_converted_object_model_compiles():
    model_file = DEFAULT_MODELS_ROOT / "025_mug" / "model.xml"
    model, data, center, extent, metadata = build_model(model_file)
    assert metadata["object_id"] == "025_mug"
    assert model.body("025_mug").id > 0
    assert model.body("openarm_right_hand").id > 0
    assert model.nu > 0
    assert np.isfinite(data.qpos).all()
    assert np.isfinite(center).all() and np.all(extent > 0)
