"""
bone_marrow: library code for the bone marrow cell classifier.

None of these modules is an entry point; scripts/, tools/ and tests/ import them.
    config      TrainConfig dataclass (every tunable field), typed YAML / --set loading
    preprocess  per-image transforms and the augmentation presets (AUG_PRESETS)
    data        labels, stratified subsets, class merge/drop, class weights, sampler,
                loaders, build_data(cfg)
    loader      multiprocess batch loader whose workers never import torch
    models      baseline CNN, ResNet-50, EfficientNet-B0 and fine-tuning helpers
    engine      losses, optimizer and LR schedule, train/eval loops, fit(), benchmark()
    metrics     macro-F1 (primary) and the other metrics, plots
    cli         run_and_exit() for the entry scripts

Keep this file free of imports. Loader worker processes import
bone_marrow.preprocess and bone_marrow.loader, which runs this file first, so
anything imported here would load into every worker (torch alone is ~1 GB each).

preprocess.py and models.py have sanity-check mains; run them from the repo root:
    python -m bone_marrow.preprocess
    python -m bone_marrow.models
"""
