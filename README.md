# GenerativeModelsOnPhaseSpace
Public code repository for arXiv:2604.02415

The model in Sec. III.A is `muon_model.py`, trained with `train_muon.py`. The models of Sec. III.B and IV.A are `model_singular.py`, trained with `train_singular.py`. Both modes are sampled with `generate.py`.

The comparisons with other diffusion and flow matching architectures in Sec. IV.B. are `model_pspace_ddpm.py`, `model_pspace_fm.py`, and `model_qspace_fm.py`, trained and sampled with the corresponding `train` and `generate` scripts.

All models use the utilities in `utils.py`.

Trained models, training data, and generated samples used to make the plots in the paper are in the directories TrainedModels, TrainingData, and GeneratedSamples, respectively. Models are saved as PyTorch files which may be loaded with the `load` method in the corresponding model Python file. Training data is saved as `(NEvents, NParticles,3)` torch arrays of 3-vectors in p-space in the CM frame with unit energy. The 1M SARGE events are too large to store here, so they may be generated with `SARGE.py`.

All figures used in the paper may be generated with `PlotsForPaper.ipynb`.

[to do: add generated samples from the Sec. IV.B. models, add plotting notebook]
