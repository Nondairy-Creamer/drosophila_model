# Drosophila connectome models

<p align="center">
  <img src="docs/media/drone_flight.webp" alt="A small drone hovering in front of a textured panel" width="704">
  <br>
  <em>Synthetic drosophila brain in a drone using dense optic flow to do visual stabilization</em>
</p>

<p align="center">
  <img src="docs/media/predicted_flow.webp" alt="The image after the input conv and the predicted optic flow for each eye" width="704">
  <br>
  <em>Predicted optic flow to a randomly rotating filtered natural scene</em>
</p>

This repository is for specifying and training models of the *Drosophila*
nervous system in PyTorch. Each model is a recurrent network whose neurons and
signed synapses come from the FlyWire connectome; only per-cell-type parameters,
such as gains, time constants and input filters, are learned. The default
configuration trains the right optic lobe to report optic flow and ego-motion
from visual input. The default model includes 37,758 neurons, 216 cell types
and 5,887 trainable parameters.

## Setup

**Recommended hardware:** with default settings, a training run takes about
an hour on an NVIDIA A40 GPU and uses about 17 GB of GPU memory. Memory
scales with `[training].batch_size` (32 by default).

1. Download the code:
   ```bash
    git clone https://github.com/Nondairy-Creamer/drosophila_model.git
    cd drosophila_model
   ```
2. Set up a Python environment (Python 3.11 or newer) and activate it, for
   example:
   ```bash
    python -m venv .venv
    source .venv/bin/activate   # Windows (Git Bash): source .venv/Scripts/activate
   ```

    To train on a GPU, install the [PyTorch](https://pytorch.org) build for
    your CUDA version now. Otherwise the next step installs the CPU build.
3. Run the setup script:
   ```bash
    bash setup.sh   # Windows (PowerShell): & "C:\Program Files\Git\bin\bash.exe" setup.sh
   ```

    It installs the dependencies and downloads the data (about 250 MB). The data
    go to `../drosophila_model_data`, next to the repo, unless you pass another
    folder: `bash setup.sh /path/to/data`.

## Train a model

Run all commands from the repo root.

```bash
python -m drosophila_model.training.train
```

This trains with the default settings and saves the run in `models/<timestamp>/`,
including the best model (`model.pt`) and a loss plot
(`figures/training/loss.png`).

To train several models on different seeds, across all your GPUs:

```bash
python -m drosophila_model.training.start_runs --runs-per-gpu 2
```

## Change the settings

The defaults are in
[`config_default.toml`](drosophila_model/config_default.toml) (model and training)
and [`config_default_data.toml`](drosophila_model/config_default_data.toml)
(stimuli). To overwrite them, write the keys you want to change, with their new
values, in a `config.toml` or `config_data.toml` at the repo root. `setup.sh`
already creates `config.toml`. For example:

```toml
[training]
num_epochs = 100
```

## Analyze a model

Analysis scripts take a run folder and save their figures in
`<run>/figures/<analysis_name>/`. To run the standard set of tests:

```bash
python -m drosophila_model.analysis.model_analysis.benchmark_model models/<run>
```

To make the flow movie shown above:

```bash
python -m drosophila_model.analysis.movie_scripts.run_model_on_stimulus models/<run> --stim-type random_pano --input-conv --duration 10
```

## Data sources

If you use these data, please cite the original sources:

- Dorkenwald S. et al. (2024). Neuronal wiring diagram of an adult brain.
  *Nature* 634, 124-138. https://doi.org/10.1038/s41586-024-07558-y
- Schlegel P. et al. (2024). Whole-brain annotation and multi-connectome cell
  typing of *Drosophila*. *Nature* 634, 139-152.
  https://doi.org/10.1038/s41586-024-07686-5
- Matsliah A. et al. (2024). Neuronal parts list and wiring diagram for a
  visual system. *Nature* 634, 166-180.
  https://doi.org/10.1038/s41586-024-07981-1
- Zhao A. et al. (2025). Eye structure shapes neuron function in *Drosophila*
  motion vision. *Nature* 646, 135-142.
  https://doi.org/10.1038/s41586-025-09276-5
- Meyer H.G., Schwegmann A., Lindemann J.P., Egelhaaf M. (2014). Panoramic
  high dynamic range images in diverse environments. Bielefeld University.
  https://doi.org/10.4119/unibi/2689637

The FlyWire connectome data are licensed under CC BY-NC 4.0 (non-commercial use
only). The eye map and data derived from it are licensed under GPL-3.0, and the
panoramas under the Open Data Commons Attribution License (ODC-By 1.0).

## License

The code is released under the MIT License (see [LICENSE](LICENSE)).
The license does not cover the data, which keep the licenses listed above,
or the images in `docs/media/`.

## More detail

- [Reference](docs/reference.md): the data pipeline, every training tool and
  option, every analysis script, and the code layout
- [Model internals](model_architecture.md)
- [Coordinate and flow sign conventions](coordinate_conventions.md)
- [Derivations (LyX)](math/)
