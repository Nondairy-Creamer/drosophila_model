# drosophila_model data ({version})

Data for [drosophila_model](https://github.com/Nondairy-Creamer/drosophila_model):
connectome-constrained recurrent models of the *Drosophila* optic lobe,
trained to report optic flow and self-motion (rotation and translation) from
visual input. They are the repository's GitHub release `data-{version}`, in
two bundles:

- `drosophila_model_baked_{version}.zip` (about 250 MB): the processed files
  the code reads. This is all you need to use the code.
- `drosophila_model_raw_{version}.zip` (about 3.2 GB): the raw files the
  processed ones are built from, for rebuilding them from scratch. GitHub caps
  a release file at 2 GiB, so it is split into
  `drosophila_model_raw_{version}.zip.001`, `.002`, ...: join the parts in
  order to get the zip
  (`cat drosophila_model_raw_{version}.zip.0* > drosophila_model_raw_{version}.zip`,
  or open the `.001` with 7-Zip).

## Usage

The repository downloads and unpacks these files itself. From a clone of the
repository, inside the Python environment you want to use:

```
bash setup.sh
```

This installs the dependencies, downloads
`drosophila_model_baked_{version}.zip` into the data folder
(`../drosophila_model_data` by default), checks its SHA-256 and unpacks it.
`bash setup.sh --raw`
downloads the raw zip instead and rebuilds the processed files from it.

With a copy of either zip you downloaded yourself, run
`bash setup.sh <data_folder> <path to the zip>`, or
`python -m drosophila_model.data_handling.prepare_data --zip <path>` once the
package is installed.

## Contents

Each zip holds one folder, `drosophila_model_data/`, laid out as the code
expects its data folder, so both can be unpacked into the same place.

### drosophila_model_baked_{version}.zip

Built from the raw bundle by the repository's `data_handling` scripts.

`connectome_data/baked/`

- `cell_info.parquet`: every neuron, with its cell type, dominant
  neurotransmitter, the neuropils it has synapses in, its size statistics and
  classification.
- `synapses.parquet`: synapse counts for every (presynaptic neuron,
  postsynaptic neuron, neuropil), with the predicted neurotransmitter.
- `optic_lobe_column_index.parquet`: the hexagonal optic-lobe column of each
  columnar neuron.
- `optic_lobe_columns.parquet`: the viewing direction of each of the 796
  right-eye columns, from the eye map. Eye-map columns are matched one-to-one
  to FlyWire Mi1 neurons; the 29 rim columns the eye map misses are
  interpolated over the hex grid (`source == 'interpolated'`).
- `ct1_pieces.parquet`, `ct1_connections.parquet`: the CT1 split. CT1, which
  spans every column, is replaced by per-column medulla and lobula pieces:
  each CT1 synapse goes to the column most common among its 50 nearest
  synapses of unicolumnar neurons. Already merged into the tables above.

`stimulus/baked/`

- `all_panoramic.pt`: the 100 panoramas as one PyTorch float32 tensor
  (panorama, height, width).

`eyemap_T4_LICENSE`: the GPL-3.0 license of the eye map that
`optic_lobe_columns.parquet` is derived from.

### drosophila_model_raw_{version}.zip

`connectome_data/raw/`: FlyWire Codex downloads for the FAFB connectome,
release v783 (https://codex.flywire.ai, downloaded 2026-10-26), unmodified.

- `consolidated_cell_types.csv.gz`: the cell type of each neuron.
- `connections_princeton_no_threshold.csv.gz`: synapse counts for every
  (presynaptic neuron, postsynaptic neuron, neuropil), with the predicted
  neurotransmitter, unthresholded.
- `column_assignment.csv.gz`: the hexagonal optic-lobe column of each
  columnar neuron.
- `fafb_v783_princeton_synapse_table.csv.gz`: every synapse with its
  position. Used to split CT1 into per-column pieces and to match the eye map
  to the FlyWire columns.
- `cell_stats.csv.gz`: length, surface area and volume of each neuron.
- `classification.csv.gz`: flow, super class, class, sub class and side of
  each neuron.

`connectome_data/raw/eyemap_T4/`: from https://github.com/reiserlab/eyemap_T4
at commit `99d2a43123db636cedb55af9ff31a59657e7d17e`, unmodified, with its
`LICENSE` (GPL-3.0).

- `data/eyemap.RData`: the eye map of Zhao et al. (2025), medulla column
  positions and the viewing direction of each ommatidium.

`stimulus/raw/`

- `panorama_001-100.rar`: the archive `data_001-100.rar` of Meyer et al.
  (2014), renamed. 100 panoramic high-dynamic-range images as MATLAB `.mat`
  files.

## Citation and licenses

If you use these data, please cite the repository and the original sources:

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

Licenses: the FlyWire data are released under CC BY-NC 4.0, so these data
are shared under CC BY-NC 4.0 (non-commercial use only). Two parts keep their
own licenses: the eye map (`eyemap_T4/`, and `optic_lobe_columns.parquet`
derived from it) is GPL-3.0, and the panoramas (`panorama_001-100.rar`,
`all_panoramic.pt`) are under the Open Data Commons Attribution License
(ODC-By 1.0).
