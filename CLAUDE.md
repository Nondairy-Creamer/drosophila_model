# Project Notes

## Data folder

There is NO default data path. ALWAYS read `[data].data_folder` from
`config.toml` every single time you need the data location. Never hardcode or
assume a path (in particular, do not assume `../drosophila_model_data`).

Current value (this machine, for reference only — re-read config each time):

```
/Users/mcreamer/Library/CloudStorage/GoogleDrive-matthew.s.creamer@gmail.com/My Drive/memazing/drosophila_model_data
```

### Layout

- `<data_folder>/config.toml` — a copy of run config lives here too
- `<data_folder>/connectome_data/` — connectome (`baked/cell_info.parquet`, `baked/synapses.parquet`)
- `<data_folder>/stimulus/raw/` — raw stimulus source files plus the caches
  produced by `drosophila_model/data_handling/process_raw_data.py <raw_type>`
  (`small_drone`, `rpi_drone`, `random_pano`). The large raw inputs — the drone
  rigs `drone_videos/` + `drone_poses/` (small_drone) and `rpi_drone/` (rpi_drone,
  with `rpi_drone/dome_calib/`), and the raw `panoramic/*.mat` panoramas — are
  read once by `process_raw_data.py`, which writes the cached `.pt` files used by
  the on-the-fly generators in `stimulus_generators.py`: `small_drone.pt`,
  `rpi_drone.pt` (ommatidium-sampled clips at native fps) and `all_panoramic.pt`
  (consolidated raw panorama images). Also here: `fib_fly_view_dataset.pkl`,
  `flyview.pt`.
- `<data_folder>/stimulus/baked/` — fixed, training-ready datasets produced by
  `drosophila_model/data_handling/create_training_data.py <stimulus_type>`,
  one `<stimulus_type>.pt` per type (e.g. `small_drone.pt`, `rpi_drone.pt`).
  Training loads these via `training_utilities.load_processed_stimulus` (no
  on-the-fly generation).
