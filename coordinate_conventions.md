# Coordinate System & Conventions

## World Frame

Right-handed coordinate system:

- **+x**: forward
- **+y**: left
- **+z**: up

Positive rotation is **counterclockwise** when viewed from the tip of the
rotation axis (right-hand rule).

| Rotation axis | Positive direction | Common name |
|---|---|---|
| x (forward) | left ear up, right ear down | right roll |
| y (left) | nose down, tail up | pitch down |
| z (up) | nose left | yaw left |

## Spherical Coordinates

Standard physics convention:

- **theta** (polar angle): 0 at +z (up), pi/2 at the equator (forward), pi at -z (down)
- **phi** (azimuthal angle): 0 at +x (forward), +pi/2 at +y (left), -pi/2 at -y (right)

Cartesian conversion:

```
x = sin(theta) * cos(phi)
y = sin(theta) * sin(phi)
z = cos(theta)
```

## Eyes

The model is of the **right eye** only. The right eye covers the right half
of visual space (phi < 0). Column positions are stored as `(phi, theta)` in
`connectome_data/baked/optic_lobe_columns.parquet` and represent right-eye viewing directions.

The **left eye** is the mirror image of the right eye. Its viewing directions
are at `(-phi, theta)` — the same polar angle but with negated azimuth.
Two-eye column sets are ordered `[right | left]`
(`OpticLobeRegion.column_phi_theta_two_eyes`,
`ego_motion.load_viewing_directions`). Likewise, the left-eye camera offset
is the sagittal mirror `(x, -y, z)` of the right-eye offset
(`OpticLobeRegion.mirror_camera_offset`).

## Optic Flow

The observer is a spherical observer looking outward in many directions
simultaneously. As the observer rotates and translates, the visual scene
moves across the retina, producing optic flow.

Flow is represented in local spherical coordinates at each column:

- **u_right**: positive = rightward motion on the retina
- **u_up**: positive = upward motion on the retina

`generate_optic_flow` (in `model_definition/ego_motion.py`) computes flow
by projecting the 3D velocity onto the local `e_phi` and `e_theta` basis
vectors.  The projection `u_right = -(u · e_phi)` already yields the
**same** `u_right` convention for left- and right-eye columns (positive =
toward decreasing phi, i.e. rightward in the world frame), so no per-eye
sign correction is applied.  Under mirror-symmetric motion the two eyes'
values therefore differ in sign (forward translation: `u_right > 0` on the
right eye, `< 0` on the left), while a yaw gives both eyes the same sign.

### Two-Eye Flow Pipeline

1. `get_stimulus` (in `data_handling/stimulus_generators.py`, with
   `num_eyes=2`) samples right-eye columns (phi) followed by left-eye
   columns (-phi); the synthetic generators (sine gratings, panoramas) call
   `generate_optic_flow` on all columns at once.  No sign correction is
   needed on the resulting `u_right`.
2. The model processes left-eye input by feeding it through the right-eye
   model (which sees it as mirrored right-eye data;
   `OpticLobeRegion.predict_two_eyes`).  The flow readout outputs
   `u_right` in the right-eye convention (horizontally flipped relative to
   physical left-eye flow).
3. The model's left-eye `u_right` predictions are **flipped** (multiplied
   by -1, the readout's `flip_for_left_eye`) in `predict_two_eyes` — and in
   `estimate_ego_motion_from_image` for a left-eye-only solve — to undo the
   mirror and produce correct physical flow.
4. Training loss compares the flipped predictions against the ground truth
   directly (`training_utilities.model_outputs` concatenates both eyes'
   readouts in the same `[right | left]` column order as the targets).
