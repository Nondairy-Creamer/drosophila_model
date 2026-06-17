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

## Optic Flow

The observer is a spherical observer looking outward in many directions
simultaneously. As the observer rotates and translates, the visual scene
moves across the retina, producing optic flow.

Flow is represented in local spherical coordinates at each column:

- **u_right**: positive = rightward motion on the retina
- **u_up**: positive = upward motion on the retina

`generate_optic_flow` (in `analysis/analysis_utilities.py`) computes flow
by projecting the 3D velocity onto the local `e_phi` and `e_theta` basis
vectors.  The projection `u_right = -(u · e_phi)` already yields the
**same** `u_right` sign for symmetric left/right-eye columns — both the
physical 3D flow and the `e_phi` basis vector mirror together, so the
angular flow component is consistent across eyes without any correction.

### Two-Eye Flow Pipeline

1. `get_stimulus` (with `num_eyes=2`) computes flow for both eyes by concatenating
   right-eye columns (phi) with left-eye columns (-phi) and calling
   `generate_optic_flow` on all columns at once.  No sign correction is
   needed on the resulting `u_right`.
2. The model processes left-eye input by feeding it through the right-eye
   model (which sees it as mirrored right-eye data).  The decoder outputs
   `u_right` in the right-eye convention (horizontally flipped relative to
   physical left-eye flow).
3. The model's left-eye `u_right` predictions are **flipped** (multiplied
   by -1) in `estimate_ego_motion_from_image` to undo the mirror and
   produce correct physical flow.
4. Training loss compares the flipped predictions against the ground truth
   directly.
