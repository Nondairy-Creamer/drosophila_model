"""
Run the HandTunedModel with default Julia-style parameters on sine wave stimuli
and plot average responses per cell type, grouped like the Julia plot_motion_inputs.
"""
import torch
import numpy as np
import pandas as pd
from functools import partial
from pathlib import Path
import matplotlib
#matplotlib.use('Agg')
from matplotlib import pyplot as plt

from drosophila_model.model_definition import neural_models as nm
from drosophila_model.model_definition import regions as regions_mod
from drosophila_model.config import load_config
from drosophila_model.training import training_utilities as tu
from drosophila_model.utilities.stimulus_utilities import make_lr_sine_wave, make_ud_sine_wave, make_lr_moving_edge, make_ud_moving_edge

def set_taus(model, mi9_tau=0.1, mi4_tau=0.1, ct1_l_tau=0.25, ct1_m_tau=0.25):
    model.set_tau('Mi9', mi9_tau)    # 2.0 scaled by dt_ratio
    model.set_tau('Mi4', mi4_tau)    # 2.0 scaled by dt_ratio
    #model.set_tau('CT1_L', ct1_l_tau) # 5.0 scaled by dt_ratio
    model.set_tau('CT1_M', ct1_m_tau) # 5.0 scaled by dt_ratio

def flip_signs(model):
    # C2/C3 -> L1/L2 are GABAergic-but-functionally-excitatory; that flip
    # is now declared as an anatomy prior on OpticLobeRegion.sign_overrides
    # and applied automatically by ConnectomeModel._apply_region_sign_overrides.
    model.flip_sign_between_cell_types('L1', 'R1-6', factor=-1)
    model.flip_sign_between_cell_types('Lai', 'R1-6', factor=-1)
    model.flip_sign_between_cell_types('R1-6', 'R1-6', factor=0)

def amplify_inhibitory(model, value=2.0):
    model.amplify_inhibitory(value)

def set_mi4_mi9_synapse_amplification(model, value=3.0):
    model.set_synapse_amplification('Mi4', value)
    model.set_synapse_amplification('Mi9', value)

def set_ct1_m_synapse_amplification(model, value=5.0):
    model.set_synapse_amplification('CT1_M', value)

def set_ct1_l_synapse_amplification(model, value=5.0):
    model.set_synapse_amplification('CT1_L', value)

def ablate_r7_r8(model):
    model.ablate_cell_type('R7')
    model.ablate_cell_type('R8')

def set_t4_t5_offsets(model, value=-0.1):
    for ct in T4_types + T5_types:
        model.set_offset(ct, value)

def set_omega(model, value=1.0):
    model.set_omega(value)

def set_default_activation(model, activation_name='sigmoid'):
    model.set_default_activation(activation_name)

def set_activation(model, cell_type, activation_name):
    model.set_activation(cell_type, activation_name)

def set_default_offset(model, value=-1.0):
    for ct in model.unique_cell_types:
        model.set_offset(ct, value)

def find_steady_state(model):
    model.find_steady_state()

def set_offset(model,cell_type,value):
    model.set_offset(cell_type,value)

model_configs = {
        'sigmoid_very_quiet_omega_20': [set_default_activation, partial(set_default_offset, value=-2.0), partial(set_omega, value=20.0),partial(set_t4_t5_offsets, value=-0.05),find_steady_state],

  }

"""
    'sigmoid_very_quiet': [set_default_activation, partial(set_default_offset, value=-2.0)],
    'sigmoid_quiet': [set_default_activation, partial(set_default_offset, value=-1.0)],
    'sigmoid_quie_omega_5': [set_default_activation, partial(set_omega, value=5.0),partial(set_default_offset, value=-1.0)],
        'vanilla_sigmoid':   [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,set_default_activation],

'vanilla_sigmoid_no_inhibitory_ct1_m_L2_L3_offset_0_5':   [ablate_r7_r8,flip_signs,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,set_ct1_m_synapse_amplification,set_default_activation,partial(set_offset, cell_type='L2', value=0.5),partial(set_offset, cell_type='L3', value=0.5)],

'vanilla_sigmoid_no_inhibitory_ct1_m_L2_offset_0_5':   [ablate_r7_r8,flip_signs,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,set_ct1_m_synapse_amplification,set_default_activation,partial(set_offset, cell_type='L2', value=0.5)],
    'vanilla_sigmoid_no_inhibitory_ct1_m':   [ablate_r7_r8,flip_signs,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,set_ct1_m_synapse_amplification,set_default_activation],

'vanilla_sigmoid':   [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,set_default_activation],

'sigmoid_set_omega_3': [set_default_activation,partial(set_omega, value=3.0)],
'sigmoid_set_omega_5': [set_default_activation,partial(set_omega, value=5.0)],
'sigmoid_set_omega_10': [set_default_activation,partial(set_omega, value=10.0)],
    'sigmoid': [set_default_activation],
    'sigmoid_set_ct1_m_synapse_amplification': [set_default_activation,set_ct1_m_synapse_amplification],
    'sigmoid_flip_signs': [set_default_activation,flip_signs],
    'sigmoid_amplify_inhibitory': [set_default_activation,amplify_inhibitory],
    'sigmoid_set_mi4_mi9_synapse_amplification': [set_default_activation,set_mi4_mi9_synapse_amplification],
    'sigmoid_set_ct1_l_synapse_amplification': [set_default_activation,set_ct1_l_synapse_amplification],
    'sigmoid_set_omega': [set_default_activation,set_omega],
    'sigmoid_set_taus': [set_default_activation,set_taus],

    'set_ct1_m_synapse_amplification': [set_ct1_m_synapse_amplification],
    'flip_signs': [flip_signs],
    'amplify_inhibitory': [amplify_inhibitory],
    'set_mi4_mi9_synapse_amplification': [set_mi4_mi9_synapse_amplification],
    'set_ct1_l_synapse_amplification': [set_ct1_l_synapse_amplification],

    'flip_signs_ct1_m': [flip_signs,set_ct1_m_synapse_amplification],

'vanilla':   [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification],
'no_inhib':   [ablate_r7_r8,flip_signs,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification],
'omega_1':   [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,set_omega],
'no_mi4_mi9': [ablate_r7_r8,flip_signs,amplify_inhibitory,set_ct1_l_synapse_amplification,set_omega],
'offset_0_0005': [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,partial(set_t4_t5_offsets, value=-0.0005),set_omega],
'offset_0_001': [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,partial(set_t4_t5_offsets, value=-0.001),set_omega],
'offset_omega_1_5_offset_0_0005': [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,partial(set_t4_t5_offsets, value=-0.0005),partial(set_omega, value=1.5)],
'offset_omega_1_5_offset_0_0005_tau': [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,partial(set_t4_t5_offsets, value=-0.0005),partial(set_omega, value=1.5),set_taus],
'offset_omega_1_5_offset_0_0005_no_inhibitory': [ablate_r7_r8,flip_signs,set_mi4_mi9_synapse_amplification,set_ct1_l_synapse_amplification,partial(set_t4_t5_offsets, value=-0.0005),partial(set_omega, value=1.5)],
'stripped': [],
'vanilla_ct1_m':  [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_m_synapse_amplification,set_ct1_l_synapse_amplification,set_omega],
'vanilla_ct1_m_offset_0_0005':  [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_m_synapse_amplification,set_ct1_l_synapse_amplification,partial(set_t4_t5_offsets, value=-0.0005)],
'vanilla_ct1_m_no_mi4_mi9':  [ablate_r7_r8,flip_signs,amplify_inhibitory,set_ct1_m_synapse_amplification,set_ct1_l_synapse_amplification,set_omega,set_omega],
'vanilla_ct1_m_no_mi4_mi9_no_ct1_l':  [ablate_r7_r8,flip_signs,amplify_inhibitory,set_ct1_m_synapse_amplification,set_omega,set_omega],

'vanilla_ct1_m_omega_1':  [ablate_r7_r8,flip_signs,amplify_inhibitory,set_mi4_mi9_synapse_amplification,set_ct1_m_synapse_amplification,set_ct1_l_synapse_amplification,partial(set_omega, value=1.0)],
'vanilla_ct1_m_omega_1_no_mi4_mi9':  [ablate_r7_r8,flip_signs,amplify_inhibitory,set_ct1_m_synapse_amplification,set_ct1_l_synapse_amplification,partial(set_omega, value=1.0)],
'vanilla_ct1_m_omega_1_no_mi4_mi9_no_ct1_l':  [ablate_r7_r8,flip_signs,amplify_inhibitory,set_ct1_m_synapse_amplification,partial(set_omega, value=1.0)],
"""


# ── Configuration ─────────────────────────────────────────────────────
device = 'cpu'
dtype = 'float32'

dt = 0.005
stim_dt = 0.005
seed = 42

omega = 2.0

stim_time = 6.0        # seconds of moving grating
pre_stim_time = 0.5     # blank before stimulus
wavelength = 30         # degrees (Julia `sine_traveling` uses wavelength in *column spacings*, not degrees)
# Julia uses temporal *period* in seconds: `ω = 2π/period`. Here we use Hz: f = 1/period.
# Example: Julia `sine_traveling(..., period=5)` → use temporal_frequency = 1/5  # 0.2 Hz, not 5.
temporal_frequency = 1   # Hz 

tau_default = 0.05          # 1.0 scaled by dt_ratio (0.005/0.1)
#tau_default=1.0
#stimulus_strength_init = 20.0  # 1.0 scaled by 1/dt_ratio (0.1/0.005)
stimulus_strength_init = 1.0 #1.0  # 1.0 scaled by 1/dt_ratio (0.1/0.005) * 10 to account for exponential highpass filter
input_neurons = ['R1-6']

# Cell types to monitor (flattened later into output_neurons for predict)
cell_type_groups = {
    'Photoreceptors':   ['R1-6'],
    'Lamina (L1-L3)':   ['L1', 'L2', 'L3'],
    'Lamina (L4-L5)':   ['L4', 'L5'],
    'C2 / C3':          ['C2', 'C3'],
    'T4 inputs':        ['Mi1', 'Tm3', 'Mi4', 'Mi9'],
    'T5 inputs':        ['Tm1', 'Tm2', 'Tm4', 'Tm9'],
    'T4':               ['T4a', 'T4b', 'T4c', 'T4d'],
    'T5':               ['T5a', 'T5b', 'T5c', 'T5d'],
}

T4_types = ['T4a', 'T4b', 'T4c', 'T4d']
T5_types = ['T5a', 'T5b', 'T5c', 'T5d']

activation_overrides = {ct: 'relu' for ct in T4_types + T5_types}


# ── Load connectome ───────────────────────────────────────────────────
config = load_config('config.toml')
from drosophila_model.model_definition import connectome_loading as cl
connectome = cl.load_connectome(
    region_specs=regions_mod.build_region_specs(config['model'].get('regions', {})),
    data_folder=config['data']['data_folder'],
    count_threshold=config['model']['synapse_count_threshold'],
)

HandTunedModel = nm.compose_model_class('HandTunedDynamics')

for name, operations in model_configs.items():
    print(f'Creating model: {name}')
    print(f'Operations: {operations}')
    model = HandTunedModel(
        connectome=connectome,
        nonlinearity='Tanh',
        count_threshold=config['model']['synapse_count_threshold'],
        use_layer_norm=False,
        region_kwargs={'optic_lobe': {'use_input_conv': False}},
        device=device,
        dtype=dtype,
        dt=dt,
        stimulus_strength_init=stimulus_strength_init,
        seed=seed,
        omega_init=omega,
        tau_default=tau_default,
        activation_overrides=activation_overrides,
        commit_hash=tu.get_git_commit_hash(),
        excitatory_nt=['ACH'],   
        inhibitory_nt=['GLUT', 'GABA',"HIS"],
    )
    for operation in operations:
        print(f'Applying operation: {operation}')
        operation(model)
    save_dir = Path(f'models/hand_tuned_{name}')
    model.save(save_dir / f'{name}.pt')
    print(f'Model saved to {save_dir / f"{name}.pt"}')

"""
import os

for name in model_configs.keys():
    if  (Path(f'models/hand_tuned_{name}/natural_pano_response.png').exists()):
        print(f'{save_dir}/natural_pano_response.png already exists')
        continue
    print(f'Sweeping model: {name}')
    save_dir = f'models/hand_tuned_{name}'
    command_string = f"python view_natural_scene.py --hex --hex-cell-type T4a T5a --highpass-tau 1 --stim-time 10 --pano-index 30 --save {save_dir}/natural_pano_response.png --model {save_dir}/{name}.pt"
    print(command_string)
    os.system(command_string)
    direction1 = "right"
    direction2 = "left"
    ffmpeg_string= f"ffmpeg -i {save_dir}/hex_stim_pano30_{direction1}_30dps.mp4 -i {save_dir}/hex_T4a_pano30_{direction1}_30dps.mp4 -i {save_dir}/hex_T5a_pano30_{direction1}_30dps.mp4 -filter_complex [0:v][1:v][2:v]hstack=inputs=3[v] -map [v] {save_dir}/natural_pano_response_{direction1}.mp4"
    print(ffmpeg_string)
    os.system(ffmpeg_string)
    command_string+=" --direction left"
    print(command_string)
    os.system(command_string)
    ffmpeg_string= f"ffmpeg -i {save_dir}/hex_stim_pano30_{direction2}_30dps.mp4 -i {save_dir}/hex_T4a_pano30_{direction2}_30dps.mp4 -i {save_dir}/hex_T5a_pano30_{direction2}_30dps.mp4 -filter_complex [0:v][1:v][2:v]hstack=inputs=3[v] -map [v] {save_dir}/natural_pano_response_{direction2}.mp4"
    print(ffmpeg_string)
    os.system(ffmpeg_string)
    ffmpeg_string= f"ffmpeg -i {save_dir}/natural_pano_response_{direction1}.mp4 -i {save_dir}/natural_pano_response_{direction2}.mp4 -filter_complex [0:v][1:v]vstack=inputs=2[v] -map [v] -c:v libx264 -crf 23 {save_dir}/natural_pano_response_stacked.mp4"
    print(ffmpeg_string)
    os.system(ffmpeg_string)
"""