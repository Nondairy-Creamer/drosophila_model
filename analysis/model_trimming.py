"""
Model trimming: iteratively silence least important neurons and track loss.

Importance score = incoming_synapses * outgoing_synapses * input_strength * output_strength
"""
import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import numpy as np
import neural_models as nm
import data_utilities as du
import model_utilities as mu
from loss_functions import mse_loss

# --- Configuration ---
model_path = 'models/20260217_172210_gpu0_run0/'
loss_threshold = 1.10  # Stop when loss increases to 110% of starting value (10% worse)
local_config = du.load_config()

# --- Load the model ---
print("Loading model...")
device = local_config['model']['device']
print('running on', device)
model = nm.NeuralModel.load(model_path, device=device)
dtype = model.dtype

# --- Load training data ---
print("Loading training data...")
model_data_path = Path(model_path)
stimulus = torch.load(model_data_path / 'cell_inputs.pt', map_location=device).to(dtype=dtype)
target = torch.load(model_data_path / 'target.pt', map_location=device).to(device=device, dtype=dtype)

# --- Calculate importance score for each cell type ---
print("Calculating cell type importance scores...")

# Get average connections matrix (num_cell_types, num_cell_types)
# ave_con[post_type, pre_type] = average synapse count from pre_type to post_type
ave_con = mu.compute_average_connections(model)

# Get input and output strengths per cell type (already at cell type level)
input_strength = model.input_strength_log.exp().detach().cpu().numpy()  # (num_cell_types,)
output_strength = model.output_strength_log.exp().detach().cpu().numpy()  # (num_cell_types,)

# Effective incoming weight to type k: input_strength[k] * Σ_i |ave_con[k,i]| * output_strength[i]
effective_incoming = input_strength * (np.abs(ave_con) @ output_strength)  # (num_cell_types,)

# Effective outgoing weight from type k: output_strength[k] * Σ_j |ave_con[j,k]| * input_strength[j]
effective_outgoing = output_strength * (np.abs(ave_con).T @ input_strength)  # (num_cell_types,)

# Calculate importance score per cell type
importance = effective_incoming * effective_outgoing

# Sort cell types by importance (ascending - least important first)
sorted_indices = np.argsort(importance)

print(f"Total cell types: {model.num_cell_types}")
print(f"Importance range: {importance.min():.6f} to {importance.max():.6f}")

# --- Calculate initial loss ---
print("\nCalculating initial loss...")
input_neurons = ['L1', 'L2', 'L3']

with torch.no_grad():
    pred = model.predict(stimulus, input_neurons=input_neurons)[0]
    initial_loss = mse_loss(target, pred['decoder']).item()

print(f"Initial loss: {initial_loss:.6f}")
print(f"Target loss (110% of initial): {initial_loss * loss_threshold:.6f}")

# --- Iteratively silence cell types ---
print("\nStarting cell type trimming...")
print("-" * 60)

# silenced_cell_types = []
silenced_cell_types = ['5th-LNv', 'AOTU026', 'AOTU053', 'AOTU065', 'APDN3', 'AVLP151', 'AVLP534', 'Am1', 'CB0143', 'CB0734', 'CB0802', 'CB1329', 'CB2685', 'CB2848', 'CB3089', 'CB3235', 'CB3238', 'CB3790', 'CB3866', 'CL104', 'DCH', 'DN1-l', 'DN1a', 'DNc01', 'DNg33', 'DNp11', 'DNp31', 'DNpe053', 'Dm1', 'Dm13', 'Dm14', 'Dm16', 'Dm18', 'Dm19', 'Dm3p', 'Dm3v', 'Dm4', 'Dm9', 'DmDRA1', 'DmDRA2', 'H2', 'HSE', 'HSN', 'HSS', 'L4', 'LAL047', 'LC10a', 'LC10b', 'LC10c', 'LC10f', 'LC11', 'LC12', 'LC13', 'LC15', 'LC16', 'LC17', 'LC18', 'LC19', 'LC20a', 'LC20b', 'LC21', 'LC22', 'LC24', 'LC25', 'LC26', 'LC27', 'LC28a', 'LC29', 'LC31b', 'LC31c', 'LC33a', 'LC34', 'LC35', 'LC36', 'LC37a', 'LC39', 'LC4', 'LC40', 'LC41', 'LC43', 'LC44', 'LC45', 'LC46', 'LC6', 'LC9', 'LCe01', 'LCe01a', 'LCe01b', 'LCe02', 'LCe04', 'LCe05', 'LCe06', 'LCe07', 'LCe08', 'LCe09', 'LHPV2i2a', 'LLPC2', 'LLPC3', 'LLPC4', 'LLPt', 'LM102a_L234-M89', 'LMa2', 'LMa4', 'LMt1', 'LMt2', 'LMt4', 'LNd_CRY+_ITP+', 'LPC1', 'LPC2', 'LPT04_HST', 'LPT21', 'LPT22', 'LPT23', 'LPT26', 'LPT27', 'LPT28', 'LPT29', 'LPT30', 'LPT31', 'LPT42_Nod4', 'LPT47_vCal2', 'LPT48_vCal3', 'LPT49', 'LPT51', 'LPT52', 'LPT53', 'LPT54', 'LPT57', 'LPT58', 'LPTe01', 'LPTe02', 'LPi01', 'LPi02', 'LPi03', 'LPi04', 'LPi05', 'LPi06', 'LPi08', 'LPi09', 'LPi10', 'LPi11', 'LPi12', 'LPi13', 'LPi14', 'LPi15', 'LT11', 'LT1a', 'LT1b', 'LT1c', 'LT1d', 'LT33', 'LT34', 'LT36', 'LT37', 'LT38', 'LT39', 'LT40', 'LT41', 'LT42', 'LT43', 'LT47', 'LT51', 'LT52', 'LT56', 'LT57', 'LT58', 'LT59', 'LT60', 'LT61a', 'LT61b', 'LT62', 'LT63', 'LT65', 'LT67', 'LT68', 'LT69', 'LT70', 'LT72', 'LT73', 'LT74', 'LT75', 'LT76', 'LT77', 'LT78', 'LT79', 'LT80', 'LT81', 'LT82a', 'LT83', 'LT84', 'LT85', 'LT86', 'LT87', 'LTe01', 'LTe02', 'LTe03', 'LTe04', 'LTe05', 'LTe06', 'LTe07', 'LTe08', 'LTe09', 'LTe10', 'LTe11', 'LTe12', 'LTe13', 'LTe14', 'LTe15', 'LTe16', 'LTe17', 'LTe18', 'LTe19', 'LTe20', 'LTe21', 'LTe22', 'LTe23', 'LTe24', 'LTe25', 'LTe26', 'LTe27', 'LTe28', 'LTe29', 'LTe30', 'LTe31', 'LTe32', 'LTe33', 'LTe35', 'LTe36', 'LTe37', 'LTe38a', 'LTe38b', 'LTe38c', 'LTe40', 'LTe41', 'LTe42a', 'LTe42b', 'LTe42c', 'LTe43', 'LTe44', 'LTe45', 'LTe46', 'LTe47', 'LTe48', 'LTe49a', 'LTe49b', 'LTe49c', 'LTe49d', 'LTe49e', 'LTe49f', 'LTe50', 'LTe51', 'LTe52a', 'LTe52b', 'LTe53', 'LTe54', 'LTe55', 'LTe56', 'LTe57', 'LTe58', 'LTe59', 'LTe60', 'LTe61', 'LTe62', 'LTe63', 'LTe64', 'LTe65', 'LTe66', 'LTe67', 'LTe68', 'LTe69', 'LTe70', 'LTe72', 'LTe73', 'LTe74', 'LTe75', 'LTe76', 'Lai', 'Lat', 'LatB', 'Li01', 'Li02', 'Li04', 'Li05', 'Li07', 'Li08', 'Li09', 'Li11', 'Li12', 'Li124_56', 'Li14', 'Li15', 'Li16', 'Li18', 'Li19', 'Li20', 'Li21', 'Li22', 'Li23', 'Li25', 'Li26', 'Li27', 'Li28', 'Li29', 'Li30', 'Li31', 'Li32', 'Li33', 'MC65', 'MLt1', 'MLt4', 'MLt5', 'MLt6', 'MLt7', 'MLt8', 'MTe01a', 'MTe01b', 'MTe03', 'MTe04', 'MTe05', 'MTe06', 'MTe08', 'MTe09', 'MTe10', 'MTe11', 'MTe12', 'MTe13', 'MTe14', 'MTe15', 'MTe16', 'MTe17', 'MTe18', 'MTe19', 'MTe20', 'MTe21', 'MTe22', 'MTe23', 'MTe24', 'MTe25', 'MTe26', 'MTe27', 'MTe28', 'MTe29', 'MTe30', 'MTe31', 'MTe32', 'MTe33', 'MTe34', 'MTe35', 'MTe36', 'MTe37', 'MTe38', 'MTe39', 'MTe40', 'MTe41', 'MTe42', 'MTe43', 'MTe44', 'MTe45', 'MTe48', 'MTe49', 'MTe50', 'MTe51', 'MTe52', 'MTe53', 'MTe54', 'MeTu1', 'MeTu2a', 'MeTu2b', 'MeTu3a', 'MeTu3b', 'MeTu3c', 'MeTu4b', 'MeTu4c', 'MeTu4d', 'Mi10', 'Mi13', 'Mi15', 'Nod1', 'Nod2', 'Nod3', 'Nod5', 'PLP020', 'PLP022', 'PLP023', 'PLP032', 'PLP035', 'PLP036', 'PLP037b', 'PLP051', 'PLP068', 'PLP081', 'PLP103b', 'PLP104', 'PLP150a', 'PLP155', 'PLP163', 'PLP177', 'PLP185', 'PLP215', 'PLP231', 'PLP239', 'PLP248', 'PLP249', 'PLP251', 'PLP252', 'Pm02', 'Pm03', 'Pm04', 'Pm06', 'Pm07', 'Pm08', 'Pm09', 'Pm10', 'Pm11', 'Pm12', 'Pm13', 'Pm14', 'Sm03', 'Sm06', 'Sm09', 'Sm11', 'Sm13', 'Sm14', 'Sm16', 'Sm17', 'Sm18', 'Sm19', 'Sm20', 'Sm21', 'Sm22', 'Sm23', 'Sm24', 'Sm25', 'Sm26', 'Sm27', 'Sm28', 'Sm29', 'Sm30', 'Sm31', 'Sm33', 'Sm34', 'Sm35', 'Sm36', 'Sm37', 'Sm38', 'Sm39', 'Sm40', 'Sm41', 'Sm42', 'Sm43', 'T3', 'Tlp14', 'Tlp4', 'Tlp5', 'Tm25', 'Tm27', 'Tm31', 'Tm33', 'Tm34', 'Tm35', 'Tm37', 'Tm5e', 'Tm7', 'Tm8b', 'TmY16', 'TmY3', 'VCH', 'VS1', 'VS2', 'VS3', 'VS4', 'VS5', 'VS6', 'VS7', 'VS8', 'VST1', 'VST2', 'VSm', 'Y11', 'Y12', 'aMe1', 'aMe12', 'aMe15', 'aMe17a1', 'aMe17a2', 'aMe17b', 'aMe17c', 'aMe20', 'aMe22', 'aMe24', 'aMe25', 'aMe26', 'aMe3', 'aMe4', 'aMe5', 'aMe6a', 'aMe8', 'cL01', 'cL02a', 'cL02b', 'cL02c', 'cL02d', 'cL03', 'cL04', 'cL05', 'cL06', 'cL07', 'cL08', 'cL09', 'cL10', 'cL11', 'cL12', 'cL13', 'cL14', 'cL15', 'cL20', 'cL21', 'cL22a', 'cL22b', 'cL22c', 'cLLPM01', 'cLLPM02', 'cLP01', 'cLP02', 'cLP03', 'cLP04', 'cLP05', 'cLPL01', 'cM01a', 'cM01b', 'cM01c', 'cM02a', 'cM02b', 'cM04', 'cM05', 'cM06', 'cM07', 'cM08b', 'cM08c', 'cM09', 'cM10', 'cM12', 'cM13', 'cM14', 'cM15', 'cM16', 'cM17', 'cM18', 'cML01', 'cML02', 'cMLLP01', 'cMLLP02', 'l-CPDN3', 'mALC3', 'mALC4', 'mALC5', 'mALC6', 'mALD1', 'mALD2', 's-CPDN3A', 's-LNv', 'vCal1']
skipped_types = ['L1', 'L2', 'L3', 'T4a', 'T4b', 'T4c', 'T4d', 'T5a', 'T5b', 'T5c', 'T5d']
current_loss = initial_loss

for cell_type_idx in sorted_indices:
    # Get the cell type name
    cell_type = model.unique_cell_types[cell_type_idx]
    
    if cell_type in skipped_types:
        continue

    if cell_type in silenced_cell_types:
        continue

    # Add to silenced list
    silenced_cell_types.append(cell_type)
    
    # Create a fresh model copy and apply silencing
    model_copy = nm.NeuralModel.load(model_path, device=device)
    model_copy.silence_neurons(silenced_cell_types, invert_silencing=False)
    
    # Calculate loss with silenced cell types
    with torch.no_grad():
        pred = model_copy.predict(stimulus, input_neurons=input_neurons)[0]
        current_loss = mse_loss(target, pred['decoder']).item()
    
    loss_ratio = current_loss / initial_loss
    
    print(f"Silenced {len(silenced_cell_types)} / {model.num_cell_types} cell types | Latest: {cell_type:10s} | "
          f"Loss: {current_loss:.6f} | Ratio: {loss_ratio:.4f}")
    
    # Check if we've reached the threshold
    if loss_ratio >= loss_threshold:
        print("-" * 60)
        print(f"\nStopped: Loss reached {loss_ratio:.2%} of initial value")
        print(f"Total cell types silenced: {len(silenced_cell_types)}")
        print(f"Silenced cell types: {sorted(silenced_cell_types)}")
        break

else:
    print("-" * 60)
    print(f"\nCompleted all cell types without reaching threshold")
    print(f"Final loss ratio: {current_loss / initial_loss:.2%}")

print("\nDone!")
