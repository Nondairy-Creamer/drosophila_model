import sys
from pathlib import Path

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent))

import torch
import neural_models as nm
import data_utilities as du

model_folder = Path('models/20260126_125933')

# Load config and determine model class
model_config = du.load_config(model_folder / 'config.toml')
local_config = du.load_config
model_type = model_config['model'].get('model_type')
ModelClass = getattr(nm, model_type)
trained_model = ModelClass.load(model_folder / 'model.pt', device='cpu')
init_model = ModelClass.load(model_folder / 'model_init.pt', device='cpu')
target_model = ModelClass.load(model_folder / 'target_model.pt', device='cpu')
stimulus_in = torch.load(model_folder / 'cell_inputs.pt').to(trained_model.device)

trained_input_weight = trained_model.input_strength_log.exp()
trained_input_weight_init = init_model.input_strength_log.exp()
target_input_weight = target_model.input_strength_log.exp()

trained_output_weight = trained_model.output_strength_log.exp()
trained_output_weight_init = init_model.output_strength_log.exp()
target_output_weight = target_model.output_strength_log.exp()

print('target model sigma max', target_model.calc_sigma_max())
print('trained model sigma max', trained_model.calc_sigma_max())

print('target model input weights', target_model.input_strength_log.exp())
print('trained model input weights', trained_model.input_strength_log.exp())
print('init distance', torch.mean(torch.abs(target_input_weight - trained_input_weight_init)))
print('trained distance', torch.mean(torch.abs(target_input_weight - trained_input_weight)))

print('target model output weights', target_model.output_strength_log.exp())
print('trained model output weights', trained_model.output_strength_log.exp())
print('init distance', torch.mean(torch.abs(target_output_weight - trained_output_weight_init)))
print('trained distance', torch.mean(torch.abs(target_output_weight - trained_output_weight)))

a=1