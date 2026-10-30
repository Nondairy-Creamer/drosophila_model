"""Training-path tests: baked-dataset loading, and one-step ``fit``
runs on the small test model with one- and two-eye data."""
import math

import numpy as np
import pytest
import torch

from drosophila_model.training import training_utilities as tu
from tests.model_factory import build_model_from_test_config

# Right-eye camera offset of a two-camera rig (metres); the left eye mirrors it.
EYE_OFFSET = (0.05, -0.05, 0.013)


def _dataset(model, num_eyes, num_stimuli=2, num_time=4):
    """Random inputs and targets laid out like a baked training dataset."""
    gen = torch.Generator().manual_seed(0)
    num_cols = num_eyes * model.regions['optic_lobe'].num_columns

    def randn(*shape):
        return torch.randn(*shape, generator=gen, dtype=model.dtype).to(model.device)

    inputs = randn(num_stimuli, num_time, num_cols)
    target = {'optic_flow': 0.1 * randn(num_stimuli, num_time, num_cols, 2)}
    if num_eyes == 2:
        target['rotation'] = 0.1 * randn(num_stimuli, num_time, 3)
        target['translation'] = randn(num_stimuli, num_time, 3)
    train_mask = torch.ones(inputs.shape, dtype=torch.bool, device=model.device)
    return inputs, target, train_mask


def _train_one_step(model, num_eyes, loss_components, **kwargs):
    inputs, target, train_mask = _dataset(model, num_eyes)
    initial = {k: v.detach().clone() for k, v in model.named_parameters()}
    _, loss_history, _, info = tu.fit(
        model, inputs, target,
        output_neurons=None, input_neurons=None,
        loss_components=loss_components,
        rng=np.random.default_rng(0),
        num_epochs=1, save_every=1, alpha=1e-3,
        train_mask=train_mask, num_eyes=num_eyes,
        **kwargs,
    )
    changed = [k for k, v in model.named_parameters() if not torch.equal(v.detach(), initial[k])]
    return loss_history, info, changed


@pytest.mark.parametrize("dynamics_type", ["CurrentDynamics", "ConductanceDynamics"])
def test_train_one_eye_flow_loss(tmp_path, dynamics_type):
    model, _ = build_model_from_test_config(
        readouts=['FlowConv2Readout', 'L1Reg'], dynamics_type=dynamics_type)
    run_dir = tmp_path / "run"

    loss_history, info, changed = _train_one_step(
        model, 1, ['flow_loss'], model_save_folder=run_dir)

    # The regularizer readout is logged next to the configured loss.
    assert set(loss_history) == {'flow_loss', 'L1Reg'}
    assert all(len(v) == 1 and math.isfinite(v[0]) for v in loss_history.values())
    assert math.isfinite(info['best_loss'])
    assert changed, "one optimizer step changed no parameters"
    assert (run_dir / "model.pt").exists()
    assert (run_dir / "model_opt.pt").exists()
    assert (run_dir / "figures" / "training" / "loss.png").exists()


@pytest.mark.parametrize("history_best, saves", [(1e-12, False), (1e12, True)])
def test_resumed_run_saves_only_on_improvement(tmp_path, history_best, saves):
    # Resuming with a loss history: model.pt already holds that history's best
    # epoch, so it is only overwritten once a new epoch beats it.
    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'])
    run_dir = tmp_path / "run"

    loss_history, info, _ = _train_one_step(
        model, 1, ['flow_loss'], model_save_folder=run_dir,
        loss_history={'flow_loss': [history_best]})

    assert len(loss_history['flow_loss']) == 2
    assert (run_dir / "model.pt").exists() == saves
    if not saves:
        assert info['best_loss'] == pytest.approx(history_best)


def test_train_two_eye_flow_loss():
    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'])

    loss_history, _, changed = _train_one_step(
        model, 2, ['flow_loss'], eye_offset=EYE_OFFSET)

    assert list(loss_history) == ['flow_loss']
    assert math.isfinite(loss_history['flow_loss'][0])
    assert changed


def test_train_two_eye_odometry_loss():
    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'])
    loss_components = ['flow_loss', 'odometry_loss']

    # The training forward pass, reproduced by the shared evaluation helpers.
    inputs, target, train_mask = _dataset(model, 2)
    with torch.no_grad():
        outputs, _ = tu.model_outputs(
            model, inputs, train_mask, num_eyes=2, eye_offset=EYE_OFFSET)
        components = tu.compute_loss_components(
            model, outputs, target, train_mask, loss_components)
    assert outputs['pred_rotation'].shape == target['rotation'].shape
    assert outputs['pred_translation'].shape == target['translation'].shape
    assert outputs['FlowConv2Readout'].shape == target['optic_flow'].shape
    assert set(components) == set(loss_components)
    assert all(torch.isfinite(v) for v in components.values())

    loss_history, _, changed = _train_one_step(
        model, 2, loss_components, eye_offset=EYE_OFFSET)

    assert set(loss_history) == set(loss_components)
    assert all(math.isfinite(v[0]) for v in loss_history.values())
    assert changed


@pytest.mark.parametrize("loss_components, num_eyes, match", [
    (['not_a_loss'], 1, "Unknown loss component"),
    (['odometry_loss'], 1, "requires a two-eye dataset"),
    (['flow_loss'], 3, "num_eyes must be 1 or 2"),
])
def test_fit_rejects_bad_settings(loss_components, num_eyes, match):
    # Settings are validated before the model or data are touched.
    with pytest.raises(ValueError, match=match):
        tu.fit(None, None, None, None, None, loss_components, num_eyes=num_eyes)


def _write_baked(folder, name, num_eyes, eye_offset, model_dt=0.01, num_stimuli=2,
                 train_fraction=None):
    """A tiny ``<type>_<split>.pt`` in the layout create_training_data writes."""
    num_time, num_cols = 3, 5 * num_eyes
    two_eye = num_eyes == 2
    folder.mkdir(parents=True, exist_ok=True)
    payload = {
        'stimulus': torch.rand(num_stimuli, num_time, num_cols, dtype=torch.float64),
        'optic_flow': torch.rand(num_stimuli, num_time, num_cols, 2),
        'train_mask': torch.ones(num_stimuli, num_time, num_cols, dtype=torch.bool),
        'rotation': torch.rand(num_stimuli, num_time, 3) if two_eye else None,
        'translation': torch.rand(num_stimuli, num_time, 3) if two_eye else None,
        'num_eyes': num_eyes,
        'eye_offset': eye_offset,
        'model_dt': model_dt,
    }
    if train_fraction is not None:
        payload['train_fraction'] = train_fraction
    torch.save(payload, folder / name)


def test_load_processed_stimulus(tmp_path):
    training = tmp_path / "stimulus" / "training"
    _write_baked(training, "a_train.pt", 2, EYE_OFFSET)
    _write_baked(training, "b_train.pt", 2, EYE_OFFSET, num_stimuli=3)
    _write_baked(training, "a_test.pt", 2, EYE_OFFSET, num_stimuli=1)
    _write_baked(training, "one_eye_train.pt", 1, None)
    _write_baked(training, "slow_train.pt", 2, EYE_OFFSET, model_dt=0.02)
    # Baked train-only (create_training_data writes no _test.pt).
    _write_baked(training, "full_train.pt", 2, EYE_OFFSET, train_fraction=1.0)
    config = {
        'model': {'device': 'cpu', 'dtype': 'float32', 'dt': 0.01},
        'stimulus': {'stimulus_type': ['a', 'b']},
        'data': {'data_folder': str(tmp_path)},
    }

    data = tu.load_processed_stimulus(config)
    assert data['stimulus'].shape == (5, 3, 10)
    assert data['stimulus'].dtype == torch.float32
    assert data['optic_flow'].shape == (5, 3, 10, 2)
    assert data['rotation'].shape == data['translation'].shape == (5, 3, 3)
    assert data['num_eyes'] == 2
    assert data['eye_offset'] == EYE_OFFSET

    config['stimulus']['stimulus_type'] = 'a'
    assert tu.load_processed_stimulus(config, split='test')['stimulus'].shape[0] == 1

    config['stimulus']['stimulus_type'] = 'one_eye'
    data = tu.load_processed_stimulus(config)
    assert data['num_eyes'] == 1 and data['eye_offset'] is None
    assert data['rotation'] is None and data['translation'] is None

    config['stimulus']['stimulus_type'] = ['a', 'one_eye']
    with pytest.raises(ValueError, match="Mixed eye counts"):
        tu.load_processed_stimulus(config)
    config['stimulus']['stimulus_type'] = 'slow'
    with pytest.raises(ValueError, match="baked at dt"):
        tu.load_processed_stimulus(config)
    config['stimulus']['stimulus_type'] = 'b'
    with pytest.raises(FileNotFoundError, match="create_training_data"):
        tu.load_processed_stimulus(config, split='test')
    with pytest.raises(ValueError, match="split"):
        tu.load_processed_stimulus(config, split='validation')

    # A train-only type is skipped in the test split, unless it is the only one.
    config['stimulus']['stimulus_type'] = ['a', 'full']
    assert tu.load_processed_stimulus(config, split='test')['stimulus'].shape[0] == 1
    assert tu.load_processed_stimulus(config)['stimulus'].shape[0] == 4
    config['stimulus']['stimulus_type'] = 'full'
    with pytest.raises(FileNotFoundError, match="train-only"):
        tu.load_processed_stimulus(config, split='test')


@pytest.mark.parametrize("dynamics_type", [
    "CurrentDynamics", "CurrentDivisiveDynamics", "ConductanceDynamics",
    "CellAnnDynamics", "ArbitraryDynamics",
])
def test_every_dynamics_lifecycle(tmp_path, dynamics_type):
    """Build, train a step, save/load, silence (and restore), and trim."""
    from drosophila_model.model_definition import neural_models as nm

    model, _ = build_model_from_test_config(readouts=['FlowConv2Readout'],
                                            dynamics_type=dynamics_type)
    _, _, changed = _train_one_step(model, 2, ['flow_loss'], eye_offset=EYE_OFFSET)
    assert changed

    path = tmp_path / "model.pt"
    model.save(path)
    loaded = nm.BaseRnn.load(path, device='cpu')
    for (name, p), (_, q) in zip(model.state_dict().items(), loaded.state_dict().items()):
        assert torch.equal(p.cpu(), q.cpu()), name

    before = {k: v.clone() for k, v in model.state_dict().items()}
    with model.silenced(['Mi1'], silence_inputs=True):
        assert any(not torch.equal(before[k], v) for k, v in model.state_dict().items())
    assert all(torch.equal(before[k], v) for k, v in model.state_dict().items())

    trimmed = model.remove_cell_types(['Mi1'])
    assert 'Mi1' not in trimmed.unique_cell_types
    inputs, _, _ = _dataset(trimmed, 1)
    assert torch.isfinite(trimmed.predict(inputs)['outputs']['optic_lobe']['FlowConv2Readout']).any()
