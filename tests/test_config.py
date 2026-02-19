from llm_trainer.config import merge_config


def test_merge_config_cli_overrides_yaml():
    yaml_values = {"epochs": 1, "batch_size": 4, "txt": ["data/a.txt"]}
    cli = {"epochs": 3, "batch_size": None, "txt": None, "hf": None}
    cfg = merge_config(cli, yaml_values)
    assert cfg.epochs == 3
    assert cfg.batch_size == 4
    assert cfg.txt == ["data/a.txt"]


def test_merge_config_requires_dataset_by_default():
    try:
        merge_config({}, {})
        assert False, "Expected ValueError"
    except ValueError as exc:
        assert "At least one dataset" in str(exc)


def test_merge_config_can_skip_dataset_requirement():
    cfg = merge_config({}, {}, require_dataset=False)
    assert cfg.epochs == 1


def test_merge_config_coerces_numeric_yaml_values():
    yaml_values = {
        "txt": ["data/a.txt"],
        "rms_norm_eps": "1e-06",
        "rope_theta": "10000.0",
        "epochs": "2",
    }
    cfg = merge_config({}, yaml_values)
    assert isinstance(cfg.rms_norm_eps, float)
    assert isinstance(cfg.rope_theta, float)
    assert isinstance(cfg.epochs, int)
