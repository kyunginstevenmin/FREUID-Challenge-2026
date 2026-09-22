"""2a acceptance tests (`pytest tests/`): a YAML config must resolve to the EXACT
same namespace as the equivalent CLI command; explicit CLI beats YAML beats defaults;
bad configs are rejected loudly. Importing train pulls torch/cv2/timm -- runs where
the full (CPU is fine) stack exists.
"""
import os

import pytest

from train import resolve_args, validate_cfg, build_parser, LEGACY_DATA_FLAGS

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
CFG_B = os.path.join(REPO, "configs", "bbabl_cv5_B_s42.yaml")
CFG_OWN = os.path.join(REPO, "configs", "own_v2_ctrl_s42.yaml")

# The own-run plan's step-3 control arm, as flags:
CLI_OWN = ("--split_dir ../../id-fraud-detection/dataset_final --epochs 5 --bs 6 --accum 4 "
           "--eval_bs 32 --res 448x728 --aug core --sbi 0.0 --backbone vit_large_patch14_reg4_dinov2 "
           "--head_type patch --lora_r 16 --select_on genval --workers 16 --save_every 250 "
           "--wandb --wandb_project freuid-own --seed 42 --tag own_v2_ctrl_s42 "
           "--compile --fused_opt").split()   # speed knobs adopted at 3c (2026-09-22)

# The BACKBONE_ABLATION.md arm-B command, as flags:
CLI_B = ("--full_data --epochs 5 --bs 6 --accum 4 --eval_bs 8 --res 448x728 "
         "--sbi 0.25 --attacks full "
         "--idnet_countries ESP_scanned,ALB_scanned,AZE_scanned,FIN_scanned,GRC_scanned,"
         "LVA_scanned,RUS_scanned,SRB_scanned,EST_scanned,SVK_scanned "
         "--heldout_idnet= --idn_val_from_unused --lim_idn 80000 --idn_val_n 4000 "
         "--select_on idnet --workers 16 --save_every 250 --head_type patch --seed 42 "
         "--backbone vit_base_patch14_reg4_dinov2 --tag bbabl_cv5_B_s42").split()


def ns_dict(ns):
    return {k: v for k, v in vars(ns).items() if k != "config"}


def test_yaml_equals_cli():
    a, b = resolve_args(CLI_B), resolve_args(["--config", CFG_B])
    assert ns_dict(a) == ns_dict(b)


def test_precedence():
    args = resolve_args(["--config", CFG_B, "--epochs", "1"])
    assert args.epochs == 1          # explicit CLI beats YAML
    assert args.bs == 6              # YAML beats default (24)
    assert args.lr_head == 1e-3      # default survives when neither sets it


@pytest.mark.parametrize("yaml_text", [
    "lora_rr: 8",                    # unknown key (typo)
    "- a\n- b",                      # top-level list
    "epochs: five",                  # uncoercible type
    "full_data: 1",                  # non-bool for store_true
    "select_on: leaderboard",        # value not in choices
    "config: other.yaml",            # config-in-config
], ids=["typo-key", "not-a-mapping", "bad-type", "int-for-bool", "bad-choice", "nested-config"])
def test_bad_configs_rejected(tmp_path, yaml_text):
    p = tmp_path / "bad.yaml"
    p.write_text(yaml_text)
    with pytest.raises(SystemExit, match="--config"):
        resolve_args(["--config", str(p)])


def test_coercion_and_yaml_bools(tmp_path):
    p = tmp_path / "ok.yaml"
    p.write_text("epochs: '5'\nfull_data: yes\n")     # str->int coercion + yaml bool
    args = resolve_args(["--config", str(p)])
    assert args.epochs == 5 and args.full_data is True


def test_empty_config_is_noop():
    assert validate_cfg({}, build_parser()) == {}


def test_own_ctrl_yaml_equals_cli():
    a, b = resolve_args(CLI_OWN), resolve_args(["--config", CFG_OWN])
    assert ns_dict(a) == ns_dict(b)


def test_own_ctrl_is_the_plain_recipe():
    """Control arm = split_dir mode with no legacy data flags and no speed knobs: the
    step-3c decision adds compile/fused_opt to the YAML explicitly, never by accident."""
    args = resolve_args(["--config", CFG_OWN])
    defaults = vars(build_parser().parse_args([]))
    assert args.split_dir and args.select_on == "genval"
    assert all(getattr(args, k) == defaults[k] for k in LEGACY_DATA_FLAGS)
    assert (args.precision, args.grad_ckpt) == ("fp16", False)
    if args.compile or args.fused_opt:          # allowed only once 3c has decided, on record
        assert "3c" in open(CFG_OWN).read()
