"""2a acceptance test: a YAML config must resolve to the EXACT same namespace as the
equivalent CLI command, and explicit CLI flags must beat YAML must beat defaults.

Importing train pulls torch/cv2/timm, so this runs where those exist (the GPU box, or
any full env) -- not on the laptop. No framework: `python tests/test_resolve_args.py`.
"""
import os, sys, tempfile

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src"))
from train import resolve_args, validate_cfg, build_parser

REPO = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")

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
    a = resolve_args(CLI_B)
    b = resolve_args(["--config", os.path.join(REPO, "configs", "bbabl_cv5_B_s42.yaml")])
    diff = {k: (ns_dict(a)[k], ns_dict(b)[k]) for k in ns_dict(a) if ns_dict(a)[k] != ns_dict(b)[k]}
    assert not diff, f"YAML vs CLI mismatch: {diff}"


def test_precedence():
    cfg = os.path.join(REPO, "configs", "bbabl_cv5_B_s42.yaml")
    args = resolve_args(["--config", cfg, "--epochs", "1"])
    assert args.epochs == 1          # explicit CLI beats YAML
    assert args.bs == 6              # YAML beats default (24)
    assert args.lr_head == 1e-3      # default survives when neither sets it


def test_validation():
    def rejects(yaml_text, why):
        with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
            f.write(yaml_text)
        try:
            resolve_args(["--config", f.name])
            raise AssertionError(f"accepted bad config ({why}): {yaml_text!r}")
        except SystemExit as e:
            assert "--config" in str(e), (why, str(e))
        finally:
            os.unlink(f.name)

    rejects("lora_rr: 8", "unknown key (typo)")
    rejects("- a\n- b", "top-level list")
    rejects("epochs: five", "uncoercible type")
    rejects("full_data: 1", "non-bool for store_true")
    rejects("select_on: leaderboard", "value not in choices")
    rejects("config: other.yaml", "config-in-config")

    with tempfile.NamedTemporaryFile("w", suffix=".yaml", delete=False) as f:
        f.write("epochs: '5'\nfull_data: yes\n")     # coercion + yaml bool
    args = resolve_args(["--config", f.name])
    os.unlink(f.name)
    assert args.epochs == 5 and args.full_data is True

    assert validate_cfg({}, build_parser()) == {}    # empty file -> no overrides


if __name__ == "__main__":
    test_yaml_equals_cli()
    test_precedence()
    test_validation()
    print("2a acceptance: all checks passed")
