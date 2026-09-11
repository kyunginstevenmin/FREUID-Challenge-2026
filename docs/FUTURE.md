# Future studies (parked — not pre-registered, no runs planned)

Entries here are deliberately out of scope for the current two-phase ablation
program (backbone → head). Each records the question, why it is a separate
study rather than an arm of an existing one, and what evidence would promote it
to a pre-registered protocol.

## Simpler architectures (CNN baseline)

**Question.** Does fraud detection in this regime need a pretrained ViT at all,
or can a CNN (e.g. ConvNeXt-T / EfficientNet) trained under a comparable data
recipe reach tier-1 non-inferiority at a fraction of the cost?

**Why it is not an arm of the backbone ablation.** The cv5 recipe is not
backbone-agnostic: it assumes a frozen DINOv2 ViT (LoRA adapters target
attention projections; the per-patch MIL head consumes a token grid). A CNN has
neither, so "same recipe, swap backbone" does not exist — fine-tuning strategy,
head, and hyperparameters would all change, making it a new study with its own
protocol, not a controlled arm.

**Promotion trigger.** Revisit after the backbone ablation's capacity curve is
in: if ViT-S holds near ViT-L (capacity barely matters in-regime), a cheap CNN
becomes plausible and worth a protocol; if even ViT-B loses clearly, this entry
is likely dead and should be marked so with a dated note.

**Licensing note.** ConvNeXt V2 weights were deliberately avoided in the
competition solution for license reasons (see README "Licenses") — any CNN
study must re-check backbone weight licenses first.
