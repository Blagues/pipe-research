"""SHARP amplitude-Doppler CNN baseline, adapted to run on the same data
partition (train/val/holdout split + zero-shot test set) as the contrastive
pose model in ``paper-code``.

Method summary (honest labelling for the paper):
  * Supervised end-to-end Doppler CNN ("SHARP-style"), labels only, no pose.
  * Amplitude-only Doppler (csi_abs). No phase sanitization: the upstream
    phase-sanitization step lives only in the TensorFlow scripts we do not use,
    and the raw csi_phase in this dataset is unsanitised (wrapped, ~uniform).
  * All 3 receivers x 4 antennas = 12 streams, late-fused by summed softmax.
  * 2 s Doppler windows, 0.5 s stride, one prediction per labelled segment.

The Doppler transform and CNN are vendored verbatim from the provided
"SHARP impl" so this baseline is self-contained and will not drift if that
project changes. See ``doppler.py`` and ``model.py`` for provenance notes.
"""
