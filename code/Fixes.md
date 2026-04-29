Train.py fixes

Fixes applied (cumulative):
  [FIX-1]  dataset.py: augmentation no longer zeros neighbor slices
  [FIX-2]  calc_stats runs on CLAHE-processed images
  [FIX-3]  Dice loss smooth: 1.0 → 1e-6  (restores gradient signal)
  [FIX-4]  Differential LR: encoder=3e-5, decoder/head=3e-4
  [FIX-5]  calcStats.json stale-stats warning on startup
  [FIX-6]  Dice threshold: 0.5 → 0.3 in compute_dice_safe during validation
  [FIX-7]  Hausdorff Distance actually computed in validate()
  [FIX-8]  Small bowel class weight boosted x1.5
  [FIX-9]  Sampler generator reassignment removed
  [FIX-10] GradScaler state saved + restored in crash checkpoint
  [FIX-11] hausdorff_distance_2d returns None for both-empty pairs
  [FIX-12] Dice and HD evaluated at the SAME threshold
  [FIX-13] HD computed every HD_EVAL_EVERY epochs on HD_EVAL_BATCHES batches
  [FIX-14] save_mask_overlay wraps body in try/finally
  [FIX-15] Threshold sweep: after every epoch
  [FIX-16] validate(): logits_cache capped at SWEEP_CACHE_BATCHES
  [FIX-17] train_one_epoch(): AMP branch unpacks _forward() into loss
  [FIX-18] augmentations.py: alpha_affine removed, shift_limit split
  [FIX-19] Best-model checkpoint saved on composite score when HD available
  [FIX-20] valid_pairs diagnostic reads pre-computed counts
  [FIX-21] ReduceLROnPlateau added as secondary scheduler
  [FIX-22] get_dataloaders() pin_memory threaded through from cfg
  [FIX-23] Encoder LR raised from 3e-5 → 1e-4; LR ratio reduced 10x → 3x
  [FIX-24] SequentialLR + ReduceLROnPlateau conflict resolved
  [FIX-25] dataset.py: WeightedRandomSampler weight 5.0 → 2.0
  [FIX-26] loss.py: smooth=1.0 in metric (not loss)
  [FIX-27] model.py: Dropout2d(0.2) before segmentation head
  [FIX-28] (same as FIX-24, cross-reference)
  [FIX-29] best_score comparison: dice_mean is always primary metric
  [FIX-30] Auxiliary presence detection head added to model.py
  [FIX-31] model.py: presence_head operates on detached encoder features
  [FIX-32] loss.py: Dice smooth raised 1e-6 → 1e-4
  [FIX-33] Plateau scheduler monitors negated dice_mean, not val_loss
  [FIX-34] PRESENCE_LOSS_WEIGHT annealed 0.0 → 0.3 over first 10 epochs
  [FIX-35] Gradient norm logged to TensorBoard every update step
  [FIX-36] Early stopping with best-weight restoration
  [FIX-37] Threshold sweep lower bound raised 0.20 → 0.25
  [FIX-38] Gradient accumulation boundary condition fixed
  [FIX-39] LR reduced: encoder 1e-4 → 5e-5, decoder ratio 3x → 2x
  [FIX-40] Plateau patience 7 → 3, factor 0.5 → 0.4
  [FIX-41] Weight decay raised 1e-5 → 1e-3
  [FIX-42] Early stop patience 15 → 20, min_epochs guard added
  [FIX-43] EMA smoothing on val dice before feeding plateau scheduler
  [FIX-44] Encoder frozen for first FREEZE_ENCODER_EPOCHS epochs
  [FIX-45] Early-stop counter and best_dice reset after encoder unfreeze
  [FIX-46] EMA state reset at encoder unfreeze boundary
  [FIX-47] Plateau scheduler not stepped during encoder-frozen epochs
  [FIX-48] Early stop patience raised 20 → 35, min_epochs raised 15 → 25
  [FIX-49] Plateau patience raised 3 → 8, factor 0.4 → 0.5
  [FIX-50] EMA alpha raised 0.3 → 0.6
  [FIX-51] min_lr raised 1e-7 → 5e-6
  [FIX-52] Boundary (soft-HD) loss added
  [FIX-53] Threshold sweep range extended 0.10–0.85
  [FIX-54] PRESENCE_LOSS_WEIGHT reduced 0.3 → 0.05
  [FIX-55] Stomach class weight boosted x2.0
  [FIX-56] --reset_lr flag added
  [FIX-57] PRESENCE_LOSS_WEIGHT set to 0.0 — disabled entirely
  [FIX-58] Boundary loss gated on Dice > BOUNDARY_LOSS_DICE_GATE
  [FIX-59] Val loss = seg loss only
  [FIX-60] WeightedPerClassCombinedLoss Dice fix: per-class independent
  [FIX-61] Boundary loss REMOVED from _forward() entirely.
           Root cause of epoch-9 collapse: boundary loss gated on
           dice>0.10, but dice hit 0.15 at epoch 1 (lucky init), so
           boundary loss activated immediately on an untrained model
           whose Laplacian edges are near-random noise. The boundary
           F1 loss then produced large gradients that fought the seg
           loss, causing the train_loss spike at epoch 9 (0.66→0.80)
           and the subsequent dice crash to 0.04. Even with a higher
           gate threshold the gate is not reliable because early dice
           is noisy — a single good epoch opens the gate permanently.
           Boundary loss removed entirely from the training forward
           pass. It will be re-introduced only after the model
           achieves stable dice > 0.25 for 5+ consecutive epochs,
           at which point the Laplacian edges are meaningful.
  [FIX-62] Threshold sweep restored to [0.25, 0.30, ..., 0.50].
           The extended sweep to 0.10 was masking model collapse:
           at threshold=0.10 almost every pixel is predicted positive,
           giving inflated dice on positive slices and hiding the true
           calibration. When the model collapses to predicting near-
           uniform low probabilities (epoch 17-19, dice<0.003), the
           0.10 threshold picks up noise as signal. The narrow sweep
           forces the metric to reflect genuine confidence.
  [FIX-63] Plateau patience raised 8 → 12, factor kept at 0.5.
           With boundary loss removed and PRESENCE disabled, the only
           loss signal is clean BCE+Dice. The EMA-smoothed plateau
           signal is now more stable, so patience=8 was reducing LR
           too quickly during the post-collapse recovery phases where
           dice oscillates for 8-10 epochs before finding a new basin.
           patience=12 gives the model two full oscillation cycles
           before triggering a reduction.
  [FIX-64] Stomach class weight boost REMOVED (back to x1.0 after
           the x2.0 boost from FIX-55).
           With only BCE+Dice and no presence/boundary auxiliary
           losses, the x2.0 stomach weight was over-concentrating
           gradients on stomach during the critical early epochs where
           large_bowel (the easiest class) provides the stabilising
           gradient signal the model needs first. Large bowel dice
           was 0.001 at epoch 1 in the collapsed runs — the stomach
           boost was suppressing it. Restored to x1.0; only small
           bowel gets the x1.5 boost.
  [FIX-65] Encoder freeze extended: FREEZE_ENCODER_EPOCHS 3 → 5.
           With only 3 frozen epochs, the encoder unfreezes during
           the warmup phase (warmup_epochs=5), meaning the warmup
           LR ramp and the encoder unfreeze happen simultaneously.
           The combined effect is a sudden large LR on a freshly
           unfrozen encoder, producing the large gradient spikes
           seen at epoch 4-9. With FREEZE_ENCODER_EPOCHS=5, the
           encoder unfreezes exactly when warmup ends and the LR
           stabilises at its target value, giving a clean handoff.

Model.py fixes


Differential LR (FIX-4) is applied in train.py by accessing:
    model.model.encoder           → lr = 1e-4   [FIX-23]
    model.model.decoder           → lr = 3e-4   (3× encoder, not 10×)
    model.model.segmentation_head → lr = 3e-4

FIXES applied:
  [FIX-27] Dropout(0.2) injected before the segmentation head.
  [FIX-30] Auxiliary presence detection head added.
  [FIX-31] presence_head now operates on DETACHED encoder features.
           Previously presence gradients flowed back through the full
           encoder (2048-channel deepest feature map), contributing ~30%
           of total encoder gradient signal and actively competing with
           segmentation gradients during the critical first 10–15 epochs
           when encoder features are forming. This caused the encoder to
           oscillate between "segment" and "classify presence" objectives,
           producing the loss spikes and dice collapses observed at epochs
           14, 19, 21, 29. Fix: features[-1].detach() stops all presence
           gradients at the encoder boundary. The presence head still
           trains (its own parameters receive gradients), but it no longer
           corrupts encoder feature learning.

loss.py fixes
FIXES applied:
  [FIX-3]  DiceLoss smooth: 1.0 → 1e-6. Near-zero preds now yield loss≈1.0
           instead of ≈0.0, restoring Dice gradient signal early in training.
  [FIX-7]  hausdorff_distance_2d exported so train.py validate() can import
           and call it directly per-item per-class.
  [FIX-11] hausdorff_distance_2d returns None (not 0.0) when both masks are
           empty, so callers skip the pair from the HD mean.
  [FIX-26] compute_metrics uses smooth=1.0 in the METRIC (not the loss) so
           small-structure classes don't report a misleadingly near-zero Dice.
  [FIX-32] DiceLoss smooth raised from 1e-6 → 1e-4.
           With 57% empty slices, smooth=1e-6 caused loss→0 on both-empty
           pairs, making "predict nothing" a local optimum. 1e-4 keeps a
           gradient signal flowing without materially affecting non-empty
           predictions. The metric still uses smooth=1.0 (FIX-26).

dataset.py fixes
FIXES applied:
  [FIX-1]  Augmentation no longer zeros neighbor slices.
  [FIX-2]  calc_stats runs on CLAHE-processed images.
  [FIX-5]  /tmp/ split CSVs are now named with the n_slices + img_size
           suffix so parallel runs (n_slices=3 vs 5, img_size=320 vs 448)
           no longer overwrite each other's val split mid-epoch.
  [FIX-7]  Augmentation now passes a 2-D binary union mask as `mask` (for
           MaskAwareRandomCrop anchor sampling) and the full (H,W,C) array
           as `multichan` via additional_targets.
  [FIX-22] get_dataloaders() now accepts a `pin_memory` parameter and passes
           it through to both DataLoaders.
  [FIX-25] WeightedRandomSampler positive weight reduced from 5.0 → 2.0.
  [FIX-30] WeightedRandomSampler removed entirely.
           The 2:1 positive oversampling (FIX-25) still caused a systematic
           train/val calibration mismatch: the model trained on ~67% positive
           slices but validated on ~43% (natural distribution, 57% empty GT).
           This was the primary cause of the optimal threshold oscillating
           between 0.35 and 0.50 epoch-to-epoch and empty_pred% swinging
           26%→50%: the model's sigmoid outputs were calibrated for the wrong
           prior at validation time. Loss class weights (computed per pixel
           frequency in train.py) already compensate for class imbalance at
           the gradient level; the sampler was redundant with those weights
           and actively harmful to calibration. Replaced with shuffle=True.

augmentations.py fixes
FIXES applied:
  [FIX-7]  MaskAwareRandomCrop anchor sampling now uses a 2-D binary union
           mask (max across channels) instead of the raw multi-channel array.
           Previously dataset.py passed mask.transpose(1,2,0) → (H,W,3) into
           the augment() call, and Albumentations forwarded that full 3-D array
           to get_params_dependent_on_targets(). np.argwhere on an (H,W,3)
           array returns (row, col, channel) triples, so the crop anchor was
           sampled from nonsense coordinates, silently falling back to random
           cropping on almost every sample. Fix: dataset.py now passes a
           pre-computed 2-D union_mask via additional_targets and the crop
           class reads that clean 2-D array for anchor sampling.

  [FIX-8]  GaussNoise: removed the deprecated `var_limit` kwarg; replaced
           with `std_range=(0.04, 0.22)` which is the correct API for
           Albumentations >= 1.4 and silences the UserWarning that was
           printed at the start of every training run.

  [FIX-18] ElasticTransform: removed `alpha_affine=3` kwarg which was dropped
           in albumentations >= 1.4 and either silently ignored or raised a
           TypeError depending on the exact version installed.
           OpticalDistortion: replaced `shift_limit=0.1` with
           `shift_limit_x=0.1, shift_limit_y=0.1` per the new API introduced
           in albumentations >= 1.4. The old kwarg was silently ignored,
           meaning optical distortion ran without any shift, producing weaker
           augmentation than intended.

  [FIX-33] ShiftScaleRotate replaced with A.Affine.
           ShiftScaleRotate is now a restricted alias of Affine in
           albumentations >= 1.4 and emits a UserWarning on every import.
           A.Affine is the canonical replacement and accepts the same
           logical parameters (translate_percent, scale, rotate) with a
           cleaner API. `value` and `mask_value` are not valid kwargs on
           Affine — border fill is controlled via `cval` (image) and
           `cval_mask` (mask).

  [FIX-34] ElasticTransform, GridDistortion, OpticalDistortion: removed
           `value` and `mask_value` kwargs. These were silently dropped in
           albumentations >= 1.4; the correct kwargs are `fill` (image fill
           value) and `fill_mask` (mask fill value). Updated all three
           transforms accordingly.

Pipeline is applied to the CENTER slice + all masks inside dataset.py.
Neighbor slices are spatially resized to match augmented dims but do not
receive intensity augmentations (FIX-1 — unchanged from previous version).