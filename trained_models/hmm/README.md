# HMM decoder

The decoder consumes route-free candidates, model probabilities, a projected `pathsnap_prep.Trip`, projected GPS coordinates and timestamps. It does not accept a reference route. From the repository root import `decode_trip` with `from trained_models.hmm.hmm_decoder import decode_trip`. Pass a DataFrame containing `gps_row` and local `arc_id`, a row-aligned probability vector, the trip, `trip.gps_xy`, `trip.gps_t`, and the parsed JSON config. It returns one local arc per GPS point, the collapsed arc sequence and diagnostics. The initial point must have candidates; subsequent empty points carry the chosen previous arc.

The current settings are preliminary, not learned. Training data and out-of-fold probabilities must be used to select settings; the student partition is reserved for the final comparison. Run `make_oof.py` with the repository's Python dependencies installed to generate seed-42, trip-grouped out-of-fold probabilities. Run `smoke_train.py` only as a functional check: it uses the fitted model on its training data and is not an accuracy estimate.

Configuration keys:

* `model_path`: repository-relative untuned model path. To swap a trained model, change this path in the scoring code/config; recheck `alpha` on out-of-fold probabilities because calibration may change.
* `seed`: reproducibility seed for model training and parameter search.
* `alpha`, `eps`: emission weight and probability floor.
* `beta`: distance discrepancy scale in metres; `gap_beta_factor` multiplies it after a gap.
* `u_turn_penalty`: log-score penalty for a reverse-direction twin.
* `gap_seconds`: threshold for relaxing a gap; `v_max_mps` and elapsed time set the directed search cutoff, with `min_cutoff_m` as the minimum.
* `hmm_top_k`: maximum retained candidates per GPS point.
* `backward_tolerance_m`: small same-arc backward projection movement treated as stationary.
* `unreachable_logp`: score assigned to unreachable pairs when another pair is reachable; when every pair is unreachable, decoding restarts.
* `cutoff_bucket_m`, `cache_size`: search cutoff rounding and maximum cached source/bucket searches. Bucket rounding is upward; the actual route cutoff is still checked.
