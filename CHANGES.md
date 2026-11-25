# pyBASS (development)
* new `bassGibbs`: BASS as one block of a larger Gibbs sampler, advanced a few steps at a time with a response that changes between calls (`set_response`, `step`, `fitted`, `s2`, `save`, `model`, `accept_rate`), with options for no intercept, basis functions centred over the training inputs, and a fixed error variance (as `bassGibbs` in bass_matlab).
* `BassState.update` is now `rj_step` followed by `gibbs_step`; `bass()` draws the same random numbers as before.

# pyBASS 0.3.1.9999
* added sobol for Basis

# pyBASS 0.3.1

# pyBASS 0.3.2
* initial version of package
