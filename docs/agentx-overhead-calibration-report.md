# Step-overhead calibration audit

The B200 and B300 device definitions both provide `step_overhead_us = 0.0`
with grade-D analytical provenance. The final native checkpoints also report
exactly zero `overhead` component seconds after warmup; the component totals
conserve with DP local compute time.

Adding a positive framework overhead cannot explain the observed absolute
error: the native simulator is already slower than the reference by 85.49% on
B200 and 83.84% on B300. A positive fitted overhead would lower throughput
further. The B200-to-B300 comparison therefore freezes the only admissible
overhead parameter at zero and attributes the remaining error to the measured
attention/operator provenance documented in the native attention report. No
SKU-specific correction is introduced.

Evidence:

- `data/devices/b200_sxm.yaml` and `data/devices/b300_sxm.yaml` both set the
  sourced step overhead to zero.
- `results/agentx-dsv4-full/native-checkpoints-final/summary.csv` contains the
  three-seed checkpoint comparison.
- Every checkpoint result has `operator_component_seconds.overhead == 0` and
  passes component/DP time conservation.
