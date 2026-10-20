# On-device isolated ACFQ staging (not installation)

Do NOT use this until the Android build passes and an experimental
artifact has been downloaded from GitHub Actions. A successful shared-
library build alone does not prove integration or acceleration.

Unpack the experimental Android ARM64 artifact and run:

  cd ~/jaco-qwant-llm
  git pull --ff-only origin main
  python3 native/stage_litert_lm_arm64.py --artifact-dir /path/to/unpacked --check
  python3 native/stage_litert_lm_arm64.py --artifact-dir /path/to/unpacked --stage

This creates only ~/jaco-acfq-stage-0171/litert_lm/, using a private
copy of the installed Python wheel plus the experimental shared library.
It refuses existing target folders, unexpected versions, bad ELF metadata,
and a SHA256 mismatch. It never overwrites the installed package.

Optional library-load test (NO Gemma model inference):

  PYTHONPATH=$HOME/jaco-acfq-stage-0171 python3 -c 'from litert_lm import _ffi; print(_ffi._get_lib()._name)'

Stop if the native loader fails. Do not launch an experimental full model
beside a production model until device memory use has been assessed.
Enabling JACO_ACFQ_SHADOW=1 runs diagnostics but the original operator
still computes the output; this code does not yet accelerate production.
