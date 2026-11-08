# Historical protected-year test reads

Before the explicit real-data test gate, plain local test discovery could run
two cache-dependent checks: published practice CDD errors over 2011-2018, and
byte hashing of real suite files including reserved assessment artifacts.
The Opus round-5 reviewer disclosed executing these checks; earlier local full
sweeps also ran cache-dependent tests. There is no complete historical log of
these reads, so their count cannot be certified as zero.

These are pre-2019 training-era observations. No observed 2019+ weather was
obtained, no private official answer key was accessed, and no challenge entry
was submitted. Published reference-number checks validated scoring arithmetic;
their results were not used here to choose a forecasting model. The broader
historical years were already disclosed as reused evidence.

The old manifest field `real_assessment_access_count: 0` counted only evaluator
assessment-phase ledger events. It did not cover raw-cache tests, manual opens
or other label reads. The manifest now calls it `assessment_phase_access_count`
and links this disclosure. Existing sealed run records are preserved.

Real-data test classes now require explicit `PROTOSTAR_RUN_REAL_DATA_TESTS=1`.
Default discovery and CI leave it unset and skip them, including on machines
with the pinned cache. Opting in does not create untouched evidence or authorize
selecting models against reserved assessment results. Future audit briefs must
inspect test scope before any real-data opt-in.
