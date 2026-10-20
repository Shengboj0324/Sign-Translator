"""Grammar measured on continuous motion: the spatial and non-manual layer of ASL.

The symbolic layers (``planning`` typed plans, ``grammar`` SIR, ``facial_nmm``
intervals) describe what a sentence's grammar *is*; this package measures it in
canonical motion states and turns it into conditioning, losses and metrics
(docs/GRAMMAR_MOTION.md, reports/ASL grammar and spatial reference.md):

* ``frames``     — chest-centred body frame, azimuth/elevation, dominant hand;
* ``pointing``   — index-point detection, pointing rays, least-squares loci and
                   DP-vMF-means locus clustering;
* ``nmm``        — brow raise / furrow / squint / mouth-morpheme tracks from
                   blendshapes, headshake and nod from head rotation, spans;
* ``role_shift`` — torso/head/gaze change points (exact PELT) and shifted spans;
* ``gaze_locus`` — gaze-to-locus angles and agreement-window gaze tests;
* ``agreement``  — verb–locus direction/endpoint/facing errors and their loss;
* ``prosody``    — movement units, holds, blinks and phrase-boundary evidence;
* ``text``       — grammatical features of the English sentence;
* ``tracks``     — all of the above for one state, serialisable.
"""
