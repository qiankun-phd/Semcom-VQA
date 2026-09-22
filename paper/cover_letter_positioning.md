# TGCN cover letter — positioning paragraph (3 sentences)

This manuscript extends the green semantic-communication line recently
developed in IEEE TGCN — energy-efficient aerial semantic transmission
(Zheng et al., 2024), joint communication-plus-computation energy
minimization for probabilistic semantic communication (Dai et al., 2025),
and green RSMA-enabled semantic networks (Xu et al., 2025) — from
optimizing the energy of a fixed semantic representation toward measuring
which representation a task actually needs. To our knowledge it is the
first work on that line to evaluate a real, frozen vision--language-model
receiver by measured end-to-end task accuracy over coded fading links
(548+490 aerial images, five question types, three channels, eight
methods) and to report the resulting per-answer joint
communication-plus-computation energy frontier, with GPU power measured
during the campaign workload itself. The headline finding is directly
actionable for green network design: the frontier is compute-dominated,
so question-conditioned evidence routing — which decides per query
whether the VLM forward pass runs at all — cuts measured joules per
answered question by 2.2x against rate-adaptive image transmission at
every SNR while raising accuracy, a lever that composes with, rather than
competes against, the power/bandwidth allocators of the existing green
semantic-communication literature.
