# GAP-01: isolated BPG true-bitstream Mode2 benchmark

This experiment adds a conventional BPG baseline on the completed JPEG2400
receiver/PHY stack. It does not register a production codec/policy or edit the
scanner. Commit only on `exp/bpg-phy-benchmark-n2400`.

`budgeted_bpg.py` uses native [libbpg0.9.8](https://bellard.org/bpg/) encode/decode:
RGB PNG input, x265,8bit,420/YCbCr,level1; descending maximum side
384/320/256/224/192/160/128/96/64, ascending QP20/25/30/35/40/45/50/51.
Each cap independently chooses the first fitting candidate, sharing identical
encode attempts. One isolated byte `B` precedes native BPG bytes.

`bpg_bitstream_mode2_full_grid.py` inherits JPEG's actual-bit PHY and receiver
prediction functions verbatim. It requires completed JPEG2400 true-PHY data to
reuse the learned baseline and refuses missing baseline outcomes. Set
`VQA_SEMCOM_ROOT`; legacy absolute paths in manifests are rebased in memory.
The existing live scanner must pass exact Rule B/default AST checks first.
The repository's older scanner snapshot is deliberately unchanged.

Mode2: caps2k/4k,medium; SNR0/5/10/20; seeds1701–1703; actual
Ns=510*ceil(framed_bytes/48), gamma_eff=gamma+10log10(21420/Ns) once.
Rule B rows compose the fixed2k rows at≤10 and fixed4k rows above10.
Whole decoded bytes must exactly equal the sent packet. No analytical cutoff,
historical noise proxy, training or deferred-gap work.

Resume uses an output lock and immutable source/input/binary fingerprints,
per-image encoding journal, clean prediction journal and real-PHY journal.
Encoding/prediction/channel work precedes sealed-label scoring. Interrupted
final journal writes can be repaired; interior corruption fails. Unique physical
tasks may be fewer than logical grid evaluations for byte-identical packets.

See `COLAB_LAUNCH.md` for provisioning after user authentication and asset gates.
Codec fixture smoke is not a semantic/PHY benchmark. `validate_results.py`
checks completed real tables, symbol normalization and compositional Rule B.
No full-grid result exists at local preparation time.
