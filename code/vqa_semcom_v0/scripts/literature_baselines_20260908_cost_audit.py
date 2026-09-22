#!/usr/bin/env python3
"""Audit recorded payloads without inferring radio joules or coded symbol counts."""
import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
import statistics


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--repo', type=Path, required=True)
    parser.add_argument('--root', type=Path, required=True)
    args = parser.parse_args()
    payloads = defaultdict(set)
    sources = {}
    for name in ['main', 'cmp', 'extra']:
        path = args.repo / 'outputs/vlm' / f'v25_rician_{name}_predictions.csv'
        sources[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        with path.open(newline='') as handle:
            for row in csv.DictReader(handle):
                if row['service_level'] != '2':
                    continue
                key = (row['image_id'], row['question'], float(row['sensed_snr_db']))
                payloads[key].add(int(float(row['payload_bytes'])))
    result = {'source_sha256': sources, 'by_split': {},
              'rsvqa': 'Same cached received JPEG and recorded payload as matched VLM; changing receiver adds no uplink.',
              'tdeepsc': {'complex_image_symbols': [24, 48, 96], 'question_uplink_symbols': 0,
                         'mean_complex_input_power': 1, 'fading': 'Rician K6dB, perfect CSI',
                         'comparison_limit': 'Same SNR is not resource matching; feature transmission differs from rate-adaptive JPEG simulation.'},
              'do_not_infer': ['Recorded payload bytes are not LDPC-coded channel-use counts.',
                               'A received outage placeholder file size is not attempted transmitted payload.',
                               'No watts-to-normalized-power conversion or measured model energy is available.',
                               'No end-to-end latency or joule claim follows from cached-feature training timing.']}
    for split in ['train', 'validation', 'test']:
        rows = json.loads((args.root / 'data' / f'{split}.json').read_text())
        groups = defaultdict(list)
        for row in rows:
            values = payloads[(row['image'], row['question'], float(row['snr']))]
            assert len(values) == 1, (row, values)
            groups[row['snr']].append(next(iter(values)))
        result['by_split'][split] = {str(snr): {'query_decisions': len(values),
            'recorded_payload_bytes_mean': statistics.mean(values),
            'recorded_payload_bytes_min': min(values), 'recorded_payload_bytes_max': max(values)}
            for snr, values in groups.items()}
    (args.root / 'communication_cost_audit.json').write_text(json.dumps(result, indent=2))
    print(json.dumps(result['by_split']['test'], indent=2))


if __name__ == '__main__':
    main()
