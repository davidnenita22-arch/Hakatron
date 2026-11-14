"""Replay labelled local CSVs in order.  This is not the organizer evaluator."""
import argparse
import csv
import json
import time
from pathlib import Path

from detector import Detector


def evaluate(readings, labels):
    detector = Detector()
    start = time.perf_counter()
    scores = [detector.update(row) for row in readings]
    prediction = [score >= 0.5 for score in scores]
    truth = [int(row['is_anomaly']) == 1 for row in labels]
    tp = sum(p and t for p, t in zip(prediction, truth))
    fp = sum(p and not t for p, t in zip(prediction, truth))
    fn = sum(not p and t for p, t in zip(prediction, truth))
    f1 = 2 * tp / max(1, 2 * tp + fp + fn)
    recalls = {}
    for typ in ('point', 'drift', 'collective'):
        indices = [i for i, row in enumerate(labels) if row['anomaly_type'] == typ]
        recalls[typ] = sum(prediction[i] for i in indices) / max(1, len(indices))
    metric = 100 * (0.8 * f1 + 0.05 * recalls['point'] + 0.075 * recalls['drift'] + 0.075 * recalls['collective'])
    return {'rows': len(readings), 'tp': tp, 'fp': fp, 'fn': fn, 'f1': f1, 'recall': recalls,
            'score': metric, 'seconds': time.perf_counter() - start}, scores


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', default=str(Path(__file__).with_name('train-input.csv')))
    parser.add_argument('--labels', default=str(Path(__file__).with_name('train-output.csv')))
    parser.add_argument('--predictions')
    args = parser.parse_args()
    with open(args.input, newline='') as handle:
        readings = list(csv.DictReader(handle))
    with open(args.labels, newline='') as handle:
        labels = list(csv.DictReader(handle))
    assert len(readings) == len(labels)
    assert all(a['timestamp'] == b['timestamp'] for a, b in zip(readings, labels))
    metric, scores = evaluate(readings, labels)
    print(json.dumps(metric, indent=2))
    if args.predictions:
        with open(args.predictions, 'w', newline='') as handle:
            writer = csv.writer(handle)
            writer.writerow(['timestamp', 'score', 'predicted', 'is_anomaly', 'anomaly_type'])
            writer.writerows((r['timestamp'], s, int(s >= .5), y['is_anomaly'], y['anomaly_type'])
                             for r, s, y in zip(readings, scores, labels))


if __name__ == '__main__':
    main()
