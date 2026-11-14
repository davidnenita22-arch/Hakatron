"""Walu Season: starter main.py (the greedy baseline).

This file already speaks the evaluator's protocol. The part to change is the
Strategy class. As shipped it is pure greedy: try every spot once, then keep
going back to whichever spot has the best average so far. It runs end to end
and it does NOT reach the threshold.

Check it locally on the practice reef:

    python evaluation_script.py main.py --dataset train

Rules of the protocol (see task_description.md):
  * read test-input.json from the current directory (bare filename),
  * for every cast print one spot number on stdout, then read one line from
    stdin with the catch in kg,
  * start a fresh Strategy every day (spot numbers are reshuffled daily),
  * anything else you want to print goes to stderr.
"""

import json
import sys


class Strategy:
    """Finite-horizon UCB with a variance-aware exploration bonus.

    Empty casts do not rule out a valuable school spot. Sample every spot
    several times, then balance its average catch against uncertainty.
    """

    def __init__(self, n_spots, n_casts):
        self.n_spots = n_spots
        self.n_casts = n_casts
        self.casts = [0] * n_spots
        self.total = [0.0] * n_spots
        self.squares = [0.0] * n_spots
        self.turn = 0
        self.warmup = min(5, n_casts // n_spots)

    def choose(self):
        import math

        if self.turn < self.warmup * self.n_spots:
            return self.turn % self.n_spots

        best_spot = 0
        best_score = -math.inf
        for spot in range(self.n_spots):
            n = self.casts[spot]
            if n == 0:
                return spot
            mean = self.total[spot] / n
            variance = max(1.0, self.squares[spot] / n - mean * mean)
            confidence = math.log(max(math.e, self.n_casts / (self.n_spots * n)))
            score = (mean + math.sqrt(2.0 * variance * confidence / n)
                     + 10.0 * confidence / n)
            if score > best_score:
                best_spot, best_score = spot, score
        return best_spot

    def update(self, spot, catch):
        self.turn += 1
        self.casts[spot] += 1
        self.total[spot] += catch
        self.squares[spot] += catch * catch


def main():
    with open("test-input.json") as f:
        rules = json.load(f)

    for _day in range(rules["n_days"]):
        agent = Strategy(rules["n_spots"], rules["n_casts"])
        for _cast in range(rules["n_casts"]):
            spot = agent.choose()
            print(spot, flush=True)
            catch = float(sys.stdin.readline())
            agent.update(spot, catch)


if __name__ == "__main__":
    main()
