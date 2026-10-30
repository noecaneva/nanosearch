"""Unit test for the Exp 04 summary judge (nanosearch.env.judge_score, unsupported_titles) on 5
hand-made episodes. No model, no index.

  python -m checks.judge_score
"""

from nanosearch.env import judge_score, parse_turn, unsupported_titles

GOLD = ["The City Heiress", "Aphra Behn"]


def turns(*texts):
    return [dict(zip(("action", "arg"), parse_turn(t)), text=t) for t in texts]


# (name, turns, retrieved titles, judge, titles named, unsupported)
CASES = [
    ("both named in the done turn, both retrieved",
     turns("<search>The City Heiress</search>", "<search>Aphra Behn</search>",
           "Found THE CITY HEIRESS and aphra behn. <done/>"),
     ["The City Heiress", "Aphra Behn", "Ari Behn"], 1.0, 2, 0),
    ("one named, the other retrieved but not named",
     turns("<search>The City Heiress play</search>", "I have The City Heiress. <done/>"),
     ["The City Heiress", "Aphra Behn"], 0.5, 1, 0),
    ("titles in earlier turns only: judge reads the final turn",
     turns("The City Heiress by Aphra Behn. <search>Aphra Behn</search>", "<done/>"),
     ["The City Heiress", "Aphra Behn"], 0.0, 0, 0),
    ("search cap, no done turn: the last search turn is judged",
     turns("<search>x</search>", "<search>y</search>", "<search>Aphra Behn playwright</search>"),
     ["Comedy of intrigue"], 0.5, 1, 1),
    ("both named, none retrieved: full judge score, both unsupported",
     turns("<search>playwright</search>", "The City Heiress is by Aphra Behn. <done/>"),
     ["Ari Behn", "The Heiress (1786 play)"], 1.0, 2, 2),
]


def main():
    for name, ts, got, j, n, u in CASES:
        assert judge_score(ts, GOLD) == j, (name, judge_score(ts, GOLD))
        assert unsupported_titles(ts, GOLD, got) == (n, u), (name, unsupported_titles(ts, GOLD, got))
        print(f"ok  judge {j:.1f}  named {n}  unsupported {u}  {name}")
    assert judge_score([], GOLD) == 0.0
    print("PASS")


if __name__ == "__main__":
    main()
