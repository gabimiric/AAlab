# Strategy: Trust Trapper

## Overview
This algorithm is designed to appear friendly and cooperative at first, but it incorporates strategic backstabbing and fake forgiveness to outmaneuver opponents. It lures other strategies into a sense of safety before delivering delayed retaliation or exploiting excessive cooperation.

## Behavior Summary

- **Opening Phase**:  
  Cooperates for the first 3 rounds to build trust.

- **Fake Forgiveness**:  
  If the opponent defects, the algorithm waits 2–3 rounds before retaliating. This delay makes it seem forgiving while preparing a counterattack.

- **Opportunistic Backstabbing**:  
  If the opponent has been cooperating consistently (e.g. 4 or more times in a row), the algorithm defects to exploit their trust.

- **Sneaky Defections**:  
  Occasionally defects if it has cooperated for 2 rounds in a row and the opponent’s overall cooperation ratio is high (>70%).

## Function Signature

```python
def strategy(my_history: list[int], opponent_history: list[int], rounds: int | None) -> int:
