# Strategy: Trust Trapper
## Overview
This strategy starts off friendly, cooperating for the first few rounds to build trust, then incorporates fake forgiveness and opportunistic backstabbing. It retaliates after a delay, making it seem like it forgives the opponent’s defection. Additionally, it defects if the opponent becomes too trusting by cooperating for too long. The goal is to remain unpredictable, exploiting the opponent's trust when the time is right.

## Behavior Breakdown with Code Snippets
### 1. Friendly Opening
The strategy cooperates for the first 3 rounds to build trust with the opponent. This lays the groundwork for potential future interactions and sets up opportunities for later backstabbing.

```python
if round_number < 3:
    return 1  # Cooperate for the first 3 rounds
```

### 2. Fake Forgiveness (Delayed Retaliation)
When the opponent defects, the strategy does not retaliate immediately. Instead, it waits for 2-3 rounds before striking back, making it appear forgiving. This creates a false sense of security for the opponent, who is then caught off guard when retaliation happens.

```python
try:
    last_defect = len(opponent_history) - 1 - opponent_history[::-1].index(0)
except ValueError:
    last_defect = -1  # If no defect has occurred

# Fake forgiveness: punish 2–3 rounds after the opponent's defection
if 0 <= last_defect < round_number - 2 and round_number - last_defect == 3:
    return 0  # delayed revenge
if 0 <= last_defect < round_number - 3 and round_number - last_defect == 4:
    return 0  # optional second strike
```

### 3. Punishing Over-Trust (Backstabbing)
If the opponent has been cooperating for 4 or more rounds, the strategy defects to punish their trust. This backstabbing keeps the opponent from feeling safe and helps break their streak of cooperation.

```python
coop_streak = 0
for move in reversed(opponent_history):
    if move == 1:
        coop_streak += 1
    else:
        break
if coop_streak >= 4:
    return 0  # Punish too much trust
```
### 4. Opportunistic Defection (Sneak Attack)
After cooperating for two rounds, the strategy will defect if the opponent has cooperated more than 70% of the time. This exploits opponents who trust too much while maintaining an unpredictable game flow.

```python
coop_ratio = opponent_history.count(1) / round_number
if my_history[-2:] == [1, 1] and coop_ratio > 0.7:
    return 0  # Defect opportunistically if opponent is too trusting
```
# Design Philosophy
The goal of this strategy is to manipulate the opponent’s behavior by starting off friendly, then betraying them at the right moment. It uses fake forgiveness and delayed retaliation to create the illusion of cooperation. The algorithm’s backstabbing behavior punishes overly trusting opponents while remaining unpredictable. By combining trust-building with strategic betrayal, the algorithm is designed to dominate opponents who rely on predictable patterns, forcing them into mistakes.
