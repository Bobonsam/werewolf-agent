# Werewolf Agent

A rule-based Werewolf agent for the [make.fun](https://make.fun) bounty
**werewolf-agent-01** — "Build the best Werewolf agent".

Pure Python 3.13 standard library, no model calls, no API keys.
Deterministic: every decision is a function of the trial seed, the game
number and the view the environment serves.

## Strategy

- Log-derived game state (votes, exiles, deaths, claims, accusations)
  feeding a suspicion score. Seer claims are first-class evidence.
- Village: vote the most suspicious; abstain on weak evidence rather than
  help exile a random villager. As seer, check the most suspicious and
  reveal a wolf result at once. As doctor, protect the seer claimant.
- Wolf: blend in, coordinate kills with the partner, kill seer claimants
  first, bus a doomed partner only.

## Run

```
python3 -P -s main.py
```

The environment provides `MAKEFUN_PROMPT_FILE`, `MAKEFUN_TRIAL_SEED` and
`MAKEFUN_ENV_URL`.
