#!/usr/bin/env python3
"""Werewolf agent (rule-based, stdlib only) for make.fun bounty werewolf-agent-01.

Fresh rebuild. Plays 30 games per trial against 7 house players through the
env at MAKEFUN_ENV_URL. No model calls, no wall clock, no unseeded randomness:
every choice is a deterministic function of (MAKEFUN_TRIAL_SEED, game number,
the view served by the env). Run as: python3 -P -s main.py

Strategy (proven locally in an earlier build, re-implemented here):
- Parse the full public log every decision into a suspicion score per seat.
- Seer claims are first-class evidence: a lone seer claim naming a wolf is
  echoed and voted; conflicting claims are weaker; proven liars are ignored.
- Village abstains on weak evidence instead of exiling a random villager,
  except to tie/block a bad exile, or when few players are left.
- Seer checks the most suspicious unchecked player and reveals wolf results.
- Doctor protects the seer claimant first, then the loudest accuser.
- Wolf matches the partner's kill, kills seer claimants first, votes the
  most exilable villager, and buses a doomed partner only.
"""
import hashlib
import json
import os
import re
import sys
import urllib.request
import urllib.error

_MASK64 = (1 << 64) - 1


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & _MASK64


class Rng:
    """xoshiro256** seeded from sha256(domain || seed || label)."""

    def __init__(self, seed, label):
        h = hashlib.sha256(
            b"werewolf-rebuild/rng/v1\x00" + str(seed).encode()
            + b"\x00" + str(label).encode()).digest()
        s = [int.from_bytes(h[i:i + 8], "big") for i in range(0, 32, 8)]
        if not any(s):
            s[0] = 1
        self._s = s

    def next_u64(self):
        s0, s1, s2, s3 = self._s
        result = (_rotl((s1 * 5) & _MASK64, 7) * 9) & _MASK64
        t = (s1 << 17) & _MASK64
        self._s = [s0 ^ (s3 ^ s1), s1 ^ (s2 ^ s0),
                   (s2 ^ s0) ^ t, _rotl(s3 ^ s1, 45)]
        return result

    def random(self):
        return (self.next_u64() >> 11) / 9007199254740992

    def randint(self, lo, hi):
        span = hi - lo + 1
        limit = (1 << 64) - (1 << 64) % span
        while True:
            x = self.next_u64()
            if x < limit:
                return lo + x % span

    def choice(self, items):
        return items[self.randint(0, len(items) - 1)]

    def shuffle(self, items):
        items = list(items)
        for i in range(len(items) - 1, 0, -1):
            j = self.randint(0, i)
            items[i], items[j] = items[j], items[i]
        return items


# ---------------------------------------------------------------- env client
class Env:
    def __init__(self, base_url):
        self.base = base_url.rstrip("/")

    def _post(self, path, body):
        data = json.dumps(body).encode()
        req = urllib.request.Request(
            self.base + path, data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                return resp.status, json.loads(resp.read().decode() or "{}")
        except urllib.error.HTTPError as e:
            try:
                return e.code, json.loads(e.read().decode() or "{}")
            except Exception:
                return e.code, {}
        except Exception:
            return 0, {}

    def observe(self):
        return self._post("/observe", {})

    def act(self, game, decision, target=None, text=None, is_pass=False):
        body = {"game": game, "decision": decision}
        if decision == "speak":
            if is_pass:
                body["pass"] = True
            else:
                body["text"] = text
        else:
            body["target"] = target
        return self._post("/act", body)


# ---------------------------------------------------------------- state model
SEAT_RE = re.compile(r"seat\s+(\d+)", re.I)


def _mentions(text):
    return [int(m.group(1)) for m in SEAT_RE.finditer(text or "")
            if 1 <= int(m.group(1)) <= 8]


class State:
    """Everything deducible from one view: roles, log-derived evidence."""

    def __init__(self, view, trial_seed, game_num=0):
        self.view = view
        self.game_num = game_num
        self.seed = trial_seed
        self.day = view.get("day", 1)
        self.phase = view.get("phase", "")
        self.you = view.get("you") or {}
        self.me = self.you.get("seat")
        self.role = self.you.get("role")
        self.partner = None
        p = self.you.get("partner")
        if p:
            self.partner = p.get("seat")
        self.players = view.get("players") or []
        self.name = {pl["seat"]: pl.get("name", "?") for pl in self.players}
        self.alive = [pl["seat"] for pl in self.players if pl.get("alive")]
        self.revealed = {pl["seat"]: pl.get("role") for pl in self.players
                         if pl.get("role")}
        self.log = view.get("log") or []
        self.known_wolves = set()   # seats I am sure are wolves
        self.known_villagers = set()  # seats I am sure are NOT wolves
        self._private_knowledge()
        self.claims = []            # seer claims: {seat, wolf_target, clear_target}
        self.accusations = []       # (accuser, target) explicit "is a wolf"
        self.speech_count = {}
        self.accuse_count = {}
        self._parse_log()
        self.suspicion = self._suspicion()
        self.seer_claimants = [c["seat"] for c in self.claims]

    # -- private knowledge -------------------------------------------------
    def _private_knowledge(self):
        if self.role == "werewolf" and self.partner:
            self.known_wolves.add(self.partner)
            self.known_wolves.discard(self.me)
        for r in (self.view.get("seerResults") or []):
            t = r.get("seat")
            if t is None:
                continue
            if r.get("werewolf"):
                self.known_wolves.add(t)
            else:
                self.known_villagers.add(t)
        for seat, role in self.revealed.items():
            if role == "werewolf":
                self.known_wolves.add(seat)
            elif role:
                self.known_villagers.add(seat)
        self.known_wolves.discard(self.me)
        self.known_villagers.discard(self.me)

    # -- log parsing -------------------------------------------------------
    def _parse_log(self):
        for e in self.log:
            kind = e.get("kind")
            if kind == "speech":
                seat, text = e.get("seat"), e.get("text") or ""
                low = text.lower()
                self.speech_count[seat] = self.speech_count.get(seat, 0) + 1
                is_seer = ("i am the seer" in low or "i'm the seer" in low
                           or "i am seer" in low or "am the seer" in low)
                wolf_t = clear_t = None
                for m in SEAT_RE.finditer(text):
                    t = int(m.group(1))
                    if not (1 <= t <= 8) or t == seat:
                        continue
                    if "wolf" in low:
                        # "checked seat N ... wolf" / "seat N is a wolf"
                        wolf_t = t
                    if ("not a wolf" in low or "is innocent" in low
                            or "is a villager" in low or "cleared" in low
                            or "not wolf" in low):
                        clear_t = t
                if is_seer:
                    self.claims.append({"seat": seat, "wolf": wolf_t,
                                        "clear": clear_t})
                # explicit accusations: "seat N is a wolf" / "seat N ... wolf"
                if "wolf" in low and not is_seer:
                    for t in _mentions(text):
                        if t != seat:
                            self.accusations.append((seat, t))
                            self.accuse_count[seat] = \
                                self.accuse_count.get(seat, 0) + 1
                elif "wolf" in low and is_seer and wolf_t is not None:
                    self.accuse_count[seat] = self.accuse_count.get(seat, 0) + 1

    def claimants_naming(self, target):
        return [c for c in self.claims if c.get("wolf") == target]

    def lone_seer_target(self):
        """Target of a lone (uncontested) seer wolf-claim, else None."""
        targets = {}
        for c in self.claims:
            if c.get("wolf") is not None:
                targets.setdefault(c["wolf"], []).append(c["seat"])
        if len(targets) == 1:
            t = next(iter(targets))
            if t not in self.known_villagers and t in self.alive:
                return t
        return None

    # -- suspicion ---------------------------------------------------------
    def _suspicion(self):
        s = {seat: 0.0 for seat in range(1, 9)}
        weights = (1.0, 0.7, 0.5, 0.35, 0.25, 0.2)
        # day votes grouped per exile
        votes_by_day = {}
        for e in self.log:
            if e.get("kind") == "vote":
                votes_by_day.setdefault(e.get("day"), []).append(e)
        for e in self.log:
            if e.get("kind") == "exile" and e.get("seat") is not None:
                dead, role = e["seat"], e.get("role")
                day_votes = votes_by_day.get(e.get("day"), [])
                for i, v in enumerate(day_votes):
                    if v.get("target") != dead:
                        continue
                    w = weights[min(i, len(weights) - 1)]
                    if role == "villager":
                        s[v["seat"]] += 3.0 * w
                    elif role == "werewolf":
                        s[v["seat"]] -= 3.0 * w
                    elif role in ("seer", "doctor"):
                        s[v["seat"]] += 3.5 * w
        # accusations proved right / wrong by revealed roles
        for accuser, target in self.accusations:
            if target in self.revealed:
                if self.revealed[target] == "werewolf":
                    s[accuser] -= 2.0
                else:
                    s[accuser] += 2.0
        # seer claims: wolf target up, proven liars up, cleared down
        wolf_targets = {}
        for c in self.claims:
            if c.get("wolf") is not None:
                wolf_targets.setdefault(c["wolf"], []).append(c["seat"])
        for target, claimants in wolf_targets.items():
            boost = 4.0 if len(wolf_targets) == 1 else 2.0
            if target in self.known_villagers:
                boost = 0.0
            s[target] += boost
            # claimants whose named target turned out a villager are liars
            if target in self.revealed and self.revealed[target] != "werewolf":
                for cl in claimants:
                    s[cl] += 3.0
        for c in self.claims:
            if c.get("clear") is not None:
                s[c["clear"]] -= 2.0
        # quietness from day 2 on
        if self.day >= 2:
            for seat in self.alive:
                if seat != self.me and self.speech_count.get(seat, 0) == 0:
                    s[seat] += 0.5
        # my private knowledge overrides
        for seat in self.known_wolves:
            s[seat] = max(s[seat], 10.0)
        for seat in self.known_villagers:
            s[seat] = min(s[seat], -5.0)
        s[self.me] = -99.0
        if self.partner:
            s[self.partner] = -99.0
        return s

    # -- helpers -----------------------------------------------------------
    def rng(self, purpose):
        return Rng(self.seed, "g%d/%s/d%d/log%d" % (
            self.game_num, purpose, self.day, len(self.log)))

    def living_others(self):
        return [s for s in self.alive if s != self.me]

    def top_suspect(self, exclude=None):
        exclude = set(exclude or [])
        cands = [s for s in self.alive
                 if s != self.me and s not in exclude and s != self.partner]
        if not cands:
            return None
        best = max(self.suspicion[s] for s in cands)
        tops = [s for s in cands if self.suspicion[s] == best]
        if len(tops) == 1:
            return tops[0]
        return self.rng("tie").choice(sorted(tops))

    def today_votes(self):
        return [e for e in self.log if e.get("kind") == "vote"
                and e.get("day") == self.day]

    def vote_leader(self):
        counts = {}
        for v in self.today_votes():
            if v.get("target") is not None:
                counts[v["target"]] = counts.get(v["target"], 0) + 1
        if not counts:
            return None, 0
        top = max(counts.values())
        leaders = [s for s, c in counts.items() if c == top]
        return (leaders[0] if len(leaders) == 1 else None), top

    def name_of(self, seat):
        return self.name.get(seat, "?")


def decide_kill(st, options):
    """Wolf night kill. options: living non-wolves."""
    if not options:
        return None
    rng = st.rng("kill")
    # 1. match the partner's pick this night
    for v in (st.view.get("wolfVotes") or []):
        if v.get("night") == st.day and v.get("seat") != st.me \
                and v.get("target") in options:
            return v["target"]
    # 2. seer claimants first
    cands = [s for s in options if s in st.seer_claimants]
    if cands:
        return rng.choice(sorted(cands))
    # 3. players who accused me or my partner
    accusers = {a for (a, t) in st.accusations
                if t in (st.me, st.partner)}
    cands = [s for s in options if s in accusers]
    if cands:
        return rng.choice(sorted(cands))
    # 4. loudest villager (most speeches naming a wolf), else most suspicious
    loud = sorted(options, key=lambda s: (-st.accuse_count.get(s, 0),
                                          -st.speech_count.get(s, 0), s))
    if loud and st.accuse_count.get(loud[0], 0) > 0:
        return loud[0]
    sus = sorted(options, key=lambda s: (-st.suspicion.get(s, 0), s))
    return sus[0]


def decide_protect(st, options):
    """Doctor night protect. options exclude self and last night's target."""
    if not options:
        return None
    rng = st.rng("protect")
    # 1. seer claimant first; prefer one whose named wolf looks real
    claimants = [s for s in options if s in st.seer_claimants]
    if claimants:
        def claim_value(s):
            for c in st.claims:
                if c["seat"] == s and c.get("wolf") is not None:
                    return st.suspicion.get(c["wolf"], 0)
            return 0.0
        claimants.sort(key=lambda s: (-claim_value(s), s))
        return claimants[0]
    # 2. loudest accuser (most speeches naming a wolf)
    loud = sorted(options, key=lambda s: (-st.accuse_count.get(s, 0),
                                          -st.speech_count.get(s, 0), s))
    if loud and st.accuse_count.get(loud[0], 0) > 0:
        return loud[0]
    # 3. a seer-vouched villager
    vouched = [s for s in options if s in st.known_villagers]
    if vouched:
        return rng.choice(sorted(vouched))
    # 4. deterministic pick among the living
    return rng.choice(sorted(options))


def decide_inspect(st, options):
    """Seer night inspect: most suspicious unchecked living player."""
    if not options:
        return None
    checked = {r.get("seat") for r in (st.view.get("seerResults") or [])}
    cands = [s for s in options if s not in checked]
    if not cands:
        cands = list(options)
    cands.sort(key=lambda s: (-st.suspicion.get(s, 0), s))
    return cands[0]


def _known_wolf_target(st, options):
    for t in sorted(st.known_wolves):
        if t in options and t in st.alive:
            return t
    lone = st.lone_seer_target()
    if lone is not None and lone in options:
        return lone
    return None


def decide_vote(st, options):
    """Day vote. options exclude self."""
    if not options:
        return None
    if st.role == "werewolf":
        return _wolf_vote(st, options)
    # village side
    wolf = _known_wolf_target(st, options)
    if wolf is not None:
        return wolf
    top = st.top_suspect()
    if top is not None and top in options:
        score = st.suspicion.get(top, 0)
        if score >= 1.0 or len(st.alive) <= 5:
            # At 5 alive with 2 wolves, no exile means a night kill brings
            # parity at dawn, so the village must take its best shot.
            return top
        # weak evidence: try to tie/block a bad exile
        leader, nlead = st.vote_leader()
        if leader is not None and leader != top:
            if st.suspicion.get(leader, 0) < 0.5 and score >= 0.5:
                # vote top suspect only if it can tie the leader
                counts = {}
                for v in st.today_votes():
                    if v.get("target") is not None:
                        counts[v["target"]] = counts.get(v["target"], 0) + 1
                if counts.get(top, 0) + 1 >= nlead:
                    return top
        return None  # abstain rather than exile a random villager
    return None


def _wolf_vote(st, options):
    """Wolf day vote: most exilable villager; bus a doomed partner only."""
    villagers = [s for s in options if s != st.partner]
    if not villagers:
        return st.partner if st.partner in options else None
    # partner doomed? votes already cast against partner >= 2 and partner
    # is the clear leader, or suspicion on partner is very high
    counts = {}
    for v in st.today_votes():
        if v.get("target") is not None:
            counts[v["target"]] = counts.get(v["target"], 0) + 1
    if st.partner in options:
        p_votes = counts.get(st.partner, 0)
        p_sus = st.suspicion.get(st.partner, 0)
        leader, nlead = st.vote_leader()
        if leader == st.partner and p_votes >= 2 and p_sus >= 3.0:
            return st.partner  # bus: partner is going out anyway
    # join a plausible plurality on a villager
    leader, nlead = st.vote_leader()
    if leader is not None and leader in villagers \
            and st.suspicion.get(leader, 0) >= 0.5:
        return leader
    # most exilable villager = highest suspicion
    villagers.sort(key=lambda s: (-st.suspicion.get(s, 0), s))
    return villagers[0]


# ---------------------------------------------------------------- speech
def _voted_wolf_out(st):
    """Did I vote for a player later revealed as a wolf?"""
    for e in st.log:
        if e.get("kind") == "exile" and e.get("role") == "werewolf":
            for v in st.log:
                if v.get("kind") == "vote" and v.get("day") == e.get("day") \
                        and v.get("seat") == st.me \
                        and v.get("target") == e.get("seat"):
                    return True
    return False


def decide_speak(st):
    """Return a statement string, or None to pass."""
    rng = st.rng("speak")
    # --- seer with a fresh wolf result: reveal at once
    if st.role == "seer":
        wolves = [r for r in (st.view.get("seerResults") or [])
                  if r.get("werewolf") and r.get("seat") in st.alive]
        if wolves:
            t = wolves[0]["seat"]
            return ("I am the seer. I checked seat %d (%s) last night: wolf. "
                    "Vote seat %d out." % (t, st.name_of(t), t))
        cleared = [r["seat"] for r in (st.view.get("seerResults") or [])
                   if not r.get("werewolf") and r.get("seat") in st.alive]
        # Never announce a clear: naming myself as seer with no wolf found
        # only tells the wolves who to kill, and the clear is already used
        # privately (known_villagers feeds votes and doctor logic).
    # --- echo a lone seer claim (any village role, and wolf faking support)
    lone = st.lone_seer_target()
    if lone is not None and st.role != "werewolf":
        return ("Seat %d (%s) is named as a wolf by the seer. "
                "I agree, vote seat %d out."
                % (lone, st.name_of(lone), lone))
    if lone is not None and st.role == "werewolf":
        # as a wolf, echoing a true claim about a villager helps; but a lone
        # claim here names a real wolf only if the claimant is honest, which
        # would be my partner or me -- never echo that.
        if lone not in st.known_wolves:
            return ("Seat %d (%s) is named as a wolf by the seer. "
                    "I agree, vote seat %d out."
                    % (lone, st.name_of(lone), lone))
    # --- village: accuse top suspect when evidence is real
    if st.role != "werewolf":
        top = st.top_suspect()
        if top is not None and st.suspicion.get(top, 0) >= 1.0:
            reasons = [
                "Seat %d (%s) voted for a villager who was exiled. That points at a wolf.",
                "Seat %d (%s) keeps pushing villagers. I do not trust them.",
                "Seat %d (%s) looks off to me. Their votes do not help the village.",
            ]
            return rng.choice(reasons) % (top, st.name_of(top))
        if rng.random() < 0.35:
            return None  # pass quietly on a thin day
        neutrals = [
            "No strong read yet. I am watching the votes today.",
            "Let us not rush. A wrong exile helps the wolves.",
            "I have nothing solid yet. Listening for now.",
        ]
        return rng.choice(neutrals)
    # --- wolf: accuse the most exilable villager, defend self/partner weakly
    top = st.top_suspect()
    if top is not None and st.suspicion.get(top, 0) >= 0.5:
        reasons = [
            "Seat %d (%s) voted for a villager who was exiled. That is a wolf move.",
            "I think seat %d (%s) is hiding something. Their votes keep landing on villagers.",
            "Seat %d (%s) has been too quiet when villagers die. Watch them.",
        ]
        return rng.choice(reasons) % (top, st.name_of(top))
    if _voted_wolf_out(st):
        return "I voted out a wolf before. I am on the village side, listen to the votes."
    if rng.random() < 0.4:
        return None
    return "No strong read yet. I am watching the votes today."


# ---------------------------------------------------------------- main loop
def decide(st, decision, options, max_chars):
    if decision == "kill":
        return ("target", decide_kill(st, options or []))
    if decision == "protect":
        return ("target", decide_protect(st, options or []))
    if decision == "inspect":
        return ("target", decide_inspect(st, options or []))
    if decision == "vote":
        return ("target", decide_vote(st, options or []))
    if decision == "speak":
        text = decide_speak(st)
        if text is None:
            return ("pass", True)
        return ("text", text[:max_chars] if max_chars else text[:500])
    return ("target", None)


def main():
    env_url = os.environ.get("MAKEFUN_ENV_URL", "http://127.0.0.1:8901")
    trial_seed = os.environ.get("MAKEFUN_TRIAL_SEED", "default-seed")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE")
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                f.read(1)
        except Exception:
            pass
    env = Env(env_url)
    while True:
        status, obs = env.observe()
        if status == 0:
            # env unreachable for now; retry via a fresh observe
            continue
        if obs.get("done"):
            return 0
        pending = obs.get("pending") or []
        if not pending:
            if obs.get("done"):
                return 0
            continue
        progressed = False
        for item in pending:
            game = item.get("game")
            decision = item.get("decision")
            view = item.get("view") or {}
            options = item.get("options")
            max_chars = item.get("max_chars") or 500
            st = State(view, trial_seed, game_num=game or 0)
            if not st.alive or st.me is None:
                continue
            kind, value = decide(st, decision, options, max_chars)
            if kind == "target":
                status2, _ = env.act(game, decision, target=value)
            elif kind == "pass":
                status2, _ = env.act(game, decision, is_pass=True)
            else:
                status2, _ = env.act(game, decision, text=value)
            if status2 in (200, 409, 400):
                progressed = True
        if not progressed:
            # nothing could be answered; observe again (env may advance)
            continue
    return 0


if __name__ == "__main__":
    sys.exit(main())
