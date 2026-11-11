#!/usr/bin/env python3
"""Werewolf agent for make.fun bounty werewolf-agent-01.

Pure heuristic play: no model calls, no API keys, standard library only.
Deterministic: every choice is a function of the trial seed, the game number
and the view the env served (seeded xoshiro256** RNG; no wall clock, no
unseeded randomness, no dict-order dependence).

Strategy (per game, rebuilt from the view's log on every decision):
- Suspicion model: votes for exiled villagers/wolves, accusations that
  proved right/wrong, seer claims verified or contradicted, defenses of
  revealed wolves, quietness. Seer results override everything.
- Village: vote the most suspicious; join a plurality only when its leader
  is a plausible wolf (never help exile a trusted villager); as seer, check
  the most suspicious and reveal a wolf result at once; as doctor, protect
  the seer claimant / confirmed villagers / loud accusers.
- Wolf: blend in (accuse villagers with plausible reasons, bandwagon),
  coordinate kills (match the partner's pick; first voter picks the most
  obvious target), kill seer claimants first, never accuse the partner,
  bus the partner only when they are doomed.
- Statements are short, consistent and evidence-based; contradictions are
  avoided by deriving every statement from the same log-derived state.

Run by the grader as: python3 -P -s /bot/main.py
Env: MAKEFUN_PROMPT_FILE, MAKEFUN_TRIAL_SEED, MAKEFUN_ENV_URL.
"""
import hashlib
import json
import os
import re
import sys
import urllib.error
import urllib.request

# ---------------------------------------------------------------- seeded RNG
_MASK64 = (1 << 64) - 1


def _rotl(x, k):
    return ((x << k) | (x >> (64 - k))) & _MASK64


class Rng:
    """xoshiro256** seeded from sha256(domain, seed, label)."""

    def __init__(self, seed, label=""):
        h = hashlib.sha256(
            b"werewolf-agent/rng/v1\x00" + str(seed).encode() + b"\x00" + str(label).encode()
        ).digest()
        s = [int.from_bytes(h[i:i + 8], "big") for i in range(0, 32, 8)]
        if not any(s):
            s[0] = 1
        self._s = s

    def next_u64(self):
        s0, s1, s2, s3 = self._s
        result = (_rotl((s1 * 5) & _MASK64, 7) * 9) & _MASK64
        t = (s1 << 17) & _MASK64
        a2 = s2 ^ s0
        a3 = s3 ^ s1
        self._s = [s0 ^ a3, s1 ^ a2, a2 ^ t, _rotl(a3, 45)]
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


# ---------------------------------------------------------------- env client
_OPENER = urllib.request.build_opener(urllib.request.ProxyHandler({}))


def env_post(env_url, path, body, timeout=65):
    """POST one JSON object; returns (status, parsed JSON or {}). Retries once
    on a connection failure (a request the env never saw has no effect)."""
    data = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    req = urllib.request.Request(
        env_url + path, data=data, method="POST",
        headers={"Content-Type": "application/json", "Accept": "application/json"},
    )
    for attempt in range(2):
        try:
            with _OPENER.open(req, timeout=timeout) as res:
                raw = res.read()
                try:
                    return res.status, json.loads(raw.decode("utf-8"))
                except ValueError:
                    return res.status, {}
        except urllib.error.HTTPError as err:
            try:
                raw = err.read()
            except Exception:
                raw = b""
            try:
                return err.code, json.loads(raw.decode("utf-8"))
            except ValueError:
                return err.code, {}
        except (urllib.error.URLError, OSError):
            if attempt == 0:
                continue
            raise
    raise RuntimeError("unreachable")


def fold(text):
    """Fold a statement the env's way: runs of breaks/tabs -> one space,
    other control chars dropped, ends trimmed."""
    t = re.sub(r"[\t\n\v\f\r\x85\u2028\u2029]+", " ", text)
    t = "".join(c for c in t if not (ord(c) < 32 or ord(c) == 127))
    return t.strip()


def clip(text, max_chars):
    folded = fold(text)
    if len(folded) <= max_chars:
        return folded
    cut = folded[:max_chars]
    # prefer a word boundary
    if " " in cut:
        cut = cut.rsplit(" ", 1)[0]
    return cut.strip()


# ---------------------------------------------------------------- game state
class GameState:
    """Everything derivable from one game's view (log + private fields)."""

    def __init__(self, view):
        self.view = view
        self.day = view.get("day", 1)
        self.phase = view.get("phase", "")
        you = view.get("you") or {}
        self.my_seat = you.get("seat")
        self.my_name = you.get("name", "?")
        self.my_role = you.get("role")
        self.alive = bool(you.get("alive"))
        self.partner = (you.get("partner") or {}).get("seat") if you.get("partner") else None
        self.players = {p["seat"]: p for p in view.get("players", [])}
        self.names = {s: p.get("name", f"seat {s}") for s, p in self.players.items()}
        self.revealed = {s: p["role"] for s, p in self.players.items() if p.get("role")}
        self.living = sorted(s for s, p in self.players.items() if p.get("alive"))
        self.log = view.get("log", []) or []
        self.seer_results = list(view.get("seerResults") or [])
        self.last_protect = view.get("lastProtect")
        self.wolf_votes = list(view.get("wolfVotes") or [])
        self.known_wolf = set()
        self.known_villager = set()
        for r in self.seer_results:
            if r.get("werewolf"):
                self.known_wolf.add(r["seat"])
            else:
                self.known_villager.add(r["seat"])
        self._parse_log()

    # -- log parsing --------------------------------------------------
    def _parse_log(self):
        self.votes_by_day = {}        # day -> list of {seat, target}
        self.exiles = {}              # day -> {seat, role, tally}
        self.speeches = []            # {seat, day, round, text}
        self.passes = []              # {seat, day, round}
        self.deaths = {}              # day -> {seat, role}
        self.claims = []              # {seat, day, kind: seer|doctor, target, result}
        self.accusations = []         # {seat, day, target}
        self.defenses = []            # {seat, day, target}
        self.my_speeches = []
        for e in self.log:
            k = e.get("kind")
            d = e.get("day", 0)
            if k == "vote":
                self.votes_by_day.setdefault(d, []).append({"seat": e.get("seat"), "target": e.get("target")})
            elif k == "exile":
                if e.get("seat") is not None:
                    self.exiles[d] = {"seat": e["seat"], "role": e.get("role"), "tally": e.get("tally") or []}
            elif k == "speech":
                rec = {"seat": e.get("seat"), "day": d, "round": e.get("round"), "text": e.get("text", "")}
                self.speeches.append(rec)
                if rec["seat"] == self.my_seat:
                    self.my_speeches.append(rec)
                self._parse_speech(rec)
            elif k == "pass":
                self.passes.append({"seat": e.get("seat"), "day": d, "round": e.get("round")})
            elif k == "dawn":
                if e.get("died") is not None:
                    self.deaths[d] = {"seat": e["died"], "role": e.get("role")}
        # claim verification: a seer claim is verified when its target is revealed
        for c in self.claims:
            t = c.get("target")
            if t in self.revealed:
                c["verified"] = (self.revealed[t] == "werewolf") if c.get("result") == "wolf" else (self.revealed[t] != "werewolf")

    def _seat_from_text(self, text):
        """Find a seat referenced in a statement: 'seat N' or a player name."""
        m = re.search(r"[Ss]eat (\d)", text)
        if m:
            s = int(m.group(1))
            if 1 <= s <= 8:
                return s
        low = text.lower()
        for s, name in self.names.items():
            if name and name.lower() in low:
                return s
        return None

    def _parse_speech(self, rec):
        text = rec["text"]
        low = text.lower()
        seat = rec["seat"]
        # seer claim (a seer statement is a claim, not a generic accusation)
        is_seer_claim = False
        if re.search(r"\bi am the seer\b|\bi'm the seer\b|\bseer here\b|\bi am seer\b|\bi'm seer\b|\bthe seer is me\b|\bseer role\b", low):
            is_seer_claim = True
            target = None
            result = None
            m = re.search(r"(?:check(?:ed|s)?|inspect(?:ed|s)?)\s+(?:seat\s+)?(\d)", low)
            if m:
                target = int(m.group(1))
            else:
                target = self._seat_from_text(text)
            if "not a wolf" in low or "isn't a wolf" in low or "is not a wolf" in low or "no wolf" in low:
                result = "villager"
            elif "wolf" in low:
                result = "wolf"
            elif "villager" in low or "village" in low:
                result = "villager"
            self.claims.append({"seat": seat, "day": rec["day"], "kind": "seer", "target": target, "result": result})
        elif re.search(r"\bchecked\b|\binspected\b|\blooked at\b", low):
            is_seer_claim = True
            target = self._seat_from_text(text)
            if "not a wolf" in low or "isn't a wolf" in low or "is not a wolf" in low:
                result = "villager"
            else:
                result = "wolf" if "wolf" in low else ("villager" if ("villager" in low or "village" in low) else None)
            if target is not None:
                self.claims.append({"seat": seat, "day": rec["day"], "kind": "seer", "target": target, "result": result})
        if re.search(r"\bi am the doctor\b|\bi'm the doctor\b", low):
            self.claims.append({"seat": seat, "day": rec["day"], "kind": "doctor", "target": None, "result": None})
        # accusations: explicit "X is a wolf" / "vote X out" only. Softer
        # suspicion language ("looks off", "my eyes are on") is too noisy:
        # counting it as an accusation drowned the vote/seer signals.
        if not is_seer_claim and "wolf" in low and (" is a wolf" in low or " is the wolf" in low or "vote seat" in low or "vote " in low):
            t = self._seat_from_text(text)
            if t is not None and t != seat:
                self.accusations.append({"seat": seat, "day": rec["day"], "target": t})
        # defenses: "trust seat N" / "N is not a wolf" / "leave N alone"
        if re.search(r"\btrust\b|\bnot a wolf\b|\bleave .* alone\b|\bdefend\b", low):
            t = self._seat_from_text(text)
            if t is not None and t != seat:
                self.defenses.append({"seat": seat, "day": rec["day"], "target": t})

    # -- derived facts -------------------------------------------------
    def trusted_seers(self):
        """Seats whose seer claim was verified by a revealed wolf."""
        out = set()
        for c in self.claims:
            if c.get("kind") == "seer" and c.get("verified") is True and c.get("result") == "wolf":
                out.add(c["seat"])
        return out

    def false_claimants(self):
        out = set()
        for c in self.claims:
            if c.get("kind") == "seer" and c.get("verified") is False:
                out.add(c["seat"])
        return out

    def seer_claimants(self):
        return {c["seat"] for c in self.claims if c.get("kind") == "seer"}

    def claimed_wolf_by_trusted(self):
        """Wolf seats named by a verified seer claim."""
        out = set()
        trusted = self.trusted_seers()
        for c in self.claims:
            if c.get("kind") == "seer" and c.get("seat") in trusted and c.get("result") == "wolf" and c.get("target"):
                out.add(c["target"])
        return out

    def claimed_villager_by_trusted(self):
        out = set()
        trusted = self.trusted_seers()
        for c in self.claims:
            if c.get("kind") == "seer" and c.get("seat") in trusted and c.get("result") == "villager" and c.get("target"):
                out.add(c["target"])
        return out

    def i_claimed_seer(self):
        return any(c["seat"] == self.my_seat and c.get("kind") == "seer" for c in self.claims)

    def revealed_wolf_seats(self):
        return {s for s, r in self.revealed.items() if r == "werewolf"}

    def speech_count(self, seat):
        return sum(1 for s in self.speeches if s["seat"] == seat)

    def pass_count(self, seat):
        return sum(1 for p in self.passes if p["seat"] == seat)

    def wolf_mention_count(self, seat):
        """Speeches by `seat` that name a wolf (how loud an accuser they are)."""
        return sum(1 for s in self.speeches if s["seat"] == seat and "wolf" in s["text"].lower())

    def votes_for(self, seat, day=None):
        """Votes cast for `seat` (optionally on one day)."""
        out = []
        for d, vs in self.votes_by_day.items():
            if day is not None and d != day:
                continue
            for v in vs:
                if v.get("target") == seat:
                    out.append(v)
        return out

    def current_day_votes(self):
        return list(self.votes_by_day.get(self.day, []))


# ---------------------------------------------------------------- suspicion
def suspicion_scores(st, for_wolf=False):
    """Suspicion of each living player (higher = more likely a wolf).
    for_wolf=True scores villagers by how exilable they look to the village."""
    scores = {s: 0.0 for s in st.living}
    reasons = {s: [] for s in st.living}

    def add(seat, pts, why):
        if seat in scores:
            scores[seat] += pts
            reasons[seat].append(why)

    # votes on exiled players: early votes matter most (a late bandwagon
    # vote on a wolf is often a wolf bussing its partner, not a clear)
    _VOTE_W = (1.0, 0.7, 0.5, 0.35, 0.25, 0.2)
    for day, ex in sorted(st.exiles.items()):
        dead, role = ex["seat"], ex.get("role")
        day_votes = st.votes_by_day.get(day, [])
        pos = 0
        for v in day_votes:
            if v.get("target") != dead:
                continue
            w = _VOTE_W[min(pos, 5)]
            pos += 1
            if role == "villager":
                add(v["seat"], 3.0 * w if pos == 1 else 1.5 * w, f"voted out villager {st.names.get(dead, dead)} day {day}")
            elif role == "werewolf":
                add(v["seat"], -2.5 * w, f"voted out wolf {st.names.get(dead, dead)} day {day}")
    # accusations that proved right/wrong
    for a in st.accusations:
        t = a["target"]
        if t in st.revealed:
            if st.revealed[t] == "villager":
                add(a["seat"], 2.0, f"accused villager {st.names.get(t, t)}")
            elif st.revealed[t] == "werewolf":
                add(a["seat"], -2.0, f"accused wolf {st.names.get(t, t)}")
    # defenses of revealed wolves / villagers
    for d in st.defenses:
        t = d["target"]
        if t in st.revealed:
            if st.revealed[t] == "werewolf":
                add(d["seat"], 2.0, f"defended wolf {st.names.get(t, t)}")
            elif st.revealed[t] == "villager":
                add(d["seat"], -1.0, f"defended villager {st.names.get(t, t)}")
    # seer claims: the TARGET of a claim is a suspect even before the claim
    # is verified. A lone seer naming a wolf is usually the real seer;
    # conflicting seer claims split the weight (one of them is lying).
    # Claims from a proven liar are ignored.
    false_c = st.false_claimants()
    wolf_namers = {c["seat"] for c in st.claims
                   if c.get("kind") == "seer" and c.get("result") == "wolf"
                   and c.get("seat") not in false_c and c.get("target") in scores}
    for c in st.claims:
        if c.get("kind") != "seer":
            continue
        if c.get("verified") is True and c.get("result") == "wolf":
            add(c["seat"], -3.0, "true seer claim")
        elif c.get("verified") is False:
            add(c["seat"], 3.0, "false seer claim")
        if c.get("seat") in false_c:
            continue
        t = c.get("target")
        if t is None or t not in scores:
            continue
        if c.get("result") == "wolf":
            pts = 4.0 if len(wolf_namers) <= 1 else 2.0
            add(t, pts, f"seer {st.names.get(c['seat'], c['seat'])} says wolf")
        elif c.get("result") == "villager":
            add(t, -2.0, f"seer {st.names.get(c['seat'], c['seat'])} cleared them")
    # unresolved accusations are weak evidence against the target (capped)
    _acc_n = {}
    for a in st.accusations:
        t = a["target"]
        if t in scores and t not in st.revealed:
            _acc_n[t] = _acc_n.get(t, 0) + 1
    for t, n in _acc_n.items():
        add(t, min(0.5 * n, 1.5), f"accused by {n} player(s)")
    # votes against me (when I am village, wolves push villagers)
    if st.my_role != "werewolf":
        for d, vs in st.votes_by_day.items():
            for v in vs:
                if v.get("target") == st.my_seat:
                    add(v["seat"], 1.0, "voted for me")
    # quiet players are slightly suspicious after day 1
    if st.day >= 2:
        for s in st.living:
            if s != st.my_seat and st.speech_count(s) == 0:
                add(s, 0.5, "never spoke")
            if st.pass_count(s) >= 2:
                add(s, 0.5, "keeps passing")
    # fine-grained tie-breakers: wolves accuse more and pass more; villagers
    # who drove a correct exile look better. These keep exact ties rare so
    # the ranking follows behaviour, never a seat number.
    for s in st.living:
        if s == st.my_seat:
            continue
        acc_n = sum(1 for a in st.accusations if a["seat"] == s)
        add(s, 0.05 * acc_n, "accuses often")
        add(s, 0.05 * st.pass_count(s), "passes often")
    # my own seat is never a candidate
    scores.pop(st.my_seat, None)
    reasons.pop(st.my_seat, None)
    # known results override
    for s in list(scores):
        if s in st.known_wolf or s in st.claimed_wolf_by_trusted():
            scores[s] = 100.0
            reasons[s] = ["seer found them a wolf"]
        elif s in st.known_villager or s in st.claimed_villager_by_trusted():
            scores[s] = -100.0
            reasons[s] = ["seer cleared them"]
    if for_wolf:
        # as a wolf I want the most exilable villager: reuse village suspicion
        # (how suspicious they look) and drop my partner entirely.
        scores.pop(st.partner, None) if st.partner in scores else None
        reasons.pop(st.partner, None) if st.partner in reasons else None
    return scores, reasons


def rank_suspects(st, for_wolf=False):
    scores, reasons = suspicion_scores(st, for_wolf=for_wolf)
    order = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
    return order, scores, reasons


# ---------------------------------------------------------------- statements
ACCUSE_TEMPLATES = [
    "I think seat {s} ({n}) is hiding something. {why}",
    "Seat {s} ({n}) looks off to me. {why}",
    "My eyes are on seat {s} ({n}). {why}",
    "I am suspicious of seat {s} ({n}). {why}",
    "Seat {s} ({n}) has not explained themselves. {why}",
]
DEFEND_TEMPLATES = [
    "I do not think seat {s} ({n}) is a wolf. {why}",
    "Leave seat {s} ({n}) alone for now, look at seat {o} instead.",
    "Seat {s} ({n}) has voted with the village. I trust them for now.",
]
NEUTRAL_TEMPLATES = [
    "No one has claimed anything yet. I am watching how everyone votes.",
    "Let us not rush. A wrong vote today only helps the wolves.",
    "I am listening. Say what you saw last night, if anything.",
]
NO_DEATH_TEMPLATES = [
    "No one died last night. The doctor saved someone, or the wolves missed.",
    "A quiet night. That tells us nothing yet, so let us talk.",
]
DAY1_TEMPLATES = [
    "Seat {d} died last night and they were village. Someone here did that.",
    "First day. I have no proof yet, but I am watching seat {s} ({n}).",
    "The wolves struck seat {d}. Let us talk before we vote anyone out.",
]
DAY1_NO_DEATH = [
    "First day, and no one died. Let us talk before anyone votes.",
    "No one died last night. I have no proof yet, so I am listening.",
]
REVEAL_TEMPLATES = [
    "I am the seer. I checked seat {s} ({n}) last night: wolf. Vote seat {s} out.",
    "I am the seer. Seat {s} ({n}) is a wolf, I checked them. Vote them out now.",
    "Listen: I am the seer, and seat {s} ({n}) is a wolf. Do not waste this vote.",
]


def reason_text(st, seat, reasons):
    rs = reasons.get(seat) or []
    if not rs:
        return "Their votes do not add up."
    r = rs[0]
    # turn the reason into a sentence
    if r.startswith("voted out villager"):
        m = re.search(r"day (\d+)", r)
        return f"They voted out a villager on day {m.group(1)}." if m else "They voted out a villager."
    if r.startswith("voted out wolf"):
        return "They helped vote out a wolf, I will give them that."
    if r.startswith("accused villager"):
        return "They accused a villager who turned out innocent."
    if r.startswith("accused wolf"):
        return "They called out a wolf before anyone else did."
    if r.startswith("defended wolf"):
        return "They defended someone who turned out to be a wolf."
    if r.startswith("false seer claim"):
        return "They claimed to be the seer and were wrong."
    if r.startswith("voted for me"):
        return "They voted for me, and I am no wolf."
    if r == "never spoke":
        return "They have said almost nothing."
    if r == "keeps passing":
        return "They keep passing instead of speaking."
    if r == "seer found them a wolf":
        return "The seer found them a wolf."
    return "Their votes do not add up."


# ---------------------------------------------------------------- agent
class Agent:
    def __init__(self, env_url, seed):
        self.env_url = env_url
        self.seed = seed
        self.decisions = 0
        self.fallbacks = 0

    def rng(self, game, label):
        return Rng(self.seed, f"g{game}:{label}")

    # -- main loop ------------------------------------------------------
    def run(self):
        while True:
            status, obs = env_post(self.env_url, "/observe", {})
            if status != 200 or not isinstance(obs, dict):
                raise RuntimeError(f"/observe answered {status}")
            if obs.get("done"):
                break
            pending = obs.get("pending") or []
            for p in pending:
                self.answer(p)
        return self.decisions, self.fallbacks

    def answer(self, p):
        game = p.get("game")
        kind = p.get("decision")
        view = p.get("view") or {}
        st = GameState(view)
        act = self.decide(p, st)
        status, _ = env_post(self.env_url, "/act", act, timeout=30)
        self.decisions += 1
        if status == 200:
            return
        if status == 409:
            return  # already answered / moved on
        # refusal: fall back to pass/abstain once
        self.fallbacks += 1
        fb = {"game": game, "decision": kind}
        if kind == "speak":
            fb["pass"] = True
        else:
            fb["target"] = None
        env_post(self.env_url, "/act", fb, timeout=30)

    # -- dispatch ---------------------------------------------------------
    def decide(self, p, st):
        kind = p.get("decision")
        game = p.get("game")
        if kind == "speak":
            return self.decide_speak(p, st)
        if kind == "vote":
            return self.decide_vote(p, st)
        if kind == "kill":
            return self.decide_kill(p, st)
        if kind == "protect":
            return self.decide_protect(p, st)
        if kind == "inspect":
            return self.decide_inspect(p, st)
        return {"game": game, "decision": kind, "target": None}

    # -- speak ------------------------------------------------------------
    def decide_speak(self, p, st):
        game = p.get("game")
        max_chars = p.get("max_chars") or 500
        rng = self.rng(game, f"speak-d{st.day}-r{self._round_of(st)}")
        text = None
        if st.my_role == "werewolf":
            text = self._speak_wolf(st, rng)
        elif st.my_role == "seer":
            text = self._speak_seer(st, rng)
        else:
            text = self._speak_village(st, rng)
        if not text:
            return {"game": game, "decision": "speak", "pass": True}
        return {"game": game, "decision": "speak", "text": clip(text, max_chars)}

    def _round_of(self, st):
        # current debate round: count my speeches today +1, else derive from log
        mine_today = [s for s in st.speeches if s["seat"] == st.my_seat and s["day"] == st.day]
        return len(mine_today) + 1

    def _speak_village(self, st, rng):
        order, scores, reasons = rank_suspects(st)
        # defend myself when under attack
        self_def = self._defend_self(st, rng)
        if self_def:
            return self_def
        # echo a lone seer wolf claim: the village must act on it
        unverified = {}
        for c in st.claims:
            if (c.get("kind") == "seer" and c.get("result") == "wolf"
                    and c.get("target") in st.living and c.get("target") != st.my_seat
                    and c.get("seat") not in st.false_claimants()
                    and c.get("verified") is not True):
                unverified.setdefault(c["target"], c["seat"])
        if len(unverified) == 1:
            t, claimer = next(iter(unverified.items()))
            return (f"Seat {claimer} says they are the seer and checked seat {t} "
                    f"({st.names.get(t, t)}): wolf. I believe them. Vote seat {t} out.")
        # defend a likely villager being pushed, if any
        for s in st.living:
            if scores.get(s, 0) <= -1.0 and len(st.votes_for(s, st.day)) >= 1:
                others = [x for x, _ in order if x != s]
                o = others[0] if others else None
                if o is not None:
                    return rng.choice(DEFEND_TEMPLATES).format(s=s, n=st.names.get(s, s), o=o, why=reason_text(st, s, reasons))
        if not order:
            return self._neutral(st, rng)
        top, top_score = order[0]
        if top_score >= 1.0:
            return rng.choice(ACCUSE_TEMPLATES).format(s=top, n=st.names.get(top, top), why=reason_text(st, top, reasons))
        if top_score >= -0.5 and st.day >= 2:
            return rng.choice(ACCUSE_TEMPLATES).format(s=top, n=st.names.get(top, top), why=reason_text(st, top, reasons))
        return self._neutral(st, rng)

    def _defend_self(self, st, rng):
        """Defend myself when accused or voted for; returns a statement or None."""
        accusers = set()
        for e in st.speeches:
            if e["seat"] == st.my_seat or e["day"] != st.day:
                continue
            if st._seat_from_text(e["text"]) == st.my_seat and "wolf" in e["text"].lower():
                accusers.add(e["seat"])
        votes_me = len(st.votes_for(st.my_seat, st.day))
        if len(accusers) < 1 and votes_me < 1:
            return None
        # cite my best record: a wolf I helped exile
        for day, ex in sorted(st.exiles.items()):
            if ex.get("role") == "werewolf":
                mine = [v for v in st.votes_by_day.get(day, []) if v.get("seat") == st.my_seat and v.get("target") == ex["seat"]]
                if mine:
                    return f"I am no wolf. I voted out {st.names.get(ex['seat'], ex['seat'])} on day {day}, and they were a wolf. Look elsewhere."
        return "I am no wolf. I have voted with the village every day. Check my votes, then look at who is pushing me."

    def _speak_seer(self, st, rng):
        # reveal an unrevealed wolf result
        for r in st.seer_results:
            if r.get("werewolf") and r["seat"] in st.living and not self._already_revealed(st, r["seat"]):
                t = r["seat"]
                return rng.choice(REVEAL_TEMPLATES).format(s=t, n=st.names.get(t, t))
        return self._speak_village(st, rng)

    def _already_revealed(self, st, seat):
        for s in st.my_speeches:
            low = s["text"].lower()
            if ("seer" in low and str(seat) in low) or (st.names.get(seat, "").lower() in low and "seer" in low):
                return True
        return False

    def _speak_wolf(self, st, rng):
        order, scores, reasons = rank_suspects(st, for_wolf=True)
        # defend myself when under attack (same words a villager would use)
        self_def = self._defend_self(st, rng)
        if self_def:
            return self_def
        # defend partner if accused and the case is weak
        if st.partner in st.living:
            accusers = [e["seat"] for e in st.speeches if e["seat"] != st.my_seat and st.partner == st._seat_from_text(e["text"])]
            votes_on_partner = len(st.votes_for(st.partner, st.day))
            if accusers and votes_on_partner == 0 and rng.random() < 0.5:
                return rng.choice(DEFEND_TEMPLATES).format(s=st.partner, n=st.names.get(st.partner, st.partner), o=(order[0][0] if order else st.partner), why="Their votes look village to me.")
        if not order:
            return self._neutral(st, rng)
        top, top_score = order[0]
        # as a wolf, accuse the most exilable villager
        return rng.choice(ACCUSE_TEMPLATES).format(s=top, n=st.names.get(top, top), why=reason_text(st, top, reasons))

    def _neutral(self, st, rng):
        dead = st.deaths.get(st.day)
        if st.day == 1:
            if dead:
                order, _, _ = rank_suspects(st)
                s = order[0][0] if order else (st.living[0] if st.living else 1)
                return rng.choice(DAY1_TEMPLATES).format(d=dead["seat"], s=s, n=st.names.get(s, s))
            return rng.choice(DAY1_NO_DEATH)
        if dead:
            return f"Seat {dead['seat']} is gone and they were village. The wolves are among us, let us watch the votes today."
        return rng.choice(NO_DEATH_TEMPLATES)

    # -- vote ---------------------------------------------------------------
    def decide_vote(self, p, st):
        game = p.get("game")
        options = p.get("options") or [s for s in st.living if s != st.my_seat]
        if not options:
            return {"game": game, "decision": "vote", "target": None}
        if st.my_role == "werewolf":
            target = self._vote_wolf(st, options)
        else:
            target = self._vote_village(st, options)
        return {"game": game, "decision": "vote", "target": target}

    def _vote_village(self, st, options):
        order, scores, reasons = rank_suspects(st)
        order = [(s, sc) for s, sc in order if s in options]
        if not order:
            return None
        # known wolf first
        known = [s for s, sc in order if sc >= 100.0]
        if known:
            return known[0]
        top, top_score = order[0]
        # weak evidence and enough villagers left: abstain rather than help
        # exile a villager at random. With 4 or fewer alive the village must
        # exile the wolf today, so vote anyway. Exception: if the current
        # leader is a likely villager, vote my top suspect to try to tie and
        # block the bad exile (abstaining would let it happen).
        if top_score < 1.0 and len(st.living) >= 5:
            counts = {}
            for v in st.current_day_votes():
                if v.get("target") is not None:
                    counts[v["target"]] = counts.get(v["target"], 0) + 1
            if counts:
                leader = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
                leader_score = scores.get(leader, -99)
                if leader in options and leader_score >= 1.0 and leader in [s for s, _ in order[:3]]:
                    return leader
                if leader in options and leader_score <= -1.0:
                    return top
            return None
        # current plurality
        counts = {}
        for v in st.current_day_votes():
            if v.get("target") is not None:
                counts[v["target"]] = counts.get(v["target"], 0) + 1
        if counts:
            leader = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
            leader_score = scores.get(leader, -99)
            # join the leader only if they are a plausible wolf
            if leader in options and leader_score >= 1.0 and leader in [s for s, _ in order[:3]]:
                return leader
            # if the leader is a trusted villager, do not join: vote my suspect
            if leader_score <= -1.0:
                return top
        return top

    def _vote_wolf(self, st, options):
        # bus a doomed partner
        if st.partner in st.living:
            counts = {}
            for v in st.current_day_votes():
                if v.get("target") is not None:
                    counts[v["target"]] = counts.get(v["target"], 0) + 1
            pv = counts.get(st.partner, 0)
            if pv >= 3:
                return st.partner
        order, scores, reasons = rank_suspects(st, for_wolf=True)
        order = [(s, sc) for s, sc in order if s in options and s != st.partner]
        if not order:
            return None
        counts = {}
        for v in st.current_day_votes():
            if v.get("target") is not None:
                counts[v["target"]] = counts.get(v["target"], 0) + 1
        # bandwagon: a villager already leading with votes is the best push
        if counts:
            leader = max(sorted(counts.items()), key=lambda kv: kv[1])[0]
            if leader in options and leader != st.partner:
                return leader
        return order[0][0]

    # -- night: kill ----------------------------------------------------------
    def decide_kill(self, p, st):
        game = p.get("game")
        options = p.get("options") or []
        if not options:
            return {"game": game, "decision": "kill", "target": None}
        # coordinate: if partner already voted, match them
        mine = [v for v in st.wolf_votes if v["night"] == st.day and v["seat"] == st.partner and v.get("target") is not None]
        if mine:
            t = mine[0]["target"]
            if t in options:
                return {"game": game, "decision": "kill", "target": t}
        # first voter: pick the most obvious target so the partner agrees
        rng = self.rng(game, f"kill-d{st.day}")
        claimants = [s for s in st.seer_claimants() if s in options]
        if claimants:
            return {"game": game, "decision": "kill", "target": rng.choice(sorted(claimants))}
        # a player who accused me or my partner
        accusers = set()
        for e in st.speeches:
            if e["seat"] in options:
                low = e["text"].lower()
                if "not a wolf" in low or "isn't a wolf" in low:
                    continue
                tgt = st._seat_from_text(e["text"])
                if tgt in (st.my_seat, st.partner) and "wolf" in low:
                    accusers.add(e["seat"])
        if accusers:
            return {"game": game, "decision": "kill", "target": rng.choice(sorted(accusers))}
        # loudest villager (most speeches)
        loud = sorted(options, key=lambda s: (-st.speech_count(s), s))
        if loud and st.speech_count(loud[0]) >= 2:
            return {"game": game, "decision": "kill", "target": loud[0]}
        return {"game": game, "decision": "kill", "target": rng.choice(sorted(options))}

    # -- night: protect ---------------------------------------------------------
    def decide_protect(self, p, st):
        game = p.get("game")
        options = p.get("options") or []
        if not options:
            return {"game": game, "decision": "protect", "target": None}
        rng = self.rng(game, f"protect-d{st.day}")
        # Wolves kill in this order: seer claimants, then the loudest accuser
        # (most speeches naming a wolf), then anyone. Protect in that order,
        # preferring likely villagers (a wolf will never be the wolves' target).
        _, scores, _ = rank_suspects(st)
        likely_villager = [s for s in options if scores.get(s, 0) < 1.0]
        pool = likely_villager if likely_villager else options
        # seer claimant first; prefer the claimant whose named wolf looks
        # like a real wolf (that claim is the most likely true)
        claimants = [s for s in st.seer_claimants() if s in pool]
        if claimants:
            def claimer_rank(s):
                for c in st.claims:
                    if c.get("seat") == s and c.get("kind") == "seer" and c.get("result") == "wolf":
                        return -scores.get(c.get("target"), 0)
                return 0.0
            claimants.sort(key=lambda s: (claimer_rank(s), s))
            return {"game": game, "decision": "protect", "target": claimants[0]}
        # loudest accuser: most speeches naming a wolf
        loud = sorted(pool, key=lambda s: (-st.wolf_mention_count(s), s))
        if loud and st.wolf_mention_count(loud[0]) >= 1:
            return {"game": game, "decision": "protect", "target": loud[0]}
        # a villager vouched by a trusted seer
        vouched = [s for s in (st.claimed_villager_by_trusted() | st.known_villager) if s in pool]
        if vouched:
            return {"game": game, "decision": "protect", "target": rng.choice(sorted(vouched))}
        return {"game": game, "decision": "protect", "target": rng.choice(sorted(pool))}

    # -- night: inspect -----------------------------------------------------------
    def decide_inspect(self, p, st):
        game = p.get("game")
        options = p.get("options") or []
        if not options:
            return {"game": game, "decision": "inspect", "target": None}
        order, scores, reasons = rank_suspects(st)
        checked = {r["seat"] for r in st.seer_results}
        for s, _ in order:
            if s in options and s not in checked:
                return {"game": game, "decision": "inspect", "target": s}
        return {"game": game, "decision": "inspect", "target": None}


def main():
    env_url = os.environ.get("MAKEFUN_ENV_URL", "").rstrip("/")
    seed = os.environ.get("MAKEFUN_TRIAL_SEED", "")
    prompt_file = os.environ.get("MAKEFUN_PROMPT_FILE", "")
    if not env_url:
        sys.stderr.write("werewolf: MAKEFUN_ENV_URL is not set\n")
        return 1
    if prompt_file:
        try:
            with open(prompt_file, "r", encoding="utf-8") as f:
                f.read()
        except OSError as err:
            sys.stderr.write(f"werewolf: cannot read prompt file: {err}\n")
    agent = Agent(env_url, seed)
    try:
        decisions, fallbacks = agent.run()
    except Exception as err:  # noqa: BLE001 - never stall the trial
        sys.stderr.write(f"werewolf: {err}\n")
        return 1
    print(f"werewolf: {decisions} decisions, {fallbacks} fallbacks; trial done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
