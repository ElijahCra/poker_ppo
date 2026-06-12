#!/usr/bin/env python3
"""
play_web.py — browser UI to play HU NLHE against a trained poker_ppo model,
styled like an online poker table.

Architecture: same engine protocol as play.py (which it imports), wrapped in
a tiny stdlib HTTP server. The browser is purely presentational; the server
owns the engine, the hand/score bookkeeping, and — importantly — keeps the
bot's hole cards out of every response until the hand is over, so the client
can't peek via dev tools.

Usage (from the repo root):
    python tools/play_web.py --model poker_ppo_model_nlhe_full_52.pt
    # then open http://localhost:8777 (opens automatically when possible)

Run it from Windows or WSL: when launched from Windows it transparently
re-targets the (Linux) engine binary through `wsl -e`.

Endpoints:
    GET  /            the table
    GET  /api/init    engine info + fresh hand state
    POST /api/act     {"action": int} — human action (validated server-side)
    POST /api/ai      bot's turn: sample model, step; returns action (+probs
                      when --xray or the UI toggle asks AND the hand is past
                      the decision, i.e. always post-action)
    POST /api/next    deal the next hand (alternates seats)
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import socket
import subprocess
import sys
import threading
import time
import webbrowser
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from play import ActionLog, Engine, EngineInfo, GameState  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parent.parent
ROUND_NAMES = ["Preflop", "Flop", "Turn", "River"]


def to_wsl_path(p: Path) -> str:
    """C:\\foo\\bar → /mnt/c/foo/bar."""
    p = p.resolve()
    drive = p.drive.rstrip(":").lower()
    rest = "/".join(p.parts[1:])
    return f"/mnt/{drive}/{rest}"


class WslEngine(Engine):
    """Engine whose subprocess is launched through `wsl -e` when the host is
    Windows (the committed binary is a Linux ELF). Reuses all of Engine's IO."""

    def __init__(self, binary: Path, model_path: Path):
        if not binary.exists():
            raise FileNotFoundError(f"binary not found: {binary}")
        if not model_path.exists():
            raise FileNotFoundError(f"model not found: {model_path}")
        if platform.system() == "Windows":
            cmd = ["wsl", "-e", to_wsl_path(binary), "--play", to_wsl_path(model_path)]
        else:
            cmd = [str(binary), "--play", str(model_path)]
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            bufsize=1,
            text=True,
        )
        deadline = time.time() + 60.0
        while True:
            if time.time() > deadline:
                raise RuntimeError("engine handshake timed out (no READY in 60s)")
            line = self._readline_until_useful()
            if line.strip() == "READY":
                break
            print(f"  [engine] {line}")


class Table:
    """Server-side game session: engine + hand/seat/score bookkeeping.
    All access serialised by `lock` (the HTTP server is threaded)."""

    def __init__(self, engine: Engine, info: EngineInfo, log: ActionLog,
                 xray: bool):
        self.lock = threading.Lock()
        self.engine = engine
        self.info = info
        self.log = log
        self.xray = xray
        self.hand_idx = 0
        self.hero_seat = 0
        self.sb_seat = 0
        self.cumulative_mbb = 0
        self.scored = False           # cumulative applied for current hand
        # Single-driver fence: only the most recent page load may act.
        # Stale tabs (or a page from a previous server instance) otherwise
        # keep driving the engine alongside the live one.
        self.client_seq = 0
        self.state: GameState = engine.reset()
        self.sb_seat = self.state.cur_player  # HU: SB acts first preflop

    def register_client(self) -> int:
        self.client_seq += 1
        return self.client_seq

    def check_client(self, client) -> None:
        if int(client or 0) != self.client_seq:
            raise ValueError("another table session is active — reload this tab")

    # ── helpers ───────────────────────────────────────────────────────────

    def raise_to_amounts(self, state: GameState) -> list[int]:
        """Raise-slot target amounts in mbb, mirroring rebuild_action_table:
        expected = current_bet + fraction × pot."""
        return [state.cb + int(f * state.pot) for f in self.info.pot_fractions]

    def bundle(self) -> dict:
        """The client-visible view of the current state."""
        s = self.state
        hero, bot = self.hero_seat, 1 - self.hero_seat
        hero_hole = s.hole_p0 if hero == 0 else s.hole_p1
        bot_hole = s.hole_p0 if bot == 0 else s.hole_p1
        reveal = s.done or self.xray

        if s.done and not self.scored:
            self.cumulative_mbb += s.utility_p0 if hero == 0 else s.utility_p1
            self.scored = True

        return {
            "hand": self.hand_idx + 1,
            "hero_seat": hero,
            "sb_seat": self.sb_seat,
            "cur_player": s.cur_player,
            "round": s.round,
            "round_name": ROUND_NAMES[s.round] if 0 <= s.round < 4 else "?",
            "pot": s.pot,
            "cb": s.cb,
            "stacks": s.stacks,
            "board": s.board,
            "hero_hole": hero_hole,
            "bot_hole": bot_hole if reveal else [],
            "mask": s.mask,
            "raise_to": self.raise_to_amounts(s),
            "done": s.done,
            "hero_utility": (s.utility_p0 if hero == 0 else s.utility_p1) if s.done else 0,
            "cumulative_mbb": self.cumulative_mbb,
            "big_blind": self.info.big_blind,
            "xray": self.xray,
        }

    # ── actions (call under lock) ─────────────────────────────────────────

    def act_human(self, action: int) -> dict:
        s = self.state
        if s.done:
            raise ValueError("hand is over")
        if s.cur_player != self.hero_seat:
            raise ValueError("not your turn")
        if not (0 <= action < len(s.mask)) or not s.mask[action]:
            raise ValueError("illegal action")
        self.log.write(self.hand_idx, s, action, self.info, self.hero_seat)
        self.state = self.engine.step(action)
        return {"label": self.info.action_label(action), "state": self.bundle()}

    def act_bot(self, want_brain: bool) -> dict:
        s = self.state
        if s.done:
            raise ValueError("hand is over")
        if s.cur_player == self.hero_seat:
            raise ValueError("hero's turn")
        model = self.engine.model()
        self.state = self.engine.step(model["sampled"])
        out = {
            "action": model["sampled"],
            "label": self.info.action_label(model["sampled"]),
            "state": self.bundle(),
        }
        # Post-action only — the probs condition on the bot's hole cards, so
        # they're for analysis, never a pre-decision aid.
        if want_brain or self.xray:
            out["probs"] = model["probs"]
            out["value"] = model["value"]
            out["greedy"] = model["greedy"]
        return out

    def next_hand(self) -> dict:
        self.hand_idx += 1
        self.hero_seat = self.hand_idx % 2  # alternate position each hand
        self.scored = False
        self.state = self.engine.reset()
        self.sb_seat = self.state.cur_player
        return {"state": self.bundle()}


# ─── HTTP layer ────────────────────────────────────────────────────────────────

TABLE: Table = None  # set in main()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # quiet
        pass

    def _json(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n) or b"{}")

    def do_GET(self):
        if self.path == "/" or self.path.startswith("/index"):
            body = PAGE.encode()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
        elif self.path == "/api/init":
            with TABLE.lock:
                self._json({
                    "client": TABLE.register_client(),
                    "info": {
                        "big_blind": TABLE.info.big_blind,
                        "small_blind": TABLE.info.small_blind,
                        "initial_stack": TABLE.info.initial_stack,
                        "pot_fractions": TABLE.info.pot_fractions,
                        "has_allin": TABLE.info.has_allin,
                        "action_count": TABLE.info.action_count,
                    },
                    "state": TABLE.bundle(),
                })
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        try:
            body = self._read_body()
            with TABLE.lock:
                if self.path == "/api/act":
                    TABLE.check_client(body.get("client"))
                    self._json(TABLE.act_human(int(body["action"])))
                elif self.path == "/api/ai":
                    TABLE.check_client(body.get("client"))
                    self._json(TABLE.act_bot(bool(body.get("brain"))))
                elif self.path == "/api/next":
                    TABLE.check_client(body.get("client"))
                    self._json(TABLE.next_hand())
                else:
                    self._json({"error": "not found"}, 404)
        except Exception as e:  # surface to the UI banner
            self._json({"error": str(e)}, 400)


# ─── The table page ────────────────────────────────────────────────────────────

PAGE = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<title>poker_ppo — heads-up table</title>
<style>
  :root {
    --felt1: #2c7a3f; --felt2: #14502a; --rail: #5b3a1e; --rail2: #3d2713;
    --bg: #15171c; --card-w: 74px; --card-h: 104px;
  }
  * { box-sizing: border-box; user-select: none; }
  body {
    margin: 0; min-height: 100vh; background:
      radial-gradient(1200px 700px at 50% 30%, #23262e 0%, var(--bg) 70%);
    color: #e8e8e8; font-family: "Segoe UI", system-ui, sans-serif;
    display: flex; flex-direction: column; align-items: center;
  }
  header {
    width: 100%; padding: 10px 22px; display: flex; gap: 18px;
    align-items: baseline; font-size: 14px; color: #9aa0ab;
  }
  header .title { font-size: 17px; font-weight: 600; color: #e8e8e8; }
  header .score { margin-left: auto; font-variant-numeric: tabular-nums; }
  header .score b { color: #ffd76a; }

  #layout { display: flex; gap: 18px; align-items: flex-start; }

  /* ── table ── */
  #table-wrap { position: relative; width: 860px; height: 560px; }
  #felt {
    position: absolute; inset: 30px 20px 70px 20px; border-radius: 50%;
    background: radial-gradient(ellipse at 50% 38%, var(--felt1), var(--felt2) 78%);
    border: 14px solid var(--rail);
    box-shadow: inset 0 0 60px rgba(0,0,0,.55), 0 18px 40px rgba(0,0,0,.5),
                inset 0 0 0 2px var(--rail2);
  }
  #felt::after {
    content: "poker_ppo"; position: absolute; left: 50%; top: 56%;
    transform: translateX(-50%); font-size: 26px; letter-spacing: 6px;
    color: rgba(255,255,255,.07); font-weight: 700;
  }

  .seat { position: absolute; left: 50%; transform: translateX(-50%);
          display: flex; flex-direction: column; align-items: center; gap: 6px; }
  #seat-bot  { top: 8px; }
  #seat-hero { bottom: 12px; }
  .nameplate {
    background: rgba(10,12,16,.85); border: 1px solid #3a3f4a; border-radius: 10px;
    padding: 5px 14px; text-align: center; min-width: 150px;
    box-shadow: 0 4px 10px rgba(0,0,0,.4);
  }
  .nameplate .who { font-weight: 600; font-size: 14px; }
  .nameplate .stack { color: #ffd76a; font-size: 13px; font-variant-numeric: tabular-nums; }
  .nameplate.turn { border-color: #ffd76a; box-shadow: 0 0 14px rgba(255,215,106,.35); }
  .badge {
    position: absolute; width: 26px; height: 26px; border-radius: 50%;
    font-size: 10px; font-weight: 700; display: flex; align-items: center;
    justify-content: center; color: #222; box-shadow: 0 2px 5px rgba(0,0,0,.5);
  }
  .badge.sb { background: #cfd8e3; }
  .badge.bb { background: #ffd76a; }

  .cards { display: flex; gap: 8px; min-height: var(--card-h); }
  .card {
    width: var(--card-w); height: var(--card-h); border-radius: 8px;
    background: #fafbfc; color: #111; position: relative;
    box-shadow: 0 4px 10px rgba(0,0,0,.45), inset 0 0 0 1px rgba(0,0,0,.08);
    animation: deal .25s ease-out;
  }
  @keyframes deal { from { transform: translateY(-14px); opacity: 0; } }
  .card .corner { position: absolute; top: 5px; left: 7px; font-weight: 700;
                  font-size: 19px; line-height: 1.0; text-align: center; }
  .card .corner small { display: block; font-size: 15px; }
  .card .pip { position: absolute; bottom: 4px; right: 7px; font-size: 30px; }
  .card.s { color: #1d212b; } .card.h { color: #c62828; }
  .card.d { color: #1565c0; } .card.c { color: #2e7d32; }
  .card.back {
    background: repeating-linear-gradient(45deg, #30509e 0 6px, #27407e 6px 12px);
    border: 4px solid #e9ecf2;
  }

  #board { position: absolute; left: 50%; top: 38%; transform: translate(-50%, -50%);
           display: flex; gap: 10px; }
  #board .card { width: 64px; height: 90px; }
  #board .slot { width: 64px; height: 90px; border-radius: 8px;
                 border: 2px dashed rgba(255,255,255,.14); }
  #pot {
    position: absolute; left: 50%; top: 51.5%; transform: translateX(-50%);
    background: rgba(10,12,16,.8); padding: 4px 16px; border-radius: 16px;
    font-size: 14px; color: #ffd76a; border: 1px solid #3a3f4a;
    font-variant-numeric: tabular-nums; z-index: 2;
  }
  #street { position: absolute; left: 50%; top: 23%; transform: translateX(-50%);
            font-size: 12px; letter-spacing: 3px; color: rgba(255,255,255,.5);
            text-transform: uppercase; }
  .speech {
    position: absolute; background: #fff; color: #222; font-size: 13px;
    font-weight: 600; padding: 4px 12px; border-radius: 12px; opacity: 0;
    transition: opacity .15s; box-shadow: 0 4px 10px rgba(0,0,0,.4);
    white-space: nowrap;
  }
  .speech.show { opacity: 1; }
  #speech-bot  { top: 116px; left: 58%; }
  #speech-hero { bottom: 150px; left: 58%; }

  /* ── action bar ── */
  #actions {
    position: absolute; bottom: 0; left: 50%; transform: translateX(-50%);
    display: flex; flex-direction: column; gap: 8px; align-items: center;
    width: 100%;
  }
  #actions .row { display: flex; gap: 10px; align-items: center; justify-content: center; }
  #slider-row {
    background: rgba(10,12,16,.75); border: 1px solid #2c313b; border-radius: 10px;
    padding: 6px 14px; display: flex; gap: 12px; align-items: center;
  }
  #slider-row input[type=range] { width: 270px; accent-color: #c8842c; }
  #slider-amt { min-width: 100px; text-align: right; color: #ffd76a;
                font-weight: 700; font-variant-numeric: tabular-nums; }
  button.act {
    border: none; border-radius: 9px; padding: 11px 16px; font-size: 14px;
    font-weight: 700; color: #fff; cursor: pointer; min-width: 86px;
    box-shadow: 0 4px 0 rgba(0,0,0,.35); transition: filter .1s, transform .05s;
  }
  button.act:hover:not(:disabled) { filter: brightness(1.15); }
  button.act:active:not(:disabled) { transform: translateY(2px); box-shadow: 0 2px 0 rgba(0,0,0,.35); }
  button.act:disabled { opacity: .35; cursor: default; }
  .b-fold { background: #b3413c; }
  .b-call { background: #2e8b57; }
  .b-raise { background: #c8842c; }
  .b-allin { background: #8e3bb5; }
  .b-next { background: #3b6fb5; min-width: 140px; }

  /* ── result banner ── */
  #result {
    position: absolute; left: 50%; top: 38%; transform: translate(-50%, -50%);
    background: rgba(8,10,14,.92); border: 1px solid #ffd76a; border-radius: 14px;
    padding: 18px 34px; text-align: center; display: none; z-index: 5;
    box-shadow: 0 10px 40px rgba(0,0,0,.6);
  }
  #result .verdict { font-size: 22px; font-weight: 700; margin-bottom: 4px; }
  #result .amount { font-size: 16px; color: #ffd76a; }

  /* ── side panel ── */
  #side { width: 300px; display: flex; flex-direction: column; gap: 10px; }
  #log {
    background: rgba(10,12,16,.7); border: 1px solid #2c313b; border-radius: 10px;
    height: 430px; overflow-y: auto; padding: 10px 12px; font-size: 13px;
    line-height: 1.55;
  }
  #log .street { color: #8fa8c8; font-weight: 700; margin-top: 4px;
                 letter-spacing: 1px; font-size: 11.5px; }
  #log .you b { color: #7fd494; } #log .bot b { color: #e2a05e; }
  #log .brain { color: #818897; font-size: 12px; margin-left: 10px; }
  #log .brain .bar { display: inline-block; height: 7px; background: #4a6f9c;
                     border-radius: 3px; vertical-align: middle; margin: 0 4px 0 2px; }
  #panel-opts {
    background: rgba(10,12,16,.7); border: 1px solid #2c313b; border-radius: 10px;
    padding: 10px 12px; font-size: 13px; color: #9aa0ab;
  }
  #err { color: #ff8a80; font-size: 13px; min-height: 17px; }
</style>
</head>
<body>
<header>
  <span class="title">♠ poker_ppo</span>
  <span id="hand-no">hand 1</span>
  <span id="blinds"></span>
  <span class="score">session: <b id="score">+0.0 BB</b></span>
</header>

<div id="layout">
  <div id="table-wrap">
    <div id="felt"></div>
    <div id="street"></div>
    <div id="board"></div>
    <div id="pot">Pot 0.0 BB</div>

    <div class="seat" id="seat-bot">
      <div class="nameplate" id="plate-bot">
        <div class="who">🤖 PPO Bot</div><div class="stack" id="stack-bot"></div>
      </div>
      <div class="cards" id="cards-bot"></div>
    </div>
    <div class="speech" id="speech-bot"></div>

    <div class="seat" id="seat-hero" style="bottom: 78px;">
      <div class="cards" id="cards-hero"></div>
      <div class="nameplate" id="plate-hero">
        <div class="who">You</div><div class="stack" id="stack-hero"></div>
      </div>
    </div>
    <div class="speech" id="speech-hero"></div>

    <div id="result">
      <div class="verdict"></div><div class="amount"></div>
    </div>

    <div id="actions"></div>
  </div>

  <div id="side">
    <div id="log"></div>
    <div id="panel-opts">
      <label><input type="checkbox" id="brain"> show bot decision probs (post-action)</label>
      <div id="err"></div>
    </div>
  </div>
</div>

<script>
const SUITS = ["♠","♥","♦","♣"], SUITCLS = ["s","h","d","c"];
const RANKS = ["2","3","4","5","6","7","8","9","T","J","Q","K","A"];
let INFO = null, S = null, busy = false, prevRound = -1, raiseSlot = 0, CLIENT = 0;

const $ = id => document.getElementById(id);
const bb = mbb => (mbb / INFO.big_blind);
const fbb = mbb => bb(mbb).toFixed(1).replace(/\.0$/, "") + " BB";

function cardEl(id, small) {
  const d = document.createElement("div");
  if (id === null) { d.className = "card back"; return d; }
  const r = (id >> 2) & 15, s = id & 3;
  d.className = "card " + SUITCLS[s];
  d.innerHTML = `<div class="corner">${RANKS[r]}<small>${SUITS[s]}</small></div>` +
                `<div class="pip">${SUITS[s]}</div>`;
  return d;
}

function setCards(el, ids, hidden, n) {
  el.innerHTML = "";
  if (hidden) { for (let i = 0; i < n; i++) el.appendChild(cardEl(null)); return; }
  for (const id of ids) el.appendChild(cardEl(id));
}

function actionLabel(i) {
  if (i === 0) return "Fold";
  if (i === 1) return S.cb > 0 ? "Call" : "Check";
  const k = i - 2;
  if (k < INFO.pot_fractions.length) return "Raise to " + fbb(S.raise_to[k]);
  return "All-in";
}

function logLine(cls, html) {
  const d = document.createElement("div");
  d.className = cls; d.innerHTML = html;
  $("log").appendChild(d); $("log").scrollTop = 1e9;
}

function brainHtml(r) {
  if (r.probs === undefined) return "";
  const pairs = r.probs.map((p, i) => [p, i]).sort((a, b) => b[0] - a[0]).slice(0, 3);
  const lbl = i => i === 0 ? "F" : i === 1 ? "C" : (i - 2 < INFO.pot_fractions.length ? "R" + (i-2) : "AI");
  const bars = pairs.filter(([p]) => p > 0.005).map(([p, i]) =>
    `${lbl(i)}<span class="bar" style="width:${Math.round(p*46)}px"></span>${(p*100).toFixed(0)}%`).join(" ");
  return `<div class="brain">V=${r.value.toFixed(2)} · ${bars}</div>`;
}

function say(who, text) {
  const el = $("speech-" + who);
  el.textContent = text; el.classList.add("show");
  setTimeout(() => el.classList.remove("show"), 1400);
}

function maybeLogStreet() {
  if (S.round !== prevRound && !S.done) {
    if (S.round === 0) logLine("street", `— HAND ${S.hand} · blinds ${fbb(INFO.small_blind)}/${fbb(INFO.big_blind)} —`);
    else {
      const newCards = S.board.map(c => { const r=(c>>2)&15, s=c&3; return RANKS[r]+SUITS[s]; }).join(" ");
      logLine("street", `— ${S.round_name.toUpperCase()}: ${newCards} —`);
    }
    prevRound = S.round;
  }
}

function render() {
  const hero = S.hero_seat, bot = 1 - hero;
  $("hand-no").textContent = "hand " + S.hand;
  $("blinds").textContent = `blinds ${fbb(INFO.small_blind)}/${fbb(INFO.big_blind)} · stacks ${fbb(INFO.initial_stack)}`;
  const cum = bb(S.cumulative_mbb);
  $("score").textContent = (cum >= 0 ? "+" : "") + cum.toFixed(1) + " BB";
  $("score").style.color = cum >= 0 ? "#7fd494" : "#ff8a80";

  $("street").textContent = S.done ? "showdown" : S.round_name;
  $("pot").textContent = "Pot " + fbb(S.pot) + (S.cb > 0 && !S.done ? "  ·  bet " + fbb(S.cb) : "");
  $("stack-hero").textContent = fbb(S.stacks[hero]);
  $("stack-bot").textContent = fbb(S.stacks[bot]);
  $("plate-hero").classList.toggle("turn", !S.done && S.cur_player === hero);
  $("plate-bot").classList.toggle("turn", !S.done && S.cur_player === bot);

  // dealer/blind badges
  document.querySelectorAll(".badge").forEach(b => b.remove());
  const mk = (seat, cls, txt) => {
    const b = document.createElement("div");
    b.className = "badge " + cls; b.textContent = txt;
    const plate = seat === hero ? $("seat-hero") : $("seat-bot");
    b.style.right = "-34px"; b.style.top = "14px"; plate.style.position = "absolute";
    plate.appendChild(b);
  };
  mk(S.sb_seat, "sb", "SB"); mk(1 - S.sb_seat, "bb", "BB");

  setCards($("cards-hero"), S.hero_hole, false, 2);
  setCards($("cards-bot"), S.bot_hole, S.bot_hole.length === 0, 2);

  const board = $("board"); board.innerHTML = "";
  for (let i = 0; i < 5; i++) {
    if (i < S.board.length) board.appendChild(cardEl(S.board[i]));
    else { const s = document.createElement("div"); s.className = "slot"; board.appendChild(s); }
  }

  // actions: Fold / Check-Call / Raise(slider) / All-in — online-poker style
  const bar = $("actions"); bar.innerHTML = "";
  if (S.done) {
    const r = $("result");
    const u = bb(S.hero_utility);
    r.querySelector(".verdict").textContent = u > 0 ? "You win!" : (u < 0 ? "Bot wins" : "Chop");
    r.querySelector(".verdict").style.color = u > 0 ? "#7fd494" : (u < 0 ? "#ff8a80" : "#e8e8e8");
    r.querySelector(".amount").textContent = (u >= 0 ? "+" : "") + u.toFixed(2) + " BB";
    r.style.display = "block";
    const btn = document.createElement("button");
    btn.className = "act b-next"; btn.textContent = "Deal next hand";
    btn.onclick = nextHand; bar.appendChild(btn);
    return;
  }
  $("result").style.display = "none";
  const myTurn = S.cur_player === hero && !busy;
  const allinIdx = 2 + INFO.pot_fractions.length;
  const raiseIdxs = [];
  S.mask.forEach((legal, i) => {
    if (legal && i >= 2 && i < allinIdx) raiseIdxs.push(i);
  });
  if (raiseSlot >= raiseIdxs.length) raiseSlot = 0;

  // slider row (only when plain raises exist)
  if (raiseIdxs.length > 0) {
    const row = document.createElement("div");
    row.id = "slider-row";
    const frac = i => INFO.pot_fractions[i - 2];
    row.innerHTML =
      `<span style="color:#9aa0ab;font-size:12px">raise size</span>
       <input type="range" min="0" max="${raiseIdxs.length - 1}" value="${raiseSlot}" id="rslider">
       <span id="slider-amt"></span>`;
    bar.appendChild(row);
    const amt = () => {
      const i = raiseIdxs[raiseSlot];
      $("slider-amt").textContent =
        fbb(S.raise_to[i - 2]) + " (" + frac(i).toLocaleString() + "× pot)";
    };
    row.querySelector("#rslider").oninput = e => {
      raiseSlot = +e.target.value; amt();
      const rb = $("raise-btn"); if (rb) rb.textContent = "Raise to " + fbb(S.raise_to[raiseIdxs[raiseSlot] - 2]);
    };
    row.querySelector("#rslider").disabled = !myTurn;
    amt();
  }

  const row = document.createElement("div");
  row.className = "row";
  const mkBtn = (cls, text, onclick, id) => {
    const b = document.createElement("button");
    b.className = "act " + cls; b.textContent = text;
    b.disabled = !myTurn; b.onclick = onclick;
    if (id) b.id = id;
    row.appendChild(b);
  };
  if (S.mask[0]) mkBtn("b-fold", "Fold", () => actHuman(0));
  if (S.mask[1]) mkBtn("b-call", actionLabel(1), () => actHuman(1));
  if (raiseIdxs.length > 0) {
    mkBtn("b-raise", "Raise to " + fbb(S.raise_to[raiseIdxs[raiseSlot] - 2]),
          () => actHuman(raiseIdxs[raiseSlot]), "raise-btn");
  }
  if (S.mask[allinIdx]) mkBtn("b-allin", "All-in", () => actHuman(allinIdx));
  bar.appendChild(row);
}

async function api(path, body) {
  if (body !== undefined) body.client = CLIENT;
  const r = await fetch(path, body === undefined ? {} :
    { method: "POST", headers: {"Content-Type": "application/json"}, body: JSON.stringify(body) });
  const j = await r.json();
  if (!r.ok) throw new Error(j.error || "request failed");
  return j;
}

async function actHuman(i) {
  if (busy) return;
  busy = true; $("err").textContent = "";
  try {
    const lbl = actionLabel(i);
    const r = await api("/api/act", { action: i });
    say("hero", lbl);
    logLine("you", `<b>You:</b> ${lbl}`);
    S = r.state; maybeLogStreet(); render();
    await botLoop();
  } catch (e) { $("err").textContent = e.message; }
  busy = false; render();
}

async function botLoop() {
  try {
    while (!S.done && S.cur_player !== S.hero_seat) {
      render();
      await new Promise(res => setTimeout(res, 650 + Math.random() * 500));
      const r = await api("/api/ai", { brain: $("brain").checked });
      // label uses pre-action cb; reuse server's label which is authoritative
      say("bot", r.label.replace(/Raise .*pot/, "Raise"));
      logLine("bot", `<b>Bot:</b> ${r.label}` + brainHtml(r));
      S = r.state; maybeLogStreet(); render();
    }
    if (S.done) logLine("street", "— hand over —");
  } catch (e) { $("err").textContent = e.message; }
}

async function nextHand() {
  if (busy) return;
  busy = true; $("err").textContent = "";
  try {
    const r = await api("/api/next", {});
    S = r.state; prevRound = -1; maybeLogStreet(); render();
    await botLoop();
  } catch (e) { $("err").textContent = e.message; }
  busy = false; render();
}

(async () => {
  try {
    const init = await api("/api/init");
    INFO = init.info; S = init.state; CLIENT = init.client;
    maybeLogStreet(); render();
    busy = true; await botLoop(); busy = false; render();
  } catch (e) { $("err").textContent = e.message; }
})();
</script>
</body>
</html>
"""


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--bin", default=str(REPO_ROOT / "cmake-build-release" / "poker_ppo"))
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8777)
    p.add_argument("--log", default=None,
                   help="JSONL file to append human (state, action) pairs to")
    p.add_argument("--xray", action="store_true",
                   help="reveal bot hole cards live (analysis mode)")
    p.add_argument("--no-browser", action="store_true")
    args = p.parse_args(argv)

    global TABLE
    engine = WslEngine(Path(args.bin), Path(args.model))
    info = engine.info()
    TABLE = Table(engine, info, ActionLog(args.log), xray=args.xray)

    srv = ThreadingHTTPServer(("127.0.0.1", args.port), Handler)
    url = f"http://localhost:{args.port}"
    print(f"table ready → {url}   (ctrl-c to quit)")
    if not args.no_browser:
        threading.Timer(0.4, lambda: webbrowser.open(url)).start()
    try:
        srv.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        engine.close()
        TABLE.log.close()
    return 0


if __name__ == "__main__":
    sys.exit(main() or 0)
