#!/usr/bin/env python3
"""
play.py — interactive CLI to play HU NLHE against a trained poker_ppo model.

Architecture: spawns the C++ binary in `--play <model>` mode and talks to it
over stdin/stdout using the simple `key value...` protocol defined in
`run_play()` in src/main.cpp. The Python side is purely UI:
  - renders cards as Unicode (e.g. K♠, T♥),
  - shows the model's action probabilities when it's the bot's turn,
  - prompts the user for an action when it's their turn,
  - logs every human (state, action) pair to a JSONL file for later use as
    behaviour-cloning / preference training data.

Usage:
    python tools/play.py --bin cmake-build-release/poker_ppo \\
                         --model poker_ppo_model_nlhe_full_52.pt \\
                         --hands 50 \\
                         --log human_actions.jsonl

The default action is "show probs and ask"; the bot's action is auto-selected
from the network's stochastic sample (matches training-time behaviour).
"""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
import textwrap
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import IO, Optional


# ─── Card rendering ────────────────────────────────────────────────────────────
# Engine encoding: card_id = (rank << 2) | suit, rank 0..12 = {2..A},
# suit 0..3 = {♣, ♦, ♥, ♠}. The exact suit->symbol mapping doesn't affect
# gameplay; only display.

RANK_LABELS = ["2", "3", "4", "5", "6", "7", "8", "9", "T", "J", "Q", "K", "A"]
# Match the engine's deck.c convention: SUIT_TO_CHAR = "shdc", so
# suit 0 = spade, 1 = heart, 2 = diamond, 3 = club.
SUIT_LABELS = ["♠", "♥", "♦", "♣"]
# Standard four-color deck colours.
SUIT_ANSI = {0: "\033[97m", 1: "\033[31m", 2: "\033[34m", 3: "\033[32m"}
RESET = "\033[0m"
BOLD = "\033[1m"
DIM = "\033[2m"


def card_str(card_id: int, color: bool = True) -> str:
    if card_id < 0 or card_id >= 52:
        return "??"
    rank = (card_id >> 2) & 0xF
    suit = card_id & 0x3
    s = f"{RANK_LABELS[rank]}{SUIT_LABELS[suit]}"
    return f"{SUIT_ANSI[suit]}{s}{RESET}" if color else s


def cards_str(card_ids: list[int], color: bool = True) -> str:
    return " ".join(card_str(c, color) for c in card_ids)


# ─── Engine info / state types ─────────────────────────────────────────────────


@dataclass
class EngineInfo:
    obs_dim: int
    action_count: int
    initial_stack: int
    small_blind: int
    big_blind: int
    min_raise: int
    max_raises_per_round: int
    pot_fractions: list[float]
    has_allin: bool

    def action_label(self, idx: int) -> str:
        if idx == 0:
            return "Fold"
        if idx == 1:
            return "Check/Call"
        raise_idx = idx - 2
        if raise_idx < len(self.pot_fractions):
            return f"Raise {self.pot_fractions[raise_idx]:g}× pot"
        if self.has_allin and raise_idx == len(self.pot_fractions):
            return "All-in"
        return f"<idx {idx}>"

    def short_label(self, idx: int) -> str:
        if idx == 0:
            return "F"
        if idx == 1:
            return "C"
        raise_idx = idx - 2
        if raise_idx < len(self.pot_fractions):
            return f"R{raise_idx}"
        if self.has_allin and raise_idx == len(self.pot_fractions):
            return "AI"
        return f"?{idx}"


@dataclass
class GameState:
    cur_player: int
    round: int
    pot: int
    cb: int
    raises: int
    hole_p0: list[int]
    hole_p1: list[int]
    stacks: list[int]
    board: list[int]
    mask: list[int]
    obs: list[float]
    done: bool
    utility_p0: int = 0
    utility_p1: int = 0


ROUND_NAMES = ["preflop", "flop", "turn", "river"]


# ─── Engine subprocess wrapper ─────────────────────────────────────────────────


class Engine:
    """Drives the C++ `--play` server over stdin/stdout."""

    def __init__(self, binary: str, model_path: str, extra_args: list[str] | None = None):
        if not os.path.exists(binary):
            raise FileNotFoundError(f"binary not found: {binary}")
        if not os.path.exists(model_path):
            raise FileNotFoundError(f"model not found: {model_path}")
        cmd = [binary, "--play", model_path] + (extra_args or [])
        self.proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=sys.stderr,  # let stderr stream straight through for debugging
            bufsize=1,           # line-buffered
            text=True,
        )
        # The C++ binary prints config / setup info to stdout before entering
        # the play REPL. Skip everything until we see the READY handshake.
        deadline = time.time() + 30.0
        while True:
            if time.time() > deadline:
                raise RuntimeError("engine handshake timed out (no READY in 30s)")
            line = self._readline_until_useful()
            if line.strip() == "READY":
                break
            # Otherwise it's pre-REPL setup output — let the user see it.
            print(f"  {DIM}[engine]{RESET} {line}")

    def close(self):
        if self.proc and self.proc.poll() is None:
            try:
                self._send("QUIT")
            except Exception:
                pass
            try:
                self.proc.wait(timeout=2)
            except subprocess.TimeoutExpired:
                self.proc.kill()

    # ── low-level IO ──────────────────────────────────────────────────────

    def _send(self, line: str) -> None:
        assert self.proc.stdin is not None
        self.proc.stdin.write(line + "\n")
        self.proc.stdin.flush()

    def _readline_until_useful(self) -> str:
        """Read lines from the engine, skipping blanks, until a non-blank
        line arrives. Returns it (newline stripped)."""
        assert self.proc.stdout is not None
        while True:
            line = self.proc.stdout.readline()
            if line == "":
                # pipe closed
                raise RuntimeError("engine pipe closed unexpectedly")
            line = line.rstrip("\r\n")
            if line:
                return line

    def _read_block(self) -> dict[str, list[str]]:
        """Read key/value lines until 'OK'. Returns {key: [tokens...]}."""
        out: dict[str, list[str]] = {}
        while True:
            line = self._readline_until_useful()
            if line == "OK":
                return out
            if line.startswith("ERR "):
                raise RuntimeError(f"engine error: {line[4:]}")
            tokens = line.split()
            if not tokens:
                continue
            key, *rest = tokens
            out[key] = rest

    # ── high-level commands ───────────────────────────────────────────────

    def info(self) -> EngineInfo:
        self._send("INFO")
        b = self._read_block()
        pf = b.get("pot_fractions", [])
        # pf format: "<n> <f0> <f1> ..."
        n = int(pf[0]) if pf else 0
        fractions = [float(x) for x in pf[1 : 1 + n]]
        return EngineInfo(
            obs_dim=int(b["obs_dim"][0]),
            action_count=int(b["action_count"][0]),
            initial_stack=int(b["initial_stack"][0]),
            small_blind=int(b["small_blind"][0]),
            big_blind=int(b["big_blind"][0]),
            min_raise=int(b["min_raise"][0]),
            max_raises_per_round=int(b["max_raises_per_round"][0]),
            pot_fractions=fractions,
            has_allin=int(b["has_allin"][0]) != 0,
        )

    def state(self) -> GameState:
        self._send("STATE")
        return self._parse_state(self._read_block())

    def step(self, action: int) -> GameState:
        self._send(f"STEP {action}")
        return self._parse_state(self._read_block())

    def reset(self) -> GameState:
        self._send("RESET")
        return self._parse_state(self._read_block())

    def model(self) -> dict:
        """Return {'sampled': int, 'greedy': int, 'value': float, 'probs': [...]}.
        Caller should typically use this only when it's the *bot's* turn —
        running it on a human-turn state would leak info about the hand.
        """
        self._send("MODEL")
        b = self._read_block()
        probs_raw = b["probs"]
        n = int(probs_raw[0])
        probs = [float(x) for x in probs_raw[1 : 1 + n]]
        return {
            "sampled": int(b["sampled"][0]),
            "greedy":  int(b["greedy"][0]),
            "value":   float(b["value"][0]),
            "probs":   probs,
        }

    # ── parsing ───────────────────────────────────────────────────────────

    @staticmethod
    def _parse_state(b: dict[str, list[str]]) -> GameState:
        board_raw = b.get("board", [])
        n_board = int(board_raw[0]) if board_raw else 0
        board = [int(x) for x in board_raw[1 : 1 + n_board]]
        mask_raw = b["mask"]
        nm = int(mask_raw[0])
        mask = [int(x) for x in mask_raw[1 : 1 + nm]]
        obs_raw = b["obs"]
        no = int(obs_raw[0])
        obs = [float(x) for x in obs_raw[1 : 1 + no]]
        return GameState(
            cur_player=int(b["cur_player"][0]),
            round=int(b["round"][0]),
            pot=int(b["pot"][0]),
            cb=int(b["cb"][0]),
            raises=int(b["raises"][0]),
            hole_p0=[int(x) for x in b["hole_p0"]],
            hole_p1=[int(x) for x in b["hole_p1"]],
            stacks=[int(x) for x in b["stacks"]],
            board=board,
            mask=mask,
            obs=obs,
            done=int(b["done"][0]) != 0,
            utility_p0=int(b.get("utility_p0", ["0"])[0]),
            utility_p1=int(b.get("utility_p1", ["0"])[0]),
        )


# ─── UI helpers ────────────────────────────────────────────────────────────────


def to_bb(amount_mbb: int, big_blind_mbb: int) -> float:
    return amount_mbb / big_blind_mbb if big_blind_mbb else 0.0


def render_table(
    state: GameState,
    info: EngineInfo,
    human_seat: int,
) -> str:
    """One-shot render of the current game state, hiding the bot's hole cards."""
    bb = info.big_blind
    bot_seat = 1 - human_seat
    human_hole = state.hole_p0 if human_seat == 0 else state.hole_p1
    bot_hole = state.hole_p0 if bot_seat == 0 else state.hole_p1
    human_hole_txt = cards_str(human_hole)
    if state.done:
        bot_hole_txt = cards_str(bot_hole)  # reveal at showdown
    else:
        bot_hole_txt = f"{DIM}?? ??{RESET}"

    board_txt = cards_str(state.board) if state.board else f"{DIM}—{RESET}"
    pot_bb = to_bb(state.pot, bb)
    cb_bb  = to_bb(state.cb,  bb)
    h_stack = to_bb(state.stacks[human_seat], bb)
    b_stack = to_bb(state.stacks[bot_seat],   bb)

    rnd = ROUND_NAMES[state.round] if 0 <= state.round < 4 else f"round {state.round}"
    turn = "YOU" if state.cur_player == human_seat else "BOT"

    lines = [
        f"{BOLD}─── {rnd}  ·  pot {pot_bb:.1f}bb  ·  to-call {cb_bb:.1f}bb  ·  raises {state.raises} ───{RESET}",
        f"  Bot  ({b_stack:>5.1f}bb): {bot_hole_txt}",
        f"  Board: {board_txt}",
        f"  You  ({h_stack:>5.1f}bb): {human_hole_txt}",
        f"  Turn: {BOLD}{turn}{RESET}",
    ]
    return "\n".join(lines)


def prompt_action(state: GameState, info: EngineInfo) -> int:
    """Show legal actions and prompt the human. Returns action index."""
    legal_idxs = [i for i, m in enumerate(state.mask) if m]
    print()
    print(f"  {BOLD}Your action:{RESET}")
    for i in legal_idxs:
        print(f"    [{i:>2}] {info.action_label(i)}")

    while True:
        try:
            raw = input(f"  > ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            raise SystemExit(0)
        if raw == "":
            continue
        if raw in ("q", "quit", "exit"):
            raise SystemExit(0)
        # Allow short labels too: F/C/R0/R1/.../AI
        upper = raw.upper()
        if upper in {"F", "FOLD"}:
            cand = 0
        elif upper in {"C", "CALL", "CHECK"}:
            cand = 1
        elif upper.startswith("R") and upper[1:].isdigit():
            cand = 2 + int(upper[1:])
        elif upper in {"AI", "ALLIN", "ALL-IN"}:
            cand = 2 + len(info.pot_fractions)
        else:
            try:
                cand = int(raw)
            except ValueError:
                print(f"  unknown action: {raw!r}")
                continue
        if cand not in legal_idxs:
            print(f"  illegal in this state. legal: {legal_idxs}")
            continue
        return cand


def render_model_view(model: dict, info: EngineInfo) -> str:
    probs = model["probs"]
    pairs = sorted(enumerate(probs), key=lambda kv: -kv[1])
    top = pairs[:6]
    parts = [f"{info.short_label(i)}={p*100:.1f}%" for i, p in top if p > 1e-3]
    val = model["value"]
    return (f"  {DIM}bot picked{RESET} {info.action_label(model['sampled'])}"
            f"  {DIM}(greedy={info.short_label(model['greedy'])}, V={val:+.2f}){RESET}\n"
            f"  {DIM}top probs:{RESET} " + ", ".join(parts))


# ─── Action logger ─────────────────────────────────────────────────────────────


class ActionLog:
    """JSONL log of human (state, action) pairs."""

    def __init__(self, path: Optional[str]):
        self.path = path
        self.fp: Optional[IO] = open(path, "a") if path else None

    def write(self, hand_idx: int, state: GameState, action: int, info: EngineInfo, human_seat: int):
        if not self.fp:
            return
        rec = {
            "ts":            time.time(),
            "hand":          hand_idx,
            "human_seat":    human_seat,
            "cur_player":    state.cur_player,
            "round":         state.round,
            "pot":           state.pot,
            "current_bet":   state.cb,
            "raises":        state.raises,
            "hole_cards":    (state.hole_p0 if human_seat == 0 else state.hole_p1),
            "board":         state.board,
            "stacks":        state.stacks,
            "legal_mask":    state.mask,
            "obs":           state.obs,
            "action":        action,
            "action_label":  info.action_label(action),
        }
        self.fp.write(json.dumps(rec) + "\n")
        self.fp.flush()

    def close(self):
        if self.fp:
            self.fp.close()


# ─── Main loop ─────────────────────────────────────────────────────────────────


def play_one_hand(engine: Engine, info: EngineInfo, log: ActionLog,
                  hand_idx: int, human_seat: int, show_model: bool):
    state = engine.reset()
    while not state.done:
        os.system("clear" if os.name == "posix" else "cls")
        print(f"{BOLD}Hand {hand_idx + 1}{RESET}  ·  you are seat {human_seat}")
        print(render_table(state, info, human_seat))

        if state.cur_player == human_seat:
            action = prompt_action(state, info)
            log.write(hand_idx, state, action, info, human_seat)
            state = engine.step(action)
        else:
            model = engine.model()
            if show_model:
                print(render_model_view(model, info))
                # Tiny pause so the user can see what the bot picked.
                time.sleep(0.6)
            state = engine.step(model["sampled"])

    # Hand over.
    os.system("clear" if os.name == "posix" else "cls")
    print(f"{BOLD}Hand {hand_idx + 1} — showdown{RESET}  ·  you are seat {human_seat}")
    print(render_table(state, info, human_seat))
    bb = info.big_blind
    you = state.utility_p0 if human_seat == 0 else state.utility_p1
    bot = state.utility_p1 if human_seat == 0 else state.utility_p0
    print()
    if you > bot:
        verdict = f"{BOLD}You win{RESET}"
    elif bot > you:
        verdict = f"{BOLD}Bot wins{RESET}"
    else:
        verdict = f"{BOLD}Tie{RESET}"
    print(f"  {verdict}.  result: you {to_bb(you, bb):+.2f} bb  ·  bot {to_bb(bot, bb):+.2f} bb")
    print()
    try:
        input("  press enter for next hand...")
    except (EOFError, KeyboardInterrupt):
        raise SystemExit(0)
    return you  # in mbb, for cumulative tracking


def main(argv=None):
    p = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    p.add_argument("--bin", default="cmake-build-release/poker_ppo",
                   help="path to the poker_ppo binary")
    p.add_argument("--model", required=True,
                   help="path to the saved model (.pt)")
    p.add_argument("--hands", type=int, default=20,
                   help="number of hands to play (default: 20)")
    p.add_argument("--log", default=None,
                   help="JSONL file to append human (state, action) pairs to")
    p.add_argument("--seat", type=int, choices=[0, 1, -1], default=-1,
                   help="which seat you play (0=BB-first or 1=SB-first); -1 alternates each hand")
    p.add_argument("--no-show-model", action="store_true",
                   help="hide the bot's chosen action / probabilities from the UI")
    args = p.parse_args(argv)

    engine = Engine(args.bin, args.model)
    log = ActionLog(args.log)
    try:
        info = engine.info()
        print(f"{BOLD}poker_ppo interactive play{RESET}")
        print(f"  binary: {args.bin}")
        print(f"  model:  {args.model}")
        print(f"  action_count={info.action_count}, blinds={info.small_blind}/{info.big_blind} mbb, "
              f"start_stack={info.initial_stack} mbb")
        print(f"  pot_fractions: {info.pot_fractions}")
        if args.log:
            print(f"  logging human actions -> {args.log}")
        print()
        try:
            input("  press enter to deal first hand...")
        except (EOFError, KeyboardInterrupt):
            return 0

        cumulative_mbb = 0
        for h in range(args.hands):
            human_seat = (h % 2) if args.seat < 0 else args.seat
            you_mbb = play_one_hand(engine, info, log, h, human_seat,
                                    show_model=not args.no_show_model)
            cumulative_mbb += you_mbb
            print(f"  cumulative: {to_bb(cumulative_mbb, info.big_blind):+.2f} bb "
                  f"over {h + 1} hands")
            print()
        print(f"{BOLD}Done.{RESET}  Final: {to_bb(cumulative_mbb, info.big_blind):+.2f} bb "
              f"over {args.hands} hands.")
    finally:
        log.close()
        engine.close()


if __name__ == "__main__":
    sys.exit(main() or 0)
