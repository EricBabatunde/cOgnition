#!/usr/bin/env python3
"""
test_step1_2.py — Unit Verification for Phase 1.2 Terminal Dashboard
=====================================================================
Validates that TerminalDashboard correctly processes CustomRPGEnv
observations and renders all layout components without runtime
exceptions or string formatting errors.

Target: Python 3.10  |  Zero external test-framework dependencies.
"""

from __future__ import annotations

import io
import sys
import traceback
from typing import Callable, List, Tuple

from rich.console import Console

from environment import CustomRPGEnv, Action, Direction
from ui.terminal_dashboard import TerminalDashboard

# ──────────────────────────────────────────────
# ANSI formatting
# ──────────────────────────────────────────────
_G = "\033[92m"
_R = "\033[91m"
_C = "\033[96m"
_B = "\033[1m"
_D = "\033[2m"
_X = "\033[0m"

PASS = f"{_G}✓ PASS{_X}"
FAIL = f"{_R}✗ FAIL{_X}"


# ──────────────────────────────────────────────
# Helper: capture Rich layout render to string
# ──────────────────────────────────────────────
def _capture_layout(dash: TerminalDashboard, obs: dict, **kwargs) -> str:
    """Render a layout into a plain-text string via an in-memory buffer."""
    layout = dash.generate_layout(obs_dict=obs, **kwargs)
    buf = io.StringIO()
    console = Console(file=buf, width=120, force_terminal=True)
    console.print(layout)
    return buf.getvalue()


# ──────────────────────────────────────────────
# Test Cases
# ──────────────────────────────────────────────

def test_layout_renders_without_exception() -> None:
    """Test 1 — Layout renders with zero exceptions on fresh reset obs."""
    env = CustomRPGEnv()
    obs, info = env.reset(seed=0)
    dash = TerminalDashboard()

    # Must not raise
    output = _capture_layout(dash, obs, step_count=0, engine_state="EXPLORING")
    assert len(output) > 0, "Rendered output is empty"


def test_layout_with_full_payloads() -> None:
    """Test 2 — Layout renders with populated memory/logic payloads."""
    env = CustomRPGEnv()
    obs, info = env.reset(seed=42)
    dash = TerminalDashboard()

    dash.add_log("Env reset — spawn at (1,1)")
    dash.add_log("Scanning FOV for entities")

    output = _capture_layout(
        dash, obs,
        fast_mem_info={
            "active_goal": "Retrieve Key_Red",
            "faiss_match": "Room_1",
            "faiss_distance": 0.9412,
            "loop_detected": False,
            "buffer_length": 5,
        },
        logic_info={
            "safe_to_step": True,
            "active_rules": ["wall_blocking", "hazard_avoidance"],
            "forbidden_actions": [],
        },
        step_count=2,
        engine_state="EXPLORING",
    )
    assert len(output) > 0, "Rendered output is empty with full payloads"


def test_layout_with_none_payloads() -> None:
    """Test 3 — Layout handles None/missing dict payloads gracefully."""
    env = CustomRPGEnv()
    obs, _ = env.reset(seed=0)
    dash = TerminalDashboard()

    # All optional args omitted — must not raise
    output = _capture_layout(dash, obs)
    assert len(output) > 0, "Rendered output is empty with default args"


def test_player_icon_in_output() -> None:
    """Test 4 — Player directional icon appears in rendered output."""
    env = CustomRPGEnv()
    obs, _ = env.reset(seed=0)  # player facing EAST
    dash = TerminalDashboard()

    output = _capture_layout(dash, obs)
    # The EAST arrow is '► ' — check for '►' in output
    assert "►" in output, (
        f"Expected player arrow '►' in rendered output, not found.\n"
        f"First 400 chars: {output[:400]}"
    )

    # Turn north and re-render
    obs, _, _, _, _ = env.step(Action.TURN_LEFT)
    output = _capture_layout(dash, obs)
    assert "▲" in output, "Expected '▲' after TURN_LEFT to NORTH"


def test_grid_elements_in_output() -> None:
    """Test 5 — Grid entity icons and fog markers present in output."""
    env = CustomRPGEnv()
    obs, _ = env.reset(seed=42)
    dash = TerminalDashboard()

    # Move around to reveal more tiles
    for _ in range(4):
        obs, _, _, _, _ = env.step(Action.MOVE_FORWARD)
    env.step(Action.TURN_RIGHT)
    for _ in range(3):
        obs, _, _, _, _ = env.step(Action.MOVE_FORWARD)

    output = _capture_layout(dash, obs)

    # Wall character (█) must appear — boundary is always explored
    assert "█" in output, "Expected wall character '█' in output"

    # Fog character (░) must appear — not everything is explored
    assert "░" in output, "Expected fog character '░' in output"

    # Floor tile (·) must appear in explored area
    assert "·" in output, "Expected floor tile '·' in output"


def test_log_messages_in_output() -> None:
    """Test 6 — Log entries added via add_log() appear in rendered output.

    Note: Rich wraps long lines within layout panels, so the full
    contiguous message string may span multiple rendered lines.
    We verify distinctive keyword fragments instead.
    """
    env = CustomRPGEnv()
    obs, _ = env.reset(seed=0)
    dash = TerminalDashboard()

    dash.add_log("Alpha sentinel log entry")
    dash.add_log("Bravo sentinel log entry")

    output = _capture_layout(dash, obs)

    # Check distinctive keywords survive Rich line-wrapping
    assert "Alpha" in output, "Keyword 'Alpha' not found in rendered output"
    assert "sentinel" in output, "Keyword 'sentinel' not found in rendered output"
    assert "Bravo" in output, "Keyword 'Bravo' not found in rendered output"

    # Verify sequential numbering
    assert "[01]" in output, "Expected log index [01] in output"
    assert "[02]" in output, "Expected log index [02] in output"


def test_log_buffer_maxlen() -> None:
    """Test 7 — Log buffer respects maxlen and oldest entries are evicted."""
    dash = TerminalDashboard(max_log_lines=3)

    dash.add_log("line_one")
    dash.add_log("line_two")
    dash.add_log("line_three")
    dash.add_log("line_four")  # should evict line_one

    assert len(dash.log_buffer) == 3, (
        f"Expected buffer length 3, got {len(dash.log_buffer)}"
    )

    # Oldest entry (line_one) should be gone
    joined = " ".join(dash.log_buffer)
    assert "line_one" not in joined, "Evicted entry 'line_one' still in buffer"
    assert "line_four" in joined, "Newest entry 'line_four' missing from buffer"


def test_status_bar_content() -> None:
    """Test 8 — Status bar displays HP, step count, inventory, and state."""
    env = CustomRPGEnv()
    obs, _ = env.reset(seed=0)
    dash = TerminalDashboard()

    # Pick up key to populate inventory
    env._player.position = (2, 2)
    obs, _, _, _, info = env.step(Action.PICK_UP)

    output = _capture_layout(
        dash, obs,
        step_count=info["step_count"],
        engine_state="REFLECTING",
    )

    assert "HP:" in output or "♥" in output, "HP indicator missing from status bar"
    assert "Step:" in output or "⏱" in output, "Step counter missing from status bar"
    assert "key_red" in output, "Inventory item 'key_red' missing from status bar"
    assert "REFLECTING" in output, "Engine state 'REFLECTING' missing from status bar"


# ──────────────────────────────────────────────
# Test Runner
# ──────────────────────────────────────────────

_TESTS: List[Tuple[str, Callable[[], None]]] = [
    ("Test 1 — Layout renders without exception",    test_layout_renders_without_exception),
    ("Test 2 — Full payload render",                  test_layout_with_full_payloads),
    ("Test 3 — None/default payload render",          test_layout_with_none_payloads),
    ("Test 4 — Player icon in output",                test_player_icon_in_output),
    ("Test 5 — Grid elements in output",              test_grid_elements_in_output),
    ("Test 6 — Log messages in output",               test_log_messages_in_output),
    ("Test 7 — Log buffer maxlen eviction",           test_log_buffer_maxlen),
    ("Test 8 — Status bar content",                   test_status_bar_content),
]


def main() -> int:
    width = 62

    print(f"\n{_B}{'═' * width}{_X}")
    print(f"{_B}  PHASE 1.2 — TERMINAL DASHBOARD UNIT VERIFICATION{_X}")
    print(f"{_B}{'═' * width}{_X}")

    results: List[Tuple[str, bool, str]] = []

    for name, fn in _TESTS:
        print(f"\n{_C}{'━' * width}{_X}")
        print(f"{_C}┃{_X} {_B}{name}{_X}")
        print(f"{_C}{'━' * width}{_X}")

        try:
            fn()
            results.append((name, True, ""))
            print(f"  {_D}└─{_X} [{PASS}] {name}")
        except AssertionError as exc:
            results.append((name, False, str(exc)))
            print(f"  {_D}└─{_X} [{FAIL}] {name}")
            print(f"       {_R}{exc}{_X}")
            traceback.print_exc()
        except Exception as exc:
            results.append((name, False, str(exc)))
            print(f"  {_D}└─{_X} [{FAIL}] {name}")
            print(f"       {_R}Unexpected: {exc}{_X}")
            traceback.print_exc()

    # ── Summary ──
    total = len(results)
    passed = sum(1 for _, ok, _ in results if ok)
    failed = total - passed

    print(f"\n{_B}{'═' * width}{_X}")
    print(f"  {_B}TEST SUMMARY{_X}")
    print(f"{'─' * width}")
    for name, ok, err in results:
        tag = PASS if ok else FAIL
        suffix = f"  {_D}({err}){_X}" if err else ""
        print(f"  [{tag}]  {name}{suffix}")
    print(f"{'─' * width}")

    if failed == 0:
        print(f"  {_G}{_B}{passed}/{total} TESTS PASSED"
              f" — Terminal dashboard verified.{_X}")
    else:
        print(f"  {_R}{_B}{failed}/{total} TESTS FAILED"
              f" — Review errors above.{_X}")

    print(f"{_B}{'═' * width}{_X}\n")
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
