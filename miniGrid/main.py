#!/usr/bin/env python3
"""
main.py — Interactive Turn-Based Execution Loop
=================================================
Provides keyboard-driven manual teleop control of the Custom RPG
Environment with real-time Rich terminal dashboard rendering.

Controls:
    W / ↑       Move Forward
    A / ←       Turn Left
    D / →       Turn Right
    E           Pick Up Item
    Space       Toggle Interact (Door / Lever)
    Q           Quit

Target: Python 3.10  |  Platform: Linux (termios)
"""

from __future__ import annotations

import os
import select
import sys
import termios
import time
import tty
from typing import Optional, Union

from rich.console import Console
from rich.live import Live

from environment import Action, CustomRPGEnv, Direction
from ui.terminal_dashboard import TerminalDashboard


# ──────────────────────────────────────────────
# Terminal Mode Management
# ──────────────────────────────────────────────

def set_raw_mode(fd: int) -> list:
    """Set the terminal to non-canonical (cbreak) mode with no echo.
    
    Returns:
        The original termios settings to be restored later.
    """
    old_settings = termios.tcgetattr(fd)
    # setcbreak is similar to raw but handles signals like Ctrl-C better
    tty.setcbreak(fd)
    
    # Explicitly turn off ECHO if setcbreak didn't
    new_settings = termios.tcgetattr(fd)
    new_settings[3] &= ~termios.ECHO
    termios.tcsetattr(fd, termios.TCSADRAIN, new_settings)
    
    return old_settings


# ──────────────────────────────────────────────
# Non-blocking Input Poller
# ──────────────────────────────────────────────

def get_action_key(fd: int, timeout: float = 0.02) -> Optional[Union[Action, str]]:
    """Poll stdin for a valid action key or arrow sequence.
    
    Reads up to 32 bytes to handle escape sequences. Discards mouse/scroll
    events and unrecognized ANSI codes.
    
    Returns:
        Action enum, 'QUIT' string, or None if no valid input.
    """
    if not select.select([fd], [], [], timeout)[0]:
        return None

    try:
        data = os.read(fd, 32)
    except BlockingIOError:
        return None

    if not data:
        return None

    # Filter out mouse and trackpad scroll sequences
    # \x1b[M is X10 mouse, \x1b[< is SGR mouse, \x1b[35 and 36 are scroll events
    if (data.startswith(b'\x1b[M') or 
        data.startswith(b'\x1b[<') or 
        data.startswith(b'\x1b[35') or 
        data.startswith(b'\x1b[36')):
        # Flush any remaining bytes from stdin buffer
        while select.select([fd], [], [], 0.0)[0]:
            os.read(fd, 1024)
        return None

    # Decode string for easier parsing of standard keys
    try:
        s = data.decode('utf-8', errors='ignore')
    except Exception:
        return None

    if not s:
        return None

    # Exact matches for arrow sequences
    if s == '\x1b[A':
        return Action.MOVE_FORWARD
    if s == '\x1b[D':
        return Action.TURN_LEFT
    if s == '\x1b[C':
        return Action.TURN_RIGHT

    # Check first character for standard bindings
    ch = s[0]
    
    if ch in ('w', 'W'):
        return Action.MOVE_FORWARD
    if ch in ('a', 'A'):
        return Action.TURN_LEFT
    if ch in ('d', 'D'):
        return Action.TURN_RIGHT
    if ch in ('e', 'E'):
        return Action.PICK_UP
    if ch == ' ':
        return Action.TOGGLE_INTERACT
    if ch in ('q', 'Q', '\x03'):  # \x03 is Ctrl-C
        return 'QUIT'

    return None


# ──────────────────────────────────────────────
# Direction label helper
# ──────────────────────────────────────────────

_DIR_SYMBOL: dict[Direction, str] = {
    Direction.NORTH: "▲ N",
    Direction.EAST:  "► E",
    Direction.SOUTH: "▼ S",
    Direction.WEST:  "◄ W",
}


# ──────────────────────────────────────────────
# Main interactive loop
# ──────────────────────────────────────────────

def main() -> int:
    """Run the interactive teleop loop."""
    console = Console()
    env = CustomRPGEnv()
    dash = TerminalDashboard()

    obs, info = env.reset(seed=42)
    ps = obs["player_state"]

    dash.add_log("[bold cyan]Engine initialised[/bold cyan] — spawn (1,1) facing EAST")
    dash.add_log("[dim]Controls: W/↑=Fwd  A/←=Left  D/→=Right  E=Pickup  Space=Interact  Q=Quit[/dim]")

    terminated = False
    truncated = False
    step_count = 0
    engine_state = "EXPLORING"

    fd = sys.stdin.fileno()
    original_term = set_raw_mode(fd)

    try:
        with Live(
            dash.generate_layout(
                obs_dict=obs,
                step_count=step_count,
                engine_state=engine_state,
            ),
            console=console,
            refresh_per_second=15,
            auto_refresh=True,
            screen=True,
        ) as live:

            while not terminated and not truncated and ps["health"] > 0:
                action_or_quit = get_action_key(fd, timeout=0.02)
                
                if action_or_quit == 'QUIT':
                    dash.add_log("[bold yellow]⏹ Manual quit requested[/bold yellow]")
                    live.update(dash.generate_layout(
                        obs_dict=obs,
                        step_count=step_count,
                        engine_state="QUIT",
                    ))
                    break
                    
                elif isinstance(action_or_quit, Action):
                    action = action_or_quit
                    prev_pos = tuple(ps["position"])

                    obs, reward, terminated, truncated, info = env.step(action)
                    ps = obs["player_state"]
                    step_count = info["step_count"]

                    # ── Build log entry ──
                    pos = ps["position"]
                    direction = ps["direction"]
                    dir_label = _DIR_SYMBOL.get(Direction(direction), "?")
                    hp = ps["health"]

                    # Detect wall collision: MOVE_FORWARD but position unchanged
                    wall_blocked = (
                        action == Action.MOVE_FORWARD
                        and tuple(pos) == prev_pos
                    )

                    if wall_blocked:
                        dash.add_log(
                            f"Action: [bold]{action.name}[/bold]"
                            f" | Reward: {reward:.1f}"
                            f" | [bold red]Wall Blocked[/bold red]"
                        )
                    else:
                        log_parts = [
                            f"[bold]{action.name}[/bold]",
                            f"→ ({pos[0]},{pos[1]}) {dir_label}",
                            f"HP:{hp}",
                        ]

                        if reward != -0.1:
                            log_parts.append(
                                f"[bold yellow]R={reward:+.1f}[/bold yellow]"
                            )

                        if ps["inventory"]:
                            log_parts.append(f"🎒 {ps['inventory']}")

                        dash.add_log("  ".join(log_parts))

                    # ── Detect special events ──
                    if terminated and hp > 0:
                        engine_state = "GOAL_REACHED"
                        dash.add_log(
                            "[bold green]🏆 GOAL REACHED![/bold green]"
                            f"  Steps: {step_count}  "
                            f"Final HP: {hp}"
                        )
                    elif terminated or hp <= 0:
                        engine_state = "GAME_OVER"
                        dash.add_log(
                            "[bold red]💀 GAME OVER[/bold red]"
                            f"  HP depleted at step {step_count}"
                        )
                    elif truncated:
                        engine_state = "TRUNCATED"
                        dash.add_log(
                            "[bold yellow]⏱ Episode truncated[/bold yellow]"
                            f" at max steps ({step_count})"
                        )

                    # ── Refresh display ──
                    live.update(dash.generate_layout(
                        obs_dict=obs,
                        step_count=step_count,
                        engine_state=engine_state,
                    ))

                # Prevent 100% CPU utilization
                time.sleep(0.01)

    except KeyboardInterrupt:
        pass
    finally:
        # Strictly restore terminal to prevent session corruption
        termios.tcsetattr(fd, termios.TCSADRAIN, original_term)

    # Post-game summary
    console.print()
    console.rule("[bold cyan]Session Summary[/bold cyan]")
    console.print(f"  [bold]Engine State:[/bold]  {engine_state}")
    console.print(f"  [bold]Total Steps:[/bold]   {step_count}")
    console.print(f"  [bold]Final HP:[/bold]      {ps['health']}")
    console.print(f"  [bold]Inventory:[/bold]     {ps['inventory']}")
    console.print(f"  [bold]Explored:[/bold]      {info['explored_pct'] * 100:.1f}%")
    console.rule()
    console.print()

    return 0


if __name__ == "__main__":
    sys.exit(main())
