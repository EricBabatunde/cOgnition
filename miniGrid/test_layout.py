import io
from rich.console import Console
from environment import CustomRPGEnv, Action
from ui.terminal_dashboard import TerminalDashboard

env = CustomRPGEnv()
obs, _ = env.reset(seed=42)
dash = TerminalDashboard()
dash.add_log('Env reset — spawn (1,1) EAST')
dash.add_log('[bold]MOVE_FORWARD[/bold] → (1,2) ►E HP:100')
dash.add_log('[bold red]⚠ HAZARD[/bold red] HP:80')
dash.add_log('Querying Graph: Key_Red → (2,2)')

buf = io.StringIO()
Console(file=buf, width=80, force_terminal=True).print(
    dash.generate_layout(
        obs_dict=obs,
        fast_mem_info={'active_goal':'Get Key','faiss_match':'R1','faiss_distance':0.51,'loop_detected':False,'buffer_length':4},
        logic_info={'safe_to_step':True,'active_rules':['wall_blocking','hazard_avoidance'],'forbidden_actions':[]},
        step_count=3,
        engine_state='EXPLORING',
    )
)
rendered = buf.getvalue()
lines = rendered.rstrip('\n').split('\n')
print(f'RENDERED HEIGHT: {len(lines)} lines')
print(f'TARGET MAX:      18 lines')
for i, l in enumerate(lines, 1):
    print(f'{i:2d}│ {l}')
