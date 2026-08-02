from environment.custom_rpg_env import CustomRPGEnv
env = CustomRPGEnv(tier=3)
env.reset(seed=42)
ent = env.unwrapped._entities.get((4, 9))
print("Entity:", ent)
print("Is locked:", ent.is_locked if ent else None)
