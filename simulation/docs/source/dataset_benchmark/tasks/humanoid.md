
# Humanoid Support  (Deprecated)
Make sure you have demonstrations (e.g. mabaoguo.pkl) and the policy network checkpoint (e.g. policy_jit.pt) in `data_isaaclab/source_data_isaaclab/humanoid`

Then, run the command to launch the humanoid environment (whole-body control with huamnoid poses):
```bash
python src/scripts/run_policy.py --task humanoid.humanoid_pose --num_envs 4
```
