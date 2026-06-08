export MUJOCO_GL=egl

cd cas4160/scripts
python run_hw7.py --env_name Hopper-v5 --ep_len 150 --discount 0.99 -lr 0.003 -n 100 -b 4000 --use_reward_to_go -na --use_baseline --gae_lambda 0.97 --use_ppo --n_ppo_epochs 4 --n_ppo_minibatches 4 --exp_name RLHF-Hopper --video_log_freq 5 --annotate_freq=5 --init_annotate_step 48 --annotate_step 8