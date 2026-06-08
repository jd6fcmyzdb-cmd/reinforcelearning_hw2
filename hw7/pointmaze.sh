export MUJOCO_GL=egl

cd cas4160/scripts
python run_hw7.py --env_name PointMaze_UMazeDense-v3 --ep_len 150 --discount 0.99 -lr 0.003 -n 21 -b 2000 --use_reward_to_go -na --use_baseline --gae_lambda 0.97 --use_ppo --n_ppo_epochs 4 --n_ppo_minibatches 4 --exp_name RLHF-PointMaze --video_log_freq 4 --annotate_freq=2 --init_annotate_step 8 --annotate_step 4
