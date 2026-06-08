import os
import time
import threading
import random

import gymnasium as gym
import gymnasium_robotics
gym.register_envs(gymnasium_robotics)

import numpy as np
import torch

from flask import Flask, request, render_template, jsonify
from moviepy.editor import ImageSequenceClip

from cas4160.agents.pg_agent import PGAgent
from cas4160.infrastructure import pytorch_util as ptu
from cas4160.infrastructure import utils
from cas4160.infrastructure.replay_buffer import ReplayBuffer
from cas4160.infrastructure.logger import Logger


# === Annotation System ==================================================================
app = Flask(__name__)
UPLOAD_FOLDER = os.path.join("static", "uploads")
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
feedback_data = {}
feedback_event = threading.Event()

# Global variables
traj_list = []
video_1, video_2 = None, None
annotated_traj_list = []
annotate_per_step = 4
global_itr = 0
max_itr = 100


@app.route('/')
def index():
    global video_1, video_2, traj_list, annotated_traj_list, annotate_per_step, global_itr, max_itr
    ## Check we have more than 2 videos
    if global_itr == max_itr:
        return render_template('index.html', done=True)
    elif len(traj_list) < 2:
        return render_template('index.html', video1=None, video2=None, iteration=global_itr, max_itr=max_itr)
    ## Get two vidoes and show
    video_1, video_2 = random.sample(traj_list, 2)
    video_1_name, video_2_name = video_1[0], video_2[0]
    return render_template('index.html', video1=video_1_name, video2=video_2_name, iteration=global_itr, max_itr=max_itr, total_step=annotate_per_step, current_step=len(annotated_traj_list))


@app.route('/feedback', methods=['POST'])
def feedback():
    global video_1, video_2, annotated_traj_list, global_itr, max_itr
    if global_itr == max_itr:
        feedback_event.set()
        return jsonify({"status": "close", "message": "Feedback received."})
    data = request.get_json()
    selected_video = data.get('selected_video')
    feedback_data['selected_video'] = selected_video
    annotated_traj_list.append((video_1[1], video_2[1], float(selected_video)))
    video_1, video_2 = None, None
    if len(annotated_traj_list) == annotate_per_step:
        feedback_event.set()
    return jsonify({"status": "success", "message": "Feedback received."})


def run_flask():
    app.run(debug=False, use_reloader=False, host='0.0.0.0', port=5000)


def save_videos_to_upload(trajs, prefix="video", max_videos=100, fps=30):
    global traj_list
    os.makedirs(UPLOAD_FOLDER, exist_ok=True)
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".mp4"):
            os.remove(os.path.join(UPLOAD_FOLDER, file))

    for i, traj in enumerate(trajs[:max_videos]):
        frames = traj["image_obs"][-int(fps*2):]
        filename = os.path.join(UPLOAD_FOLDER, f"{prefix}_{i}.mp4")
        clip = ImageSequenceClip(list(frames), fps=fps)
        clip.write_videofile(filename, verbose=False, logger=None)
        traj_list.append([f"{prefix}_{i}.mp4", traj])


def save_videos_to_logs(dir, trajs, prefix="video", max_videos=100, fps=30):
    os.makedirs(dir, exist_ok=True)
    for i, traj in enumerate(trajs[:max_videos]):
        frames = traj["image_obs"][:]
        clip = ImageSequenceClip(list(frames), fps=fps)
        filename = os.path.join(dir, f"{prefix}_{i}.mp4")
        clip.write_videofile(filename, verbose=False, logger=None)
# === End Annotation System ====================================================================


def run_training_loop(args):
    global traj_list, annotated_traj_list, feedback_event, annotate_per_step, global_itr, max_itr
    max_itr = args.n_iter

    logger = Logger(args.logdir)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    ptu.init_gpu(use_gpu=not args.no_gpu, gpu_id=args.which_gpu)

    if args.env_name in ['PointMaze_UMazeDense-v3']:
        example_map =  [
            [1,  1,  1, 1, 1, 1],
            [1, 'g', 0, 0, 0, 1],
            [1,  1,  1, 0, 1, 1],
            [1,  0,  0, 0, 1, 1],
            [1,  0,  0, 0, 1, 1],
            [1,  0,  0,'r',1, 1],
            [1,  1,  1, 1, 1, 1],]
        env = gym.make(args.env_name, maze_map=example_map, render_mode="rgb_array", width=64, height=64)
        eval_env = gym.make(args.env_name, maze_map=example_map, render_mode="rgb_array", width=256, height=256)
        ob_dim = env.observation_space['observation'].shape[0]
        ac_dim = env.action_space.shape[0]
        discrete = False
        robotics = True
        fps = 50
    else:
        if args.env_name in ['Hopper-v5', 'Ant-v5']:
            env = gym.make(args.env_name, render_mode="rgb_array", width=64, height=64, terminate_when_unhealthy=False)
            eval_env = gym.make(args.env_name, render_mode="rgb_array", width=256, height=256, terminate_when_unhealthy=False)
        else:
            env = gym.make(args.env_name, render_mode="rgb_array", width=64, height=64)
            eval_env = gym.make(args.env_name, render_mode="rgb_array", width=256, height=256)
        discrete = isinstance(env.action_space, gym.spaces.Discrete)
        max_ep_len = args.ep_len or env.spec.max_episode_steps
        ob_dim = env.observation_space.shape[0]
        ac_dim = env.action_space.n if discrete else env.action_space.shape[0]
        robotics = False
        fps = 30

    max_ep_len = args.ep_len

    agent = PGAgent(
        ob_dim,
        ac_dim,
        discrete,
        n_layers=args.n_layers,
        layer_size=args.layer_size,
        gamma=args.discount,
        learning_rate=args.learning_rate,
        use_baseline=args.use_baseline,
        use_reward_to_go=args.use_reward_to_go,
        normalize_advantages=args.normalize_advantages,
        baseline_learning_rate=args.baseline_learning_rate,
        baseline_gradient_steps=args.baseline_gradient_steps,
        gae_lambda=args.gae_lambda,
        use_ppo=args.use_ppo,
        n_ppo_epochs=args.n_ppo_epochs,
        n_ppo_minibatches=args.n_ppo_minibatches,
        ppo_cliprange=args.ppo_cliprange,
    )

    replay_buffer = ReplayBuffer()
    total_envsteps = 0
    start_time = time.time()

    for itr in range(args.n_iter):
        global_itr = itr
        if itr == 0:
            annotate_per_step = args.init_annotate_step
        else:
            annotate_per_step = args.annotate_step

        print(f"\n********** Iteration {itr} ************")
        ## Collect trajectories with videos for feedback
        if itr % args.annotate_freq == 0:
            print("\nCollecting video rollouts for feedback...")
            trajs, envsteps_this_batch = utils.rollout_trajectories(
                env, agent.actor, args.batch_size, max_ep_len, render=True, robotics=robotics
            )
            total_envsteps += envsteps_this_batch
            save_videos_to_upload(trajs, prefix=f"itr{itr}", fps=fps)
            print("✨ Waiting for user feedback...")

            if args.syn_prefs:
                for _ in range(annotate_per_step):
                    # Annotate trajectory not from human feedback but from the reward value
                    # Compare the last reward value and give synthetic preferences
                    video_1, video_2 = random.sample(traj_list, 2)
                    selected_video = 0.0 if video_1[1]['reward'][-1] > video_2[1]['reward'][-1] else 1.0
                    annotated_traj_list.append((video_1[1], video_2[1], selected_video))
            else:
                feedback_event.clear()
                feedback_event.wait()

            for annotated_traj in annotated_traj_list:
                # TODO: Store preference triplet to replay buffer
                # make sure to put appropriate values
                # HINT: traj1, traj2 is trajectory from utils.rollout_trajectories(),
                #       and prefs is float value denoting the preference between two trajectories.
                #       (0 means trja1, 1 means traj2, 0.5 means neutral)
                #       replay buffer is initialized above as `replay_buffer`
                #       please check `cas4160/infastructure/replay_buffer.py` for more details
                traj1, traj2, prefs = annotated_traj


            traj_list.clear()  # Clear the traj_list after saving videos
            annotated_traj_list.clear()  # Clear the annotated trajectories

            num_update_reward_predictor = 100
            batch_size_reward_predictor = 16
            reward_loss = []
            accuracies = []
            for _ in range(num_update_reward_predictor):
                # TODO: Train reward predictor
                # train reward predictor sampled from replay buffer
                # sample `batch_size_reward_predictor` and train reward_predictor
                # repeat `num_update_reward_predictor` times.

                batch = None
                loss, accuracy = None

                reward_loss.append(loss)
                accuracies.append(accuracy)
        else:
            trajs, envsteps_this_batch = utils.rollout_trajectories(
                env, agent.actor, args.batch_size, max_ep_len, render=False, robotics=robotics
            )
            total_envsteps += envsteps_this_batch

        # TODO: Replace reward value from reward predictor
        # Hint: make sure you set `training=False`
        for traj in trajs:
            traj["reward"] = None

        trajs_dict = {k: [traj[k] for traj in trajs] for k in trajs[0]}

        # Train PPO
        train_info: dict = agent.update(
            trajs_dict["observation"],
            trajs_dict["action"],
            trajs_dict["reward"],
            trajs_dict["terminal"],
        )

        if itr % args.scalar_log_freq == 0:
            log_video = itr % args.video_log_freq == 0
            eval_trajs, _ = utils.rollout_trajectories(
                eval_env, agent.actor, args.eval_batch_size, max_ep_len, robotics=robotics, render=log_video
            )

            # TODO: Update evaluation reward for evaluation logging
            for traj in eval_trajs:
                traj["reward"] = None

            logs = utils.compute_metrics(trajs, eval_trajs)
            logs.update(train_info)
            logs["Train_EnvstepsSoFar"] = total_envsteps
            logs["TimeSinceStart"] = time.time() - start_time
            if itr == 0:
                logs["Initial_DataCollection_AverageReturn"] = logs["Train_AverageReturn"]

            for key, value in logs.items():
                print(f"{key} : {value}")
                logger.log_scalar(value, key, itr)
            logger.flush()

            if log_video:
                logger.log_trajs_as_videos(
                    eval_trajs,
                    itr,
                    fps=fps,
                    max_videos_to_save=4,
                    video_title="eval_rollouts",
                )


    ##### Finished ###################################
    # Delete videos, wait for user input
    for file in os.listdir(UPLOAD_FOLDER):
        if file.endswith(".mp4"):
            os.remove(os.path.join(UPLOAD_FOLDER, file))
    global_itr = args.n_iter
    if not args.syn_prefs:
        feedback_event.clear()
        feedback_event.wait()
    print("Finish button clicked")
    time.sleep(0.1)


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--env_name", type=str, required=True)
    parser.add_argument("--exp_name", type=str, required=True)
    parser.add_argument("--n_iter", "-n", type=int, default=200)

    parser.add_argument("--use_reward_to_go", "-rtg", action="store_true")
    parser.add_argument("--use_baseline", action="store_true")
    parser.add_argument("--baseline_learning_rate", "-blr", type=float, default=5e-3)
    parser.add_argument("--baseline_gradient_steps", "-bgs", type=int, default=5)
    parser.add_argument("--gae_lambda", type=float, default=None)
    parser.add_argument("--normalize_advantages", "-na", action="store_true")
    parser.add_argument(
        "--batch_size", "-b", type=int, default=1024
    )  # steps collected per train iteration
    parser.add_argument(
        "--eval_batch_size", "-eb", type=int, default=512
    )  # steps collected per eval iteration

    parser.add_argument("--discount", type=float, default=1.0)
    parser.add_argument("--learning_rate", "-lr", type=float, default=5e-3)
    parser.add_argument("--n_layers", "-l", type=int, default=2)
    parser.add_argument("--layer_size", "-s", type=int, default=64)

    parser.add_argument(
        "--ep_len", type=int
    )  # students shouldn't change this away from env's default
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--no_gpu", "-ngpu", action="store_true")
    parser.add_argument("--which_gpu", "-gpu_id", default=0)
    parser.add_argument("--video_log_freq", type=int, default=-1)
    parser.add_argument("--scalar_log_freq", type=int, default=1)

    # arguments for PPO
    parser.add_argument("--use_ppo", action="store_true")
    parser.add_argument("--n_ppo_epochs", type=int, default=4)
    parser.add_argument("--n_ppo_minibatches", type=int, default=4)
    parser.add_argument("--ppo_cliprange", type=float, default=0.2)

    parser.add_argument("--syn_prefs", action="store_true")
    parser.add_argument("--annotate_freq", type=int, default=1)
    parser.add_argument("--init_annotate_step", type=int, default=16)
    parser.add_argument("--annotate_step", type=int, default=8)

    args = parser.parse_args()

    logdir_prefix = "hw7_"
    data_path = os.path.join(os.path.dirname(os.path.realpath(__file__)), "../../data")
    if not os.path.exists(data_path):
        os.makedirs(data_path)

    logdir = os.path.join(
        data_path,
        logdir_prefix + args.exp_name + "_" + args.env_name + "_" + time.strftime("%d-%m-%Y_%H-%M-%S")
    )
    os.makedirs(logdir, exist_ok=True)
    args.logdir = logdir

    flask_thread = threading.Thread(target=run_flask, daemon=True)
    flask_thread.start()

    run_training_loop(args)

if __name__ == "__main__":
    main()
