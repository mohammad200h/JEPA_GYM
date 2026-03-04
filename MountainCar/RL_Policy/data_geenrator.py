from pathlib import Path

from stable_baselines3 import PPO
import gymnasium as gym
import numpy as np
import os
import h5py
from dotenv import load_dotenv
from tqdm import tqdm
import dropbox
from dropbox.files import WriteMode, UploadSessionCursor, CommitInfo


# .env in repo root (JEPA_GYM), 3 levels up from this file
_ENV_PATH = Path(__file__).resolve().parent.parent.parent / ".env"
print(f"ENV_PATH: {_ENV_PATH}")

# Chunk size for streaming upload (4 MB) — keeps RAM low and matches Dropbox recommendation
_DROPBOX_CHUNK_SIZE = 4 * 1024 * 1024


def push_dataset_to_dropbox(db_path: str):
    """Upload the HDF5 dataset at db_path to Dropbox (path /dataset/<filename>), streaming in chunks to limit RAM/CPU."""
    load_dotenv(_ENV_PATH)
    token = os.getenv("DropBOX_OAuth2") or os.getenv("DROPBOX_OAUTH2")
    if not token:
        raise ValueError("Dropbox OAuth2 token not found. Set DropBOX_OAuth2 or DROPBOX_OAUTH2 in .env")
    dbx = dropbox.Dropbox(oauth2_access_token=token)
    path = Path(db_path)
    if not path.is_file():
        raise FileNotFoundError(f"Dataset file not found: {db_path}")
    dropbox_path = f"/dataset/{path.name}"
    file_size = path.stat().st_size

    with open(path, "rb") as f:
        with tqdm(total=file_size, unit="B", unit_scale=True, desc="Uploading to Dropbox") as pbar:
            if file_size <= _DROPBOX_CHUNK_SIZE:
                # Small file: single upload
                dbx.files_upload(f.read(), dropbox_path, mode=WriteMode("overwrite"))
                pbar.update(file_size)
            else:
                # Large file: streaming upload session (only one chunk in RAM at a time)
                first = f.read(_DROPBOX_CHUNK_SIZE)
                session = dbx.files_upload_session_start(first)
                cursor = UploadSessionCursor(session_id=session.session_id, offset=len(first))
                pbar.update(len(first))

                while True:
                    chunk = f.read(_DROPBOX_CHUNK_SIZE)
                    if len(chunk) < _DROPBOX_CHUNK_SIZE:
                        break
                    dbx.files_upload_session_append_v2(chunk, cursor)
                    cursor.offset += len(chunk)
                    pbar.update(len(chunk))

                dbx.files_upload_session_finish(chunk, cursor, CommitInfo(path=dropbox_path, mode=WriteMode("overwrite")))
                pbar.update(len(chunk))

    print(f"Uploaded {db_path} -> Dropbox{dropbox_path}")


def generate_dataset(db_path: str,num_episodes: int = 10 ):
    

    env = gym.make("MountainCarContinuous-v0", render_mode="rgb_array")

    with h5py.File(db_path, "w") as db:
        for model_path in os.listdir("policy_checkpoints"):
            if not "zip" in model_path:
                continue
            print(f"model_path: {model_path}")
            path = os.path.join("policy_checkpoints", model_path)
            model = PPO.load(path)

            model_name = model_path.replace(".zip", "")
            model_grp = db.create_group(model_name)

            for episode in range(num_episodes):
                eps_grp = model_grp.create_group(f"eps_{episode}")

                episode_obs = []
                episode_actions = []
                episode_rewards = []
                episode_terminated = []
                episode_truncated = []
                episode_infos = []
                episode_frames = []

                obs, info = env.reset()
                done = False
                while not done:
                    action, _ = model.predict(obs, deterministic=True)
                    next_obs, reward, terminated, truncated, info = env.step(action)
                    frame = env.render()  # RGB array (H, W, 3), uint8

                    episode_obs.append(obs)
                    episode_actions.append(action)
                    episode_rewards.append(reward)
                    episode_terminated.append(terminated)
                    episode_truncated.append(truncated)
                    episode_infos.append(info)
                    episode_frames.append(frame)

                    obs = next_obs
                    done = terminated or truncated

                # Store datasets inside this episode group
                eps_grp.create_dataset("obs", data=np.array(episode_obs))
                eps_grp.create_dataset("action", data=np.array(episode_actions))
                eps_grp.create_dataset("reward", data=np.array(episode_rewards))
                eps_grp.create_dataset("terminated", data=np.array(episode_terminated))
                eps_grp.create_dataset("truncated", data=np.array(episode_truncated))
                info_strs = np.array([str(i) for i in episode_infos], dtype="S")
                eps_grp.create_dataset("info", data=info_strs)
                eps_grp.create_dataset("frame", data=np.array(episode_frames))

    env.close()

def main(num_episodes: int = 10):

    # Single HDF5 database with group hierarchy: model_name / eps_{N} / datasets
    dataset_dir = "dataset"
    os.makedirs(dataset_dir, exist_ok=True)
    db_path = os.path.join(dataset_dir, "dataset.h5")
    

    push_dataset_to_dropbox(db_path)
 

 


if __name__ == "__main__":
    import sys
    num_episodes = 10
    if len(sys.argv) > 1:
        try:
            num_episodes = int(sys.argv[1])
        except ValueError:
            pass  # keep default 10
    main(num_episodes)