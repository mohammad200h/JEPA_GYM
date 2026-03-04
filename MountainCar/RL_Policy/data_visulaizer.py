"""
Dataset viewer for MountainCar RL rollout data.
Reads dataset/dataset.h5 (HDF5) and provides a UI to browse by model, episode, and step.
"""
import os
import h5py
import numpy as np
import gradio as gr

DATASET_PATH = os.path.join(os.path.dirname(__file__), "dataset", "dataset.h5")


def get_available_data():
    """Read HDF5 and return list of (model_name, episode_keys) and step counts per episode."""
    if not os.path.isfile(DATASET_PATH):
        return {}, []
    data = {}
    models = []
    with h5py.File(DATASET_PATH, "r") as f:
        for model_name in sorted(f.keys()):
            grp = f[model_name]
            episodes = []
            for key in sorted(grp.keys(), key=lambda x: int(x.split("_")[1])):
                if key.startswith("eps_"):
                    n_steps = grp[key]["obs"].shape[0]
                    episodes.append((key, n_steps))
            data[model_name] = episodes
            models.append(model_name)
    return data, models


def load_step(model_name: str, episode_key: str, step_index: int):
    """Load obs, action, reward, done (terminated|truncated), and frame for one step."""
    out = {
        "obs": None,
        "action": None,
        "reward": None,
        "done": None,
        "terminated": None,
        "truncated": None,
        "frame": None,
        "error": None,
    }
    if not os.path.isfile(DATASET_PATH):
        out["error"] = f"Dataset not found: {DATASET_PATH}"
        return out
    try:
        with h5py.File(DATASET_PATH, "r") as f:
            if model_name not in f or episode_key not in f[model_name]:
                out["error"] = f"Model {model_name} or episode {episode_key} not found."
                return out
            eps = f[model_name][episode_key]
            obs = eps["obs"][step_index]
            action = eps["action"][step_index]
            reward = float(eps["reward"][step_index])
            term = bool(eps["terminated"][step_index])
            trunc = bool(eps["truncated"][step_index])
            frame_arr = np.array(eps["frame"][step_index])
            out["obs"] = obs
            out["action"] = action
            out["reward"] = reward
            out["terminated"] = term
            out["truncated"] = trunc
            out["done"] = term or trunc
            out["frame"] = frame_arr
    except Exception as e:
        out["error"] = str(e)
    return out


def build_ui():
    data, models = get_available_data()
    if not models:
        # No dataset yet: show placeholder UI that explains how to generate data
        def no_data(*args):
            return (
                None,
                "**No dataset found.**\n\nGenerate data first by running:\n\n`python data_geenrator.py [num_episodes]`\n\nThen restart this app.",
                "obs: —\naction: —\nreward: —\ndone: —",
            )

        with gr.Blocks(title="MountainCar Dataset Viewer") as demo:
            gr.Markdown("# MountainCar Dataset Viewer")
            gr.Markdown("Select model, episode, and step to view rollout data.")
            step_image = gr.Image(label="Frame")
            message = gr.Markdown()
            step_info = gr.Markdown()
            gr.Button("Load").click(no_data, outputs=[step_image, message, step_info])
        return demo

    def on_model_change(model_name):
        if not model_name or model_name not in data:
            return gr.update(choices=[], value=None), gr.update(maximum=0, value=0)
        episodes = data[model_name]
        choices = [f"{k} ({n} steps)" for k, n in episodes]
        return gr.update(choices=choices, value=choices[0] if choices else None), gr.update(
            maximum=max((n for _, n in episodes), default=0) - 1, value=0
        )

    def on_episode_change(model_name, episode_choice):
        if not model_name or not episode_choice or model_name not in data:
            return gr.update(maximum=0, value=0)
        episodes = data[model_name]
        for (k, n) in episodes:
            if f"{k} ({n} steps)" == episode_choice:
                return gr.update(maximum=max(0, n - 1), value=0)
        return gr.update(maximum=0, value=0)

    def load_step_ui(model_name, episode_choice, step_index):
        if not model_name or not episode_choice:
            return None, None, "Select model and episode."
        episode_key = episode_choice.split(" ")[0]  # "eps_0 (42 steps)" -> "eps_0"
        step_index = int(step_index)
        loaded = load_step(model_name, episode_key, step_index)
        if loaded["error"]:
            return None, None, f"**Error:** {loaded['error']}"
        done_str = "True" if loaded["done"] else "False"
        term = "terminated" if loaded["terminated"] else ""
        trunc = "truncated" if loaded["truncated"] else ""
        extra = " | ".join(filter(None, [term, trunc]))
        if extra:
            done_str += f" ({extra})"
        info_lines = [
            f"**obs:** `{np.array(loaded['obs']).tolist()}`",
            f"**action:** `{np.array(loaded['action']).tolist()}`",
            f"**reward:** {loaded['reward']}",
            f"**done:** {done_str}",
        ]
        info_md = "\n\n".join(info_lines)
        return (loaded["frame"], None, info_md)

    with gr.Blocks(title="MountainCar Dataset Viewer") as demo:
        gr.Markdown("# MountainCar Dataset Viewer")
        gr.Markdown("Choose **model**, **episode**, and **step** to view obs, action, reward, done, and frame.")
        with gr.Row():
            model_dd = gr.Dropdown(
                label="Model",
                choices=models,
                value=models[0] if models else None,
            )
            episode_dd = gr.Dropdown(
                label="Episode",
                choices=[],
                value=None,
            )
            step_slider = gr.Slider(
                minimum=0,
                maximum=0,
                step=1,
                value=0,
                label="Step",
            )
        with gr.Row():
            step_image = gr.Image(label="Frame")
            step_info = gr.Markdown(label="Step data")
        message = gr.Markdown(visible=False)

        model_dd.change(
            on_model_change,
            inputs=[model_dd],
            outputs=[episode_dd, step_slider],
        )
        episode_dd.change(
            on_episode_change,
            inputs=[model_dd, episode_dd],
            outputs=[step_slider],
        )
        load_btn = gr.Button("Load step")

        load_btn.click(
            load_step_ui,
            inputs=[model_dd, episode_dd, step_slider],
            outputs=[step_image, message, step_info],
        )
        step_slider.release(
            load_step_ui,
            inputs=[model_dd, episode_dd, step_slider],
            outputs=[step_image, message, step_info],
        )

        # Initial state
        demo.load(
            on_model_change,
            inputs=[model_dd],
            outputs=[episode_dd, step_slider],
        )

    return demo


if __name__ == "__main__":
    demo = build_ui()
    demo.launch()
