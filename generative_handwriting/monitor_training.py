import json
import os
import sys
import time
from datetime import datetime

import matplotlib
import numpy as np
import tensorflow as tf

matplotlib.use("Agg")  # Use non-interactive backend for remote servers
import matplotlib.pyplot as plt
from loader import HandwritingDataLoader
from model.handwriting_models import DeepHandwritingPredictionModel
from model.mixture_density_network import MixtureDensityLayer, mdn_loss


def load_model_safe(model_path: str, num_components: int = 20):
    """Safely load model with custom objects."""
    try:
        model = tf.keras.models.load_model(
            model_path,
            custom_objects={
                "model_mdn_loss": lambda actual, outputs, combined_train_lengths, num_mixture_components: mdn_loss(
                    actual, outputs, combined_train_lengths, num_mixture_components
                ),
                "MixtureDensityLayer": MixtureDensityLayer,
                "DeepHandwritingPredictionModel": DeepHandwritingPredictionModel,
            },
        )
        return model
    except:
        return None


def get_model_info(model_dir: str):
    """Get information about the saved model."""
    model_path = os.path.join(model_dir, "best_model.keras")
    epochs_info_path = os.path.join(model_dir, "epochs_info.json")
    nan_report_path = os.path.join(model_dir, "nan_report.json")

    info = {
        "model_exists": os.path.exists(model_path),
        "model_size": 0,
        "last_modified": None,
        "epochs_trained": 0,
        "nan_issues": False,
    }

    if info["model_exists"]:
        info["model_size"] = os.path.getsize(model_path) / (1024 * 1024)  # MB
        info["last_modified"] = datetime.fromtimestamp(os.path.getmtime(model_path))

    if os.path.exists(epochs_info_path):
        with open(epochs_info_path, "r") as f:
            info["epochs_trained"] = json.load(f)

    if os.path.exists(nan_report_path):
        info["nan_issues"] = True

    return info


def quick_test_model(model_path: str, data_loader: HandwritingDataLoader):
    """Run a quick test of the model's predictions."""
    model = load_model_safe(model_path)
    if model is None:
        return None

    # Get a test sample
    test_stroke = data_loader.combined_test_strokes[0][:20]  # First 20 points
    test_input = test_stroke.reshape(1, 20, 3)

    # Get prediction
    prediction = model(test_input, training=False)

    # Extract parameters from last timestep
    last_pred = prediction[0, -1, :].numpy()

    # Extract pi (mixture weights) - first 20 values with softmax
    pi = tf.nn.softmax(last_pred[:20]).numpy()

    # Get end-of-stroke probability (last value with sigmoid)
    eos = tf.sigmoid(last_pred[-1]).numpy()

    return {
        "max_mixture_weight": float(pi.max()),
        "active_components": int(np.sum(pi > 0.01)),
        "entropy": float(-np.sum(pi * np.log(pi + 1e-10))),
        "eos_probability": float(eos),
        "prediction_shape": prediction.shape,
    }


def plot_training_progress(model_dir: str):
    """Create a plot showing training progress from TensorBoard logs."""
    import glob

    # Look for TensorBoard event files
    log_pattern = os.path.join(model_dir, "../logs/handwriting_synthesis/profile/*/events.out.tfevents.*")
    log_files = glob.glob(log_pattern)

    fig, ax = plt.subplots(figsize=(12, 8))

    if log_files:
        try:
            # Try to read TensorBoard logs for loss values
            from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

            # Get the most recent log file
            latest_log = max(log_files, key=os.path.getctime)
            ea = EventAccumulator(latest_log)
            ea.Reload()

            # Get available scalar summaries
            scalar_keys = ea.Tags()["scalars"]

            if "epoch_loss" in scalar_keys or "loss" in scalar_keys:
                loss_key = "epoch_loss" if "epoch_loss" in scalar_keys else "loss"
                loss_events = ea.Scalars(loss_key)
                steps = [event.step for event in loss_events]
                losses = [event.value for event in loss_events]

                ax.plot(steps, losses, "b-", linewidth=2, label="Training Loss")
                ax.set_xlabel("Step")
                ax.set_ylabel("Loss")
                ax.set_title("Synthesis Model Training Progress")
                ax.legend()
                ax.grid(True, alpha=0.3)

                # Add recent loss info
                if losses:
                    recent_loss = losses[-1]
                    ax.text(
                        0.02,
                        0.98,
                        f"Latest Loss: {recent_loss:.4f}",
                        transform=ax.transAxes,
                        va="top",
                        bbox=dict(boxstyle="round", facecolor="lightblue", alpha=0.8),
                    )
            else:
                ax.text(
                    0.5,
                    0.5,
                    f"TensorBoard logs found but no loss data\nAvailable keys: {scalar_keys}",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                    fontsize=12,
                )
                ax.set_title("Synthesis Model Training Monitor")
                ax.axis("off")

        except Exception as e:
            ax.text(
                0.5,
                0.5,
                f"Error reading TensorBoard logs:\n{str(e)}",
                ha="center",
                va="center",
                transform=ax.transAxes,
                fontsize=12,
            )
            ax.set_title("Synthesis Model Training Monitor")
            ax.axis("off")
    else:
        ax.text(
            0.5,
            0.5,
            "No TensorBoard logs found yet\nTraining may not have started",
            ha="center",
            va="center",
            transform=ax.transAxes,
            fontsize=14,
        )
        ax.set_title("Synthesis Model Training Monitor")
        ax.axis("off")

    plot_path = os.path.join(model_dir, "training_progress.png")
    plt.savefig(plot_path, dpi=100, bbox_inches="tight")
    plt.close()
    return plot_path


def main():
    """Monitor training progress and model status."""
    curr_directory = os.path.dirname(os.path.realpath(__file__))

    # Check for synthesis model first (primary focus)
    synthesis_dir = f"{curr_directory}/saved/models/handwriting_synthesis/"
    prediction_dir = f"{curr_directory}/saved_models/full_handwriting_prediction/"

    # Determine which model to monitor
    if os.path.exists(synthesis_dir) and any(os.listdir(synthesis_dir)):
        model_dir = synthesis_dir
        model_type = "SYNTHESIS"
    elif os.path.exists(prediction_dir):
        model_dir = prediction_dir
        model_type = "PREDICTION"
    else:
        model_dir = synthesis_dir  # Default for new training
        model_type = "SYNTHESIS"

    print("=" * 60)
    print(f"HANDWRITING {model_type} MODEL TRAINING MONITOR")
    print("=" * 60)
    print(f"📁 Monitoring: {model_dir}")
    print("-" * 60)

    # Get model info
    info = get_model_info(model_dir)

    print("\n📊 MODEL STATUS:")
    print("-" * 40)
    if info["model_exists"]:
        print(f"✅ Model found: {info['model_size']:.2f} MB")
        print(f"📅 Last updated: {info['last_modified']}")
        print(f"🔢 Epochs trained: {info['epochs_trained']}")
    else:
        print("❌ No saved model found")

    if info["nan_issues"]:
        print("⚠️  NaN issues detected during training")

    # Load data for testing
    if info["model_exists"]:
        print("\n🧪 RUNNING QUICK MODEL TEST...")
        print("-" * 40)

        data_loader = HandwritingDataLoader()
        data_loader.prepare_data()

        model_path = os.path.join(model_dir, "best_model.keras")
        test_results = quick_test_model(model_path, data_loader)

        if test_results:
            print("✅ Model loaded successfully")
            print(f"📈 Output shape: {test_results['prediction_shape']}")
            print(f"🎯 Max mixture weight: {test_results['max_mixture_weight']:.4f}")
            print(f"📊 Active components: {test_results['active_components']}/20")
            print(f"🔀 Distribution entropy: {test_results['entropy']:.4f}")
            print(f"🏁 End-of-stroke prob: {test_results['eos_probability']:.4f}")
        else:
            print("❌ Failed to load model for testing")

    # Create progress plot
    print("\n📈 GENERATING PROGRESS PLOT...")
    print("-" * 40)
    plot_path = plot_training_progress(model_dir)
    print(f"✅ Plot saved to: {plot_path}")

    print("\n" + "=" * 60)
    print("Monitor complete. Run this script periodically to check progress.")

    # If running in a loop (optional)
    if "--watch" in sys.argv:
        print("\n👁️  Watch mode enabled. Checking every 60 seconds...")
        print("Press Ctrl+C to stop")
        try:
            while True:
                time.sleep(60)
                os.system("clear")
                main()
        except KeyboardInterrupt:
            print("\n👋 Monitoring stopped")


if __name__ == "__main__":
    main()
