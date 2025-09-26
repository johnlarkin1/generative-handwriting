#!/usr/bin/env python3
"""
Visualization script for handwriting synthesis model.
Generates sample handwriting from the trained synthesis model and creates various visualizations.
"""

import os

from writer import Calligrapher


def main():
    """Generate handwriting samples and visualizations."""

    # Set up paths
    curr_directory = os.path.dirname(os.path.realpath(__file__))
    model_save_dir = f"{curr_directory}/saved_models/full_handwriting_synthesis/"
    model_load_path = os.path.join(model_save_dir, "best_model.keras")
    output_dir = f"{curr_directory}/visualizations/synthesis/"

    # Create output directory
    os.makedirs(output_dir, exist_ok=True)

    # Check if model exists
    if not os.path.exists(model_load_path):
        print(f"Model not found at {model_load_path}")
        print("Please ensure the model has been copied from the remote server.")
        return

    # Sample texts to generate
    sample_texts = [
        ["Hello world!"],
        ["The quick brown fox"],
        ["jumps over the lazy dog"],
        ["Handwriting synthesis"],
        ["with neural networks"],
        ["is quite fascinating"],
    ]

    try:
        # Create calligrapher
        print(f"Loading model from {model_load_path}")
        calligrapher = Calligrapher(
            model_load_path,
            num_output_mixtures=20,  # Match your model's configuration
        )
        print("Model loaded successfully!")

        # Generate samples with different temperature/bias settings (following Graves 2013 recommendations)
        settings = [
            ("default", 0.5, 0.4, False),  # Balanced settings
            ("clean", 0.3, 0.6, True),  # High bias, low temp, greedy for clean output
            ("creative", 0.7, 0.2, False),  # Higher temp, lower bias for variation
            ("stable", 0.4, 0.5, True),  # Very conservative for legibility
        ]

        for style_name, temp, bias, greedy in settings:
            print(f"Generating {style_name} style samples...")

            for i, texts in enumerate(sample_texts):
                output_filename = os.path.join(output_dir, f"sample_{i + 1}_{style_name}.svg")

                try:
                    # Create SVG with improved sampling parameters
                    calligrapher.write(
                        texts,
                        filename=output_filename,
                        temperature=temp,
                        bias=bias,
                        stroke_colors=["blue"],
                        stroke_widths=[2.0],
                        show_grid=True,
                        show_endpoints=False,
                    )
                    print(f"  Created {output_filename}")

                except Exception as e:
                    print(f"  Error generating {output_filename}: {e}")
                    continue

        # Generate a comparison sheet with all styles for one text
        comparison_text = ["Neural handwriting"]
        print("Generating style comparison...")

        for style_name, temp, bias, greedy in settings:
            output_filename = os.path.join(output_dir, f"comparison_{style_name}.svg")
            try:
                calligrapher.write(
                    comparison_text,
                    filename=output_filename,
                    temperature=temp,
                    bias=bias,
                    stroke_colors=["black"],
                    stroke_widths=[2.5],
                    show_grid=False,
                    show_endpoints=False,
                )
                print(f"  Created {output_filename}")
            except Exception as e:
                print(f"  Error generating comparison {style_name}: {e}")

        print(f"\nAll visualizations saved to: {output_dir}")
        print("\nGenerated files:")
        for file in sorted(os.listdir(output_dir)):
            if file.endswith(".svg"):
                print(f"  - {file}")

    except Exception as e:
        print(f"Error: {e}")
        print("Make sure the model file exists and is compatible.")


if __name__ == "__main__":
    main()
