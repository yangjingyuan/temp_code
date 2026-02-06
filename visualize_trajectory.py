"""Streamlit app for visualizing tool_gui_agent trajectory data.

Usage:
    streamlit run visualize_trajectory.py
"""

import gzip
import json
import pickle
from typing import Any, Dict, List, Optional

import numpy as np
import streamlit as st
from PIL import Image, ImageDraw


def load_trajectory(file_path: str) -> List[Dict[str, Any]]:
    """Load trajectory from pkl.gz file."""
    with gzip.open(file_path, 'rb') as f:
        return pickle.load(f)


def draw_action_marker(
    image: Image.Image,
    action_json: Optional[Dict[str, Any]],
    screen_size: tuple = None,
) -> Image.Image:
    """Draw action marker on the image.

    Args:
        image: PIL Image to draw on.
        action_json: Action dict with coordinate info.
        screen_size: (width, height) for coordinate conversion.

    Returns:
        Image with action marker drawn.
    """
    if not action_json:
        return image

    image = image.copy()
    draw = ImageDraw.Draw(image)
    width, height = image.size

    action_type = action_json.get('action', '')
    coord = action_json.get('coordinate')

    if coord and len(coord) >= 2:
        # Coordinates are normalized [0, 1]
        x = int(coord[0] * width)
        y = int(coord[1] * height)
        radius = 20

        # Draw different markers based on action type
        if action_type == 'click':
            # Red circle for click
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline='red',
                width=4,
            )
            draw.ellipse(
                [x - 5, y - 5, x + 5, y + 5],
                fill='red',
            )
        elif action_type == 'long_press':
            # Orange circle for long press
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline='orange',
                width=4,
            )
            draw.ellipse(
                [x - radius // 2, y - radius // 2, x + radius // 2, y + radius // 2],
                outline='orange',
                width=2,
            )
        elif action_type == 'double_click':
            # Purple double circle
            draw.ellipse(
                [x - radius, y - radius, x + radius, y + radius],
                outline='purple',
                width=4,
            )
            draw.ellipse(
                [x - radius - 8, y - radius - 8, x + radius + 8, y + radius + 8],
                outline='purple',
                width=2,
            )

    # Handle swipe/drag with start and end coordinates
    start_coord = action_json.get('start_coordinate')
    end_coord = action_json.get('end_coordinate')
    if start_coord and end_coord:
        x1 = int(start_coord[0] * width)
        y1 = int(start_coord[1] * height)
        x2 = int(end_coord[0] * width)
        y2 = int(end_coord[1] * height)
        # Draw arrow
        draw.line([x1, y1, x2, y2], fill='blue', width=4)
        draw.ellipse([x1 - 10, y1 - 10, x1 + 10, y1 + 10], fill='green')
        draw.ellipse([x2 - 10, y2 - 10, x2 + 10, y2 + 10], fill='red')

    # Handle swipe with direction
    if action_type == 'swipe' and coord:
        direction = action_json.get('direction', '')
        x = int(coord[0] * width)
        y = int(coord[1] * height)
        arrow_len = 60

        if direction == 'up':
            x2, y2 = x, y - arrow_len
        elif direction == 'down':
            x2, y2 = x, y + arrow_len
        elif direction == 'left':
            x2, y2 = x - arrow_len, y
        elif direction == 'right':
            x2, y2 = x + arrow_len, y
        else:
            x2, y2 = x, y

        draw.line([x, y, x2, y2], fill='blue', width=4)
        draw.ellipse([x - 8, y - 8, x + 8, y + 8], fill='blue')

    return image


def format_action_display(action_json: Optional[Dict[str, Any]]) -> str:
    """Format action JSON for display."""
    if not action_json:
        return "No action"
    return json.dumps(action_json, indent=2, ensure_ascii=False)


def main():
    st.set_page_config(
        page_title="Trajectory Visualizer",
        layout="wide",
    )

    st.title("🔍 Trajectory Visualizer")

    # Sidebar for file loading and navigation
    with st.sidebar:
        st.header("📁 Load Trajectory")

        # File path input
        file_path = st.text_input(
            "File path",
            value="trajectory.pkl.gz",
            help="Path to the .pkl.gz trajectory file",
        )

        # Load button
        if st.button("Load", type="primary"):
            try:
                trajectory = load_trajectory(file_path)
                st.session_state.trajectory = trajectory
                st.session_state.file_path = file_path
                st.success(f"Loaded {len(trajectory)} steps")
            except FileNotFoundError:
                st.error(f"File not found: {file_path}")
            except Exception as e:
                st.error(f"Error loading file: {e}")

        # Step navigation
        if 'trajectory' in st.session_state:
            st.divider()
            st.header("🎯 Navigation")

            trajectory = st.session_state.trajectory
            total_steps = len(trajectory)

            step_idx = st.slider(
                "Step",
                min_value=0,
                max_value=total_steps - 1,
                value=st.session_state.get('step_idx', 0),
                key='step_slider',
            )
            st.session_state.step_idx = step_idx

            st.caption(f"Step {step_idx + 1} / {total_steps}")

            # Quick navigation buttons
            col1, col2 = st.columns(2)
            with col1:
                if st.button("⬅️ Prev", disabled=step_idx == 0):
                    st.session_state.step_idx = step_idx - 1
                    st.rerun()
            with col2:
                if st.button("Next ➡️", disabled=step_idx == total_steps - 1):
                    st.session_state.step_idx = step_idx + 1
                    st.rerun()

    # Main content
    if 'trajectory' not in st.session_state:
        st.info("👈 Please load a trajectory file from the sidebar")
        return

    trajectory = st.session_state.trajectory
    step_idx = st.session_state.get('step_idx', 0)
    step_data = trajectory[step_idx]

    # Two columns: screenshot and info
    col1, col2 = st.columns([1, 1])

    with col1:
        st.subheader(f"📱 Screenshot (Step {step_idx + 1})")

        raw_screenshot = step_data.get('raw_screenshot')
        if raw_screenshot is not None:
            # Convert numpy array to PIL Image
            if isinstance(raw_screenshot, np.ndarray):
                img = Image.fromarray(raw_screenshot)
            else:
                img = raw_screenshot

            # Draw action marker
            action_json = step_data.get('action_json')
            img_with_marker = draw_action_marker(img, action_json)

            st.image(img_with_marker, use_container_width=True)
        else:
            st.warning("No screenshot available")

    with col2:
        # Action info
        st.subheader("🎬 Action")
        action_json = step_data.get('action_json')
        if action_json:
            action_type = action_json.get('action', 'unknown')
            st.markdown(f"**Type:** `{action_type}`")
            st.code(format_action_display(action_json), language='json')
        else:
            st.info("No action")

        # Summary
        st.subheader("📝 Summary")
        summary = step_data.get('summary', '')
        if summary:
            st.info(summary)
        else:
            st.caption("No summary")

        # Thinking
        st.subheader("🧠 Thinking")
        thinking = step_data.get('thinking', '')
        if thinking:
            st.text_area(
                "Model reasoning",
                value=thinking,
                height=150,
                disabled=True,
                label_visibility="collapsed",
            )
        else:
            st.caption("No thinking content")

    # Expandable sections for detailed info
    st.divider()

    with st.expander("📤 LLM Output (Raw Response)"):
        action_output = step_data.get('action_output', '')
        if action_output:
            st.code(action_output, language=None)
        else:
            st.caption("No output")

    with st.expander("📋 Action Prompt"):
        action_prompt = step_data.get('action_prompt')
        if action_prompt:
            st.json(action_prompt)
        else:
            st.caption("No prompt")

    with st.expander("🏗️ UI Elements"):
        ui_elements = step_data.get('ui_elements')
        if ui_elements:
            st.write(f"Total elements: {len(ui_elements)}")
            for i, elem in enumerate(ui_elements[:50]):  # Show first 50
                st.text(f"{i}: {elem}")
            if len(ui_elements) > 50:
                st.caption(f"... and {len(ui_elements) - 50} more")
        else:
            st.caption("No UI elements")


if __name__ == "__main__":
    main()
