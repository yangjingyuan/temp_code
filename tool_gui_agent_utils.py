# Copyright 2025 The android_world Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Utility functions for ToolGuiAgent.

This module provides coordinate conversion and action mapping functions
for adapting MAI-UI's action format to android_world's JSONAction.
"""

import base64
import copy
import json
import re
from dataclasses import dataclass, field
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

from PIL import Image

from android_world.env import json_action

# Constants
SCALE_FACTOR = 999  # MAI-UI uses 0-999 coordinate range for LLM


# =============================================================================
# Trajectory Memory Structures (adapted from MAI-UI)
# =============================================================================

@dataclass
class TrajStep:
    """Represents a single step in an agent's trajectory.

    Attributes:
        screenshot: PIL Image of the screen at this step.
        screenshot_bytes: Original screenshot as bytes.
        prediction: Raw model prediction/response.
        action: Parsed action dictionary.
        thought: Model's reasoning/thinking process.
        step_index: Index of this step in the trajectory.
        structured_action: Structured action with metadata.
        ask_user_response: Response from ask_user action.
    """

    screenshot: Image.Image
    screenshot_bytes: bytes
    prediction: str
    action: Dict[str, Any]
    thought: str
    step_index: int
    structured_action: Optional[Dict[str, Any]] = None
    ask_user_response: Optional[str] = None


@dataclass
class TrajMemory:
    """Container for a complete trajectory of agent steps.

    Attributes:
        task_goal: The goal/instruction for this trajectory.
        steps: List of trajectory steps.
    """

    task_goal: str = ''
    steps: List[TrajStep] = field(default_factory=list)

    @property
    def history_images(self) -> List[bytes]:
        """Get screenshot bytes from all trajectory steps."""
        return [step.screenshot_bytes for step in self.steps]


# =============================================================================
# Coordinate Conversion Functions
# =============================================================================

def normalized_to_pixel(
    coord: List[float],
    screen_size: Tuple[int, int],
) -> Tuple[int, int]:
    """Convert normalized [0,1] coordinates to pixel coordinates.

    Args:
        coord: [x, y] in normalized [0, 1] range.
        screen_size: (width, height) in pixels.

    Returns:
        (x, y) in pixel coordinates.
    """
    x_norm, y_norm = coord[0], coord[1]
    x_pixel = int(x_norm * screen_size[0])
    y_pixel = int(y_norm * screen_size[1])
    return (x_pixel, y_pixel)


def pixel_to_normalized(
    x: int,
    y: int,
    screen_size: Tuple[int, int],
) -> Tuple[float, float]:
    """Convert pixel coordinates to normalized [0,1] range.

    Args:
        x, y: Pixel coordinates.
        screen_size: (width, height) in pixels.

    Returns:
        (x_norm, y_norm) in [0, 1] range.
    """
    x_norm = x / screen_size[0]
    y_norm = y / screen_size[1]
    return (x_norm, y_norm)


def scale_factor_to_normalized(coord: List[int]) -> List[float]:
    """Convert SCALE_FACTOR (0-999) coordinates to normalized [0,1].

    This is used after parsing LLM output.

    Args:
        coord: [x, y] in 0-999 range from LLM.

    Returns:
        [x, y] in normalized [0, 1] range.
    """
    return [c / SCALE_FACTOR for c in coord]


def normalized_to_scale_factor(coord: List[float]) -> List[int]:
    """Convert normalized [0,1] to SCALE_FACTOR (0-999) for LLM input.

    Args:
        coord: [x, y] in normalized [0, 1] range.

    Returns:
        [x, y] in 0-999 range for LLM.
    """
    return [int(c * SCALE_FACTOR) for c in coord]


# =============================================================================
# Image Processing Functions
# =============================================================================

def pil_to_bytes(image: Image.Image) -> bytes:
    """Convert PIL Image to bytes."""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return buffer.getvalue()


def pil_to_base64(image: Image.Image) -> str:
    """Convert PIL Image to base64 string."""
    buffer = BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode('utf-8')


def bytes_to_pil(image_bytes: bytes) -> Image.Image:
    """Convert bytes to PIL Image."""
    return Image.open(BytesIO(image_bytes))


# =============================================================================
# Response Parsing Functions
# =============================================================================

def parse_tagged_text(text: str) -> Dict[str, Any]:
    """Parse text containing XML-style tags to extract thinking and tool_call.

    Args:
        text: Text containing <thinking> and <tool_call> tags.

    Returns:
        Dictionary with keys 'thinking' and 'tool_call'.

    Raises:
        ValueError: If tool_call content is not valid JSON.
    """
    # Handle thinking model output format (uses </think> instead of </thinking>)
    if '</think>' in text and '</thinking>' not in text:
        text = text.replace('</think>', '</thinking>')
        text = '<thinking>' + text

    # Define regex pattern with non-greedy matching
    pattern = r'<thinking>(.*?)</thinking>.*?<tool_call>(.*?)</tool_call>'

    result: Dict[str, Any] = {
        'thinking': None,
        'tool_call': None,
    }

    # Use re.DOTALL to match newlines
    match = re.search(pattern, text, re.DOTALL)
    if match:
        result = {
            'thinking': match.group(1).strip().strip('"'),
            'tool_call': match.group(2).strip().strip('"'),
        }

    # Parse tool_call as JSON
    if result['tool_call']:
        try:
            result['tool_call'] = json.loads(result['tool_call'])
        except json.JSONDecodeError as e:
            raise ValueError(f'Invalid JSON in tool_call: {e}')

    return result


def parse_action_to_structure_output(text: str) -> Dict[str, Any]:
    """Parse model output text into structured action format.

    Args:
        text: Raw model output containing thinking and tool_call tags.

    Returns:
        Dictionary with keys 'thinking' and 'action_json'.
        Coordinates are normalized to [0, 1] range.
    """
    text = text.strip()

    results = parse_tagged_text(text)
    thinking = results['thinking']
    tool_call = results['tool_call']

    if not tool_call:
        return {'thinking': thinking, 'action_json': {'action': None}}

    action = tool_call.get('arguments', {})

    # Normalize coordinates from SCALE_FACTOR range to [0, 1]
    if 'coordinate' in action:
        coordinates = action['coordinate']
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError(
                f'Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}'
            )
        point_x = point_x / SCALE_FACTOR
        point_y = point_y / SCALE_FACTOR
        action['coordinate'] = [point_x, point_y]

    if 'start_coordinate' in action:
        coordinates = action['start_coordinate']
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError(
                f'Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}'
            )
        point_x = point_x / SCALE_FACTOR
        point_y = point_y / SCALE_FACTOR
        action['start_coordinate'] = [point_x, point_y]

    if 'end_coordinate' in action:
        coordinates = action['end_coordinate']
        if len(coordinates) == 2:
            point_x, point_y = coordinates
        elif len(coordinates) == 4:
            x1, y1, x2, y2 = coordinates
            point_x = (x1 + x2) / 2
            point_y = (y1 + y2) / 2
        else:
            raise ValueError(
                f'Invalid coordinate format: expected 2 or 4 values, got {len(coordinates)}'
            )
        point_x = point_x / SCALE_FACTOR
        point_y = point_y / SCALE_FACTOR
        action['end_coordinate'] = [point_x, point_y]

    return {
        'thinking': thinking,
        'action_json': action,
    }


# =============================================================================
# Action Conversion Functions
# =============================================================================

def convert_mai_action_to_json_action(
    mai_action: Dict[str, Any],
    screen_size: Tuple[int, int],
) -> json_action.JSONAction:
    """Convert MAI-UI action dict to android_world JSONAction.

    Args:
        mai_action: Dict with 'action' key and action-specific parameters.
            Coordinates should be in normalized [0, 1] range.
        screen_size: (width, height) of the screen in pixels.

    Returns:
        JSONAction object ready for execution.
    """
    action_type = mai_action.get('action')

    if action_type == 'click':
        coord = mai_action.get('coordinate', [0.5, 0.5])
        x, y = normalized_to_pixel(coord, screen_size)
        return json_action.JSONAction(action_type='click', x=x, y=y)

    elif action_type == 'long_press':
        coord = mai_action.get('coordinate', [0.5, 0.5])
        x, y = normalized_to_pixel(coord, screen_size)
        return json_action.JSONAction(action_type='long_press', x=x, y=y)

    elif action_type == 'double_click':
        coord = mai_action.get('coordinate', [0.5, 0.5])
        x, y = normalized_to_pixel(coord, screen_size)
        return json_action.JSONAction(action_type='double_tap', x=x, y=y)

    elif action_type == 'type':
        text = mai_action.get('text', '')
        return json_action.JSONAction(action_type='input_text', text=text)

    elif action_type == 'swipe':
        direction = mai_action.get('direction', 'down')
        coord = mai_action.get('coordinate')
        if coord:
            x, y = normalized_to_pixel(coord, screen_size)
            return json_action.JSONAction(
                action_type='scroll', direction=direction, x=x, y=y
            )
        return json_action.JSONAction(action_type='scroll', direction=direction)

    elif action_type == 'drag':
        start = mai_action.get('start_coordinate', [0.5, 0.5])
        end = mai_action.get('end_coordinate', [0.5, 0.5])
        start_x, start_y = normalized_to_pixel(start, screen_size)
        end_x, end_y = normalized_to_pixel(end, screen_size)
        # Calculate direction based on drag movement
        dx = end_x - start_x
        dy = end_y - start_y
        if abs(dx) > abs(dy):
            direction = 'right' if dx > 0 else 'left'
        else:
            direction = 'down' if dy > 0 else 'up'
        return json_action.JSONAction(
            action_type='scroll',
            direction=direction,
            x=start_x,
            y=start_y,
        )

    elif action_type == 'open':
        app_name = mai_action.get('text', '')
        return json_action.JSONAction(action_type='open_app', app_name=app_name)

    elif action_type == 'system_button':
        button = mai_action.get('button', '')
        if button == 'back':
            return json_action.JSONAction(action_type='navigate_back')
        elif button == 'home':
            return json_action.JSONAction(action_type='navigate_home')
        elif button == 'enter':
            return json_action.JSONAction(action_type='keyboard_enter')
        elif button == 'menu':
            # Menu button not directly supported, fallback to wait
            return json_action.JSONAction(action_type='wait')

    elif action_type == 'wait':
        return json_action.JSONAction(action_type='wait')

    elif action_type == 'terminate':
        status = mai_action.get('status', 'success')
        goal_status = 'complete' if status == 'success' else 'infeasible'
        return json_action.JSONAction(action_type='status', goal_status=goal_status)

    elif action_type == 'answer':
        text = mai_action.get('text', '')
        return json_action.JSONAction(action_type='answer', text=text)

    # Fallback for unknown actions
    return json_action.JSONAction(action_type='unknown')


def traj_step_to_response(step: TrajStep) -> str:
    """Convert a trajectory step to formatted response string for LLM context.

    Args:
        step: TrajStep object containing action and thought.

    Returns:
        Formatted string with thinking and tool_call tags.
    """
    thinking = step.thought
    structured_action = step.structured_action

    if not structured_action:
        return ''

    action_json = copy.deepcopy(structured_action.get('action_json', {}))

    # Convert normalized coordinates back to SCALE_FACTOR range
    if 'coordinate' in action_json:
        coordinates = action_json.get('coordinate', [])
        if len(coordinates) == 2:
            point_x, point_y = coordinates
            action_json['coordinate'] = [
                int(point_x * SCALE_FACTOR),
                int(point_y * SCALE_FACTOR),
            ]

    if 'start_coordinate' in action_json:
        coordinates = action_json.get('start_coordinate', [])
        if len(coordinates) == 2:
            point_x, point_y = coordinates
            action_json['start_coordinate'] = [
                int(point_x * SCALE_FACTOR),
                int(point_y * SCALE_FACTOR),
            ]

    if 'end_coordinate' in action_json:
        coordinates = action_json.get('end_coordinate', [])
        if len(coordinates) == 2:
            point_x, point_y = coordinates
            action_json['end_coordinate'] = [
                int(point_x * SCALE_FACTOR),
                int(point_y * SCALE_FACTOR),
            ]

    tool_call_dict = {
        'name': 'mobile_use',
        'arguments': action_json,
    }
    tool_call_json = json.dumps(tool_call_dict, separators=(',', ':'))
    return (
        f'<thinking>\n{thinking}\n</thinking>\n'
        f'<tool_call>\n{tool_call_json}\n</tool_call>'
    )


def mask_image_urls_for_logging(
    messages: List[Dict[str, Any]]
) -> List[Dict[str, Any]]:
    """Create a copy of messages with image URLs masked for logging.

    Args:
        messages: List of message dictionaries that may contain image URLs.

    Returns:
        Deep copy of messages with image URLs replaced by '[IMAGE_DATA]'.
    """
    messages_masked = copy.deepcopy(messages)
    for message in messages_masked:
        content = message.get('content', [])
        if content and isinstance(content, list):
            for item in content:
                if isinstance(item, dict) and 'image_url' in item:
                    item['image_url']['url'] = '[IMAGE_DATA]'
    return messages_masked


# =============================================================================
# System Prompt
# =============================================================================

SYSTEM_PROMPT = """You are a GUI agent. You are given a task and your action history, with screenshots. You need to perform the next action to complete the task.

## Output Format
For each function call, return the thinking process in <thinking> </thinking> tags, and a json object with function name and arguments within <tool_call></tool_call> XML tags:
```
<thinking>
...
</thinking>
<tool_call>
{"name": "mobile_use", "arguments": <args-json-object>}
</tool_call>
```

## Action Space

{"action": "click", "coordinate": [x, y]}
{"action": "long_press", "coordinate": [x, y]}
{"action": "type", "text": ""}
{"action": "swipe", "direction": "up or down or left or right", "coordinate": [x, y]} # "coordinate" is optional. Use the "coordinate" if you want to swipe a specific UI element.
{"action": "open", "text": "app_name"}
{"action": "drag", "start_coordinate": [x1, y1], "end_coordinate": [x2, y2]}
{"action": "system_button", "button": "button_name"} # Options: back, home, menu, enter
{"action": "wait"}
{"action": "terminate", "status": "success or fail"}
{"action": "answer", "text": "xxx"} # Use escape characters \\', \\", and \\n in text part to ensure we can parse the text in normal python string format.


## Note
- Write a small plan and finally summarize your next action (with its target element) in one sentence in <thinking></thinking> part.
- Available Apps: `["Camera","Chrome","Clock","Contacts","Dialer","Files","Settings","Markor","Tasks","Simple Draw Pro","Simple Gallery Pro","Simple SMS Messenger","Audio Recorder","Pro Expense","Broccoli APP","OSMand","VLC","Joplin","Retro Music","OpenTracks","Simple Calendar Pro"]`.
You should use the `open` action to open the app as possible as you can, because it is the fast way to open the app.
- You must follow the Action Space strictly, and return the correct json object within <thinking> </thinking> and <tool_call></tool_call> XML tags.
""".strip()
