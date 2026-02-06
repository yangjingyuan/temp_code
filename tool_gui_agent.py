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

"""Tool GUI Agent adapted from MAI-UI's navigation agent.

This agent uses vision-language models to interact with Android device
interfaces based on natural language instructions.
"""

import logging
import traceback
from io import BytesIO
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from openai import OpenAI
from PIL import Image

from android_world.agents import base_agent
from android_world.agents import tool_gui_agent_utils as utils
from android_world.env import interface
from android_world.env import json_action


class ToolGuiAgent(base_agent.EnvironmentInteractingAgent):
    """Tool-based GUI Agent adapted from MAI-UI's navigation agent.

    This agent processes screenshots and natural language instructions to
    generate GUI actions for Android device automation.
    """

    def __init__(
        self,
        env: interface.AsyncEnv,
        llm_base_url: str,
        model_name: str,
        api_key: str = 'empty',
        name: str = 'ToolGuiAgent',
        history_n: int = 3,
        temperature: float = 0.0,
        max_tokens: int = 2048,
        transition_pause: float | None = 1.0,
    ):
        """Initialize the ToolGuiAgent.

        Args:
            env: The android_world environment.
            llm_base_url: Base URL for the LLM API endpoint.
            model_name: Name of the model to use.
            api_key: API key for the LLM service.
            name: Name of the agent.
            history_n: Number of history steps to include in context.
            temperature: Sampling temperature for LLM.
            max_tokens: Maximum tokens in LLM response.
            transition_pause: Pause time before grabbing state after action.
        """
        super().__init__(env, name=name, transition_pause=transition_pause)

        self.llm_base_url = llm_base_url
        self.model_name = model_name
        self.api_key = api_key
        self.history_n = history_n
        self.temperature = temperature
        self.max_tokens = max_tokens

        # Initialize OpenAI client
        self.llm = OpenAI(
            base_url=self.llm_base_url,
            api_key=self.api_key,
        )

        # Initialize trajectory memory
        self.traj_memory = utils.TrajMemory()

        # Task guidelines (can be set externally)
        self._task_guidelines: List[str] = []

    @property
    def system_prompt(self) -> str:
        """Get the system prompt, including any task guidelines."""
        prompt = utils.SYSTEM_PROMPT
        if self._task_guidelines:
            guidelines = '\n'.join(f'- {g}' for g in self._task_guidelines)
            prompt += f'\n\n## Additional Guidelines\n{guidelines}'
        return prompt

    def set_task_guidelines(self, guidelines: List[str]) -> None:
        """Set additional task-specific guidelines.

        Args:
            guidelines: List of guideline strings.
        """
        self._task_guidelines = guidelines

    def reset(self, go_home: bool = False) -> None:
        """Reset the agent state and trajectory memory.

        Args:
            go_home: Whether to navigate to home screen.
        """
        super().reset(go_home=go_home)
        self.traj_memory = utils.TrajMemory()

    def _prepare_images(
        self, screenshot_bytes: bytes
    ) -> List[Image.Image]:
        """Prepare image list including history and current screenshot.

        Args:
            screenshot_bytes: Current screenshot as bytes.

        Returns:
            List of PIL Images (history + current).
        """
        # Get history images
        history_images = self.traj_memory.history_images

        # Calculate how many history images to include
        if len(history_images) > 0:
            max_history = min(len(history_images), self.history_n - 1)
            recent_history = history_images[-max_history:] if max_history > 0 else []
        else:
            recent_history = []

        # Add current image bytes
        all_bytes = list(recent_history) + [screenshot_bytes]

        # Convert all to PIL images
        images = []
        for img_bytes in all_bytes:
            image = Image.open(BytesIO(img_bytes))
            if image.mode != 'RGB':
                image = image.convert('RGB')
            images.append(image)

        return images

    def _build_messages(
        self,
        instruction: str,
        images: List[Image.Image],
    ) -> List[Dict[str, Any]]:
        """Build the message list for the LLM API call.

        Args:
            instruction: Task instruction from user.
            images: List of prepared images.

        Returns:
            List of message dictionaries for the API.
        """
        messages = [
            {
                'role': 'system',
                'content': [{'type': 'text', 'text': self.system_prompt}],
            },
            {
                'role': 'user',
                'content': [{'type': 'text', 'text': instruction}],
            },
        ]

        image_idx = 0

        if len(self.traj_memory.steps) > 0:
            # Only the last (history_n - 1) history responses need images
            start_image_idx = max(
                0, len(self.traj_memory.steps) - (self.history_n - 1)
            )

            for history_idx, step in enumerate(self.traj_memory.steps):
                # Only include images for the last (history_n - 1) steps
                should_include_image = history_idx >= start_image_idx

                if should_include_image:
                    # Add image before the assistant response
                    if image_idx < len(images) - 1:
                        cur_image = images[image_idx]
                        encoded_string = utils.pil_to_base64(cur_image)
                        messages.append({
                            'role': 'user',
                            'content': [{
                                'type': 'image_url',
                                'image_url': {
                                    'url': f'data:image/png;base64,{encoded_string}'
                                },
                            }],
                        })
                    image_idx += 1

                # Always add the assistant response
                history_response = utils.traj_step_to_response(step)
                if history_response:
                    messages.append({
                        'role': 'assistant',
                        'content': [{'type': 'text', 'text': history_response}],
                    })

                # Add ask_user_response if present
                if step.ask_user_response:
                    messages.append({
                        'role': 'user',
                        'content': [{'type': 'text', 'text': step.ask_user_response}],
                    })

            # Add current image (last one in images list)
            if image_idx < len(images):
                cur_image = images[image_idx]
                encoded_string = utils.pil_to_base64(cur_image)
                messages.append({
                    'role': 'user',
                    'content': [{
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{encoded_string}'
                        },
                    }],
                })
        else:
            # No history, just add the current image
            if images:
                cur_image = images[0]
                encoded_string = utils.pil_to_base64(cur_image)
                messages.append({
                    'role': 'user',
                    'content': [{
                        'type': 'image_url',
                        'image_url': {
                            'url': f'data:image/png;base64,{encoded_string}'
                        },
                    }],
                })

        return messages

    def _call_llm(
        self, messages: List[Dict[str, Any]]
    ) -> Tuple[Optional[str], Optional[Dict[str, Any]]]:
        """Call LLM API with retry logic.

        Args:
            messages: List of message dictionaries.

        Returns:
            Tuple of (prediction_text, action_dict) or (None, None) on failure.
        """
        max_retries = 3
        prediction = None
        action_json = None

        for attempt in range(max_retries):
            try:
                # Log messages (with images masked)
                messages_print = utils.mask_image_urls_for_logging(messages)
                logging.info('Messages (attempt %d): %s', attempt + 1, messages_print)

                response = self.llm.chat.completions.create(
                    model=self.model_name,
                    messages=messages,
                    max_tokens=self.max_tokens,
                    temperature=self.temperature,
                    top_p=1.0,
                    frequency_penalty=0.0,
                    presence_penalty=0.0,
                    seed=42,
                )
                prediction = response.choices[0].message.content.strip()
                logging.info('Raw response: %s', prediction)

                # Parse response
                parsed_response = utils.parse_action_to_structure_output(prediction)
                action_json = parsed_response['action_json']
                logging.info('Parsed action: %s', action_json)
                break

            except Exception as e:
                logging.error('Error on attempt %d: %s', attempt + 1, e)
                traceback.print_exc()
                prediction = None
                action_json = None

        return prediction, action_json

    def step(self, goal: str) -> base_agent.AgentInteractionResult:
        """Perform one step of agent interaction.

        Args:
            goal: The task goal/instruction.

        Returns:
            AgentInteractionResult with done flag and step data.
        """
        step_data: Dict[str, Any] = {
            'raw_screenshot': None,
            'ui_elements': None,
            'action_prompt': None,
            'action_output': None,
            'action_json': None,
            'thinking': None,
            'raw_response': None,
        }

        step_num = len(self.traj_memory.steps) + 1
        logging.info('---------- step %d ----------', step_num)

        # Get current state
        state = self.get_post_transition_state()
        screen_size = self.env.logical_screen_size

        step_data['raw_screenshot'] = state.pixels.copy()
        step_data['ui_elements'] = state.ui_elements

        # Convert numpy array to PIL Image
        screenshot_pil = Image.fromarray(state.pixels)
        screenshot_bytes = utils.pil_to_bytes(screenshot_pil)

        # Set task goal if first step
        if not self.traj_memory.task_goal:
            self.traj_memory.task_goal = goal

        # Prepare images (history + current)
        images = self._prepare_images(screenshot_bytes)

        # Build messages
        messages = self._build_messages(goal, images)
        step_data['action_prompt'] = utils.mask_image_urls_for_logging(messages)

        # Call LLM
        prediction, action_json = self._call_llm(messages)

        if prediction is None or action_json is None:
            logging.error('LLM call failed after all retries')
            step_data['summary'] = 'LLM error: failed after all retries'
            return base_agent.AgentInteractionResult(False, step_data)

        step_data['raw_response'] = prediction
        step_data['action_output'] = prediction

        # Parse thinking
        try:
            parsed = utils.parse_action_to_structure_output(prediction)
            thinking = parsed.get('thinking', '')
            step_data['thinking'] = thinking
            step_data['action_json'] = action_json
        except Exception as e:
            logging.error('Failed to parse response: %s', e)
            thinking = ''

        # Create trajectory step
        traj_step = utils.TrajStep(
            screenshot=screenshot_pil,
            screenshot_bytes=screenshot_bytes,
            prediction=prediction,
            action=action_json,
            thought=thinking,
            step_index=len(self.traj_memory.steps),
            structured_action={'action_json': action_json},
        )
        self.traj_memory.steps.append(traj_step)

        # Check for null action
        if action_json.get('action') is None:
            logging.warning('Received null action from LLM')
            step_data['summary'] = 'Null action received'
            return base_agent.AgentInteractionResult(False, step_data)

        # Convert to JSONAction
        try:
            json_act = utils.convert_mai_action_to_json_action(
                action_json, screen_size
            )
        except Exception as e:
            logging.error('Action conversion failed: %s', e)
            step_data['summary'] = f'Action conversion error: {e}'
            return base_agent.AgentInteractionResult(False, step_data)

        # Check for termination
        if json_act.action_type == json_action.STATUS:
            logging.info('Agent finished with status: %s', json_act.goal_status)
            step_data['summary'] = f'Task {json_act.goal_status}'
            return base_agent.AgentInteractionResult(True, step_data)

        # Check for answer action
        if json_act.action_type == json_action.ANSWER:
            logging.info('Agent provided answer: %s', json_act.text)
            step_data['summary'] = f'Answer: {json_act.text}'
            return base_agent.AgentInteractionResult(True, step_data)

        # Handle ask_user action (if supported by action)
        if action_json.get('action') == 'ask_user':
            question = action_json.get('text', '')
            logging.info('Agent asking user: %s', question)
            try:
                response = self.env.ask_question(question)
                traj_step.ask_user_response = response
            except Exception as e:
                logging.warning('ask_question not supported: %s', e)
            step_data['summary'] = f'Asked user: {question}'
            return base_agent.AgentInteractionResult(False, step_data)

        # Execute action
        try:
            logging.info('Executing action: %s', json_act)
            self.env.execute_action(json_act)
            step_data['summary'] = f'Executed: {json_act.action_type}'
        except Exception as e:
            logging.error('Action execution failed: %s', e)
            step_data['summary'] = f'Execution error: {e}'

        return base_agent.AgentInteractionResult(False, step_data)
