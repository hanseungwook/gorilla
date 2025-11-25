import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class Olmo3Handler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)

    @override
    def decode_ast(self, result, language, has_tool_call_tag):
        # Model response is of the form:
        # "<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Taylor Swift\", \"duration\": 20}}\n</tool_call>\n<tool_call>\n{\"name\": \"spotify.play\", \"arguments\": {\"artist\": \"Maroon 5\", \"duration\": 15}}\n</tool_call>"?
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        return [
            {call["name"]: {k: v for k, v in call["arguments"].items()}}
            for call in tool_calls
        ]

    @override
    def decode_execute(self, result, has_tool_call_tag):
        tool_calls = self._extract_tool_calls(result)
        if type(tool_calls) != list or any(type(item) != dict for item in tool_calls):
            raise ValueError(f"Model did not return a list of function calls: {result}")
        decoded_result = []
        for item in tool_calls:
            if type(item) == str:
                item = eval(item)
            decoded_result.append({item["name"]: item["arguments"]})
        return convert_to_function_call(decoded_result)

    @override
    def _format_prompt(self, messages, function):
        """
        Apply the OLMo-3 chat template manually so we can inject the function list
        and convert tool responses into the expected roles / tags.

        Template reference:
        {% set has_system = messages|selectattr('role', 'equalto', 'system')|list|length > 0 %}
        ...
        {% if loop.last and add_generation_prompt %}{{ '<|im_start|>assistant\n<think>' }}{% endif %}
        """

        formatted_messages = []
        functions_str = "\n".join([json.dumps(func) for func in function])
        has_system = any(msg["role"] == "system" for msg in messages)

        if not has_system:
            default_system_msg = {
                "role": "system",
                "content": "You are a helpful AI assistant.",
            }
            if functions_str is not None:
                default_system_msg["functions"] = functions_str
            formatted_messages.append(default_system_msg)

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            if (
                message["role"] == "user"
                and isinstance(message.get("content"), str)
                and not (
                    message["content"].startswith("<tool_response>")
                    and message["content"].endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            msg = {
                key: value
                for key, value in message.items()
                if key != "reasoning_content"
            }
            role = msg["role"]

            if role == "system":
                msg["functions"] = functions_str

            if role == "assistant":
                reasoning_content = ""
                content = msg.get("content", "") or ""
                if "reasoning_content" in message and message["reasoning_content"]:
                    reasoning_content = message["reasoning_content"]
                elif isinstance(content, str) and "</think>" in content:
                    parts = content.split("</think>")
                    reasoning_content = (
                        parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                    )
                    content = parts[-1].lstrip("\n")
                    msg["content"] = content

                if idx > last_query_index:
                    if idx == len(messages) - 1 or reasoning_content:
                        think_block = (
                            "<think>\n"
                            + reasoning_content.strip("\n")
                            + "\n</think>\n\n"
                        )
                        msg["content"] = think_block + content.lstrip("\n")

                tool_calls = msg.pop("tool_calls", None)
                if tool_calls:
                    msg["function_calls"] = tool_calls

            if role == "tool":
                tool_content = msg.get("content", "")
                msg = {
                    "role": "environment",
                    "content": tool_content,
                }

            formatted_messages.append(msg)

        formatted_prompt = self.tokenizer.apply_chat_template(
            formatted_messages, tokenize=False, add_generation_prompt=True
        )
        print('Formatted prompt:', formatted_prompt)
        print('='*100)
        return formatted_prompt

    @override
    def _pre_query_processing_prompting(self, test_entry: dict) -> dict:
        functions: list = test_entry["function"]

        # FC models use its own system prompt, so no need to add any message

        return {"message": [], "function": functions}

    @override
    def _parse_query_response_prompting(self, api_response: Any) -> dict:
        model_response = api_response.choices[0].text
        extracted_tool_calls = self._extract_tool_calls(model_response)

        reasoning_content = ""
        cleaned_response = model_response

        if "</think>" in model_response:
            before_reasoning, after_reasoning = model_response.split("</think>", 1)
            reasoning_content = before_reasoning.split("<think>")[-1].strip("\n")
            cleaned_response = after_reasoning.lstrip("\n")

        if len(extracted_tool_calls) > 0:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": "",
                "tool_calls": extracted_tool_calls,
            }
        else:
            model_responses_message_for_chat_history = {
                "role": "assistant",
                "content": cleaned_response,
            }

        model_responses_message_for_chat_history["reasoning_content"] = reasoning_content

        print('Model response:', model_response)
        print('='*100)
        print('Cleaned response:', cleaned_response)

        return {
            "model_responses": cleaned_response,
            # "reasoning_content": reasoning_content,
            "reasoning_content": "Skipped for Olmo-3 model", # TODO
            "model_responses_message_for_chat_history": model_responses_message_for_chat_history,
            "input_token": api_response.usage.prompt_tokens,
            "output_token": api_response.usage.completion_tokens,
        }

    @override
    def _add_assistant_message_prompting(
        self, inference_data: dict, model_response_data: dict
    ) -> dict:
        inference_data["message"].append(
            model_response_data["model_responses_message_for_chat_history"],
        )
        return inference_data

    @staticmethod
    def _extract_tool_calls(input_string):
        pattern = r"<function_calls>(.*?)</function_calls>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # If the closing tag is missing, grab everything until the end.
        if not matches and "<function_calls>" in input_string:
            matches = re.findall(r"<function_calls>(.*)", input_string, re.DOTALL)

        result = []
        for match in matches:
            candidate = match.strip()
            if not candidate:
                continue

            parsed_candidate = None
            for payload in (candidate, f"[{candidate}]"):
                try:
                    parsed_candidate = json.loads(payload)
                    break
                except json.JSONDecodeError:
                    continue

            if parsed_candidate is None:
                try:
                    parsed_candidate = eval(candidate)
                except Exception:
                    try:
                        parsed_candidate = eval(f"[{candidate}]")
                    except Exception:
                        continue

            if isinstance(parsed_candidate, dict):
                parsed_candidate = [parsed_candidate]

            if isinstance(parsed_candidate, str):
                try:
                    parsed_candidate = json.loads(parsed_candidate)
                except json.JSONDecodeError:
                    try:
                        parsed_candidate = eval(parsed_candidate)
                    except Exception:
                        continue

            if isinstance(parsed_candidate, tuple):
                parsed_candidate = list(parsed_candidate)

            if not isinstance(parsed_candidate, list):
                continue

            for item in parsed_candidate:
                if isinstance(item, str):
                    try:
                        item = json.loads(item)
                    except json.JSONDecodeError:
                        try:
                            item = eval(item)
                        except Exception:
                            continue

                if not isinstance(item, dict):
                    continue

                if "function" in item:
                    name = item["function"].get("name")
                    arguments = item["function"].get("arguments", {})
                else:
                    name = item.get("name")
                    arguments = item.get("arguments", {})

                if isinstance(arguments, str):
                    try:
                        arguments = json.loads(arguments)
                    except json.JSONDecodeError:
                        pass

                if name is None:
                    continue

                result.append({"name": name, "arguments": arguments})

        return result
