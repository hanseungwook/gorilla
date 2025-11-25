import json
import re
from typing import Any

from bfcl_eval.model_handler.local_inference.base_oss_handler import OSSHandler
from bfcl_eval.model_handler.utils import convert_to_function_call
from overrides import override


class K2Handler(OSSHandler):
    def __init__(
        self,
        model_name,
        temperature,
        registry_name,
        is_fc_model,
        dtype="bfloat16",
        reasoning_effort=None,
        **kwargs,
    ) -> None:
        super().__init__(model_name, temperature, registry_name, is_fc_model, **kwargs)
        self.reasoning_effort = reasoning_effort

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
        Use the chat template from the model to format the prompt.
        """
        formatted_prompt = ""
        tools = function or []
        first_message = messages[0] if len(messages) > 0 else None

        tools_header = (
            "\n# Tools\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        tools_footer = (
            '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>\nYou may use multiple <tool_call> blocks if multiple function calls are needed to fully address the request.<|im_end|>\n'
        )

        if tools:
            formatted_prompt += "<|im_start|>system\n"
            if (
                first_message
                and first_message.get("role") == "system"
                and first_message.get("content")
            ):
                formatted_prompt += first_message["content"] + "\n\n"
            formatted_prompt += tools_header
            for tool in tools:
                formatted_prompt += f"\n{json.dumps(tool)}"
            formatted_prompt += tools_footer

        elif first_message and first_message.get("role") == "system":
            system_content = first_message.get("content") or ""
            system_tools = first_message.get("tools")
            if system_tools:
                formatted_prompt += "<|im_start|>system\n"
                formatted_prompt += system_content + "\n\n" if system_content else "\n\n"
                formatted_prompt += tools_header
                for tool in system_tools:
                    formatted_prompt += f"\n{json.dumps(tool)}"
                formatted_prompt += tools_footer
            else:
                formatted_prompt += f"<|im_start|>system\n{system_content}<|im_end|>\n"

        last_query_index = len(messages) - 1
        for offset, message in enumerate(reversed(messages)):
            idx = len(messages) - 1 - offset
            content = message.get("content")
            if (
                message.get("role") == "user"
                and isinstance(content, str)
                and not (
                    content.startswith("<tool_response>")
                    and content.endswith("</tool_response>")
                )
            ):
                last_query_index = idx
                break

        for idx, message in enumerate(messages):
            role = message.get("role")
            raw_content = message.get("content")
            content = raw_content if isinstance(raw_content, str) else ""

            if role == "user" or (role == "system" and idx != 0):
                formatted_prompt += f"<|im_start|>{role}\n{content}<|im_end|>\n"
                continue

            if role == "assistant":
                thinking_content = ""
                think_tag = ""
                think_value = message.get("think")
                think_fast_value = message.get("think_fast")
                think_faster_value = message.get("think_faster")

                if isinstance(think_value, str):
                    thinking_content = think_value
                    think_tag = "think"
                elif isinstance(think_fast_value, str):
                    thinking_content = think_fast_value
                    think_tag = "think_fast"
                elif isinstance(think_faster_value, str):
                    thinking_content = think_faster_value
                    think_tag = "think_faster"
                else:
                    if "</think>" in content:
                        parts = content.split("</think>")
                        thinking_content = (
                            parts[0].rstrip("\n").split("<think>")[-1].lstrip("\n")
                        )
                        content = parts[-1].lstrip("\n")
                        think_tag = "think"
                    elif "</think_fast>" in content:
                        parts = content.split("</think_fast>")
                        thinking_content = (
                            parts[0]
                            .rstrip("\n")
                            .split("<think_fast>")[-1]
                            .lstrip("\n")
                        )
                        content = parts[-1].lstrip("\n")
                        think_tag = "think_fast"
                    elif "</think_faster>" in content:
                        parts = content.split("</think_faster>")
                        thinking_content = (
                            parts[0]
                            .rstrip("\n")
                            .split("<think_faster>")[-1]
                            .lstrip("\n")
                        )
                        content = parts[-1].lstrip("\n")
                        think_tag = "think_faster"
                    else:
                        thinking_content = ""
                        think_tag = "think_faster"

                formatted_prompt += f"<|im_start|>{role}"
                if idx > last_query_index:
                    if idx == len(messages) - 1 and think_tag:
                        if thinking_content:
                            formatted_prompt += (
                                f"\n<{think_tag}>\n{thinking_content}\n</{think_tag}>\n"
                                + content.lstrip("\n")
                            )
                        else:
                            formatted_prompt += (
                                f"\n<{think_tag}>\n</{think_tag}>\n"
                                + content.lstrip("\n")
                            )
                    else:
                        formatted_prompt += f"\n{content}"
                else:
                    formatted_prompt += f"\n{content}"

                tool_calls = message.get("tool_calls") or []
                for tool_idx, tool_call in enumerate(tool_calls):
                    if (tool_idx == 0 and content) or tool_idx > 0:
                        formatted_prompt += "\n"

                    call_obj = tool_call
                    if isinstance(tool_call, dict) and "function" in tool_call:
                        call_obj = tool_call["function"]
                    elif hasattr(tool_call, "function"):
                        call_obj = tool_call.function

                    name = call_obj["name"] if isinstance(call_obj, dict) else call_obj.name
                    arguments = (
                        call_obj.get("arguments")
                        if isinstance(call_obj, dict)
                        else call_obj.arguments
                    )
                    formatted_prompt += '<tool_call>\n{"name": "'
                    formatted_prompt += name
                    formatted_prompt += '", "arguments": '
                    if isinstance(arguments, str):
                        formatted_prompt += arguments
                    else:
                        formatted_prompt += json.dumps(arguments)
                    formatted_prompt += "}\n</tool_call>"

                formatted_prompt += "<|im_end|>\n"
                continue

            if role == "tool":
                prev_role = messages[idx - 1]["role"] if idx > 0 else None
                next_role = messages[idx + 1]["role"] if idx < len(messages) - 1 else None

                if idx == 0 or prev_role != "tool":
                    formatted_prompt += "<|im_start|>tool\n"

                formatted_prompt += content

                if idx == len(messages) - 1 or next_role != "tool":
                    formatted_prompt += "<|im_end|>\n"

        effort = self.reasoning_effort or "medium"
        if effort == "high":
            formatted_prompt += "<|im_start|>assistant\n<think>\n"
        elif effort == "medium":
            formatted_prompt += "<|im_start|>assistant\n<think_fast>\n"
        elif effort == "low":
            formatted_prompt += "<|im_start|>assistant\n<think_faster>\n"
        else:
            formatted_prompt += "<|im_start|>assistant\n<think_fast>\n"

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

        if self.reasoning_effort == "high":
            start_tag, end_tag = "<think>", "</think>"
        elif self.reasoning_effort == "medium":
            start_tag, end_tag = "<think_fast>", "</think_fast>"
        elif self.reasoning_effort == "low":
            start_tag, end_tag = "<think>", "</think>"
            # TODO: need to change this for new SFT model
            # start_tag, end_tag = "<think_faster>", "</think_faster>"
        else:
            raise Exception('reasoning_effort must be "high", "medium", or "low"')

        if end_tag in model_response:
            parts = model_response.split(end_tag)
            reasoning_content = parts[0].rstrip("\n").split(start_tag)[-1].lstrip("\n")
            cleaned_response = parts[-1].lstrip("\n")

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

        return {
            "model_responses": cleaned_response,
            # "reasoning_content": reasoning_content,
            "reasoning_content": "Skipped for K2 model", # TODO
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
        pattern = r"<tool_call>\n(.*?)\n</tool_call>"
        matches = re.findall(pattern, input_string, re.DOTALL)

        # Process matches into a list of dictionaries
        result = []
        for match in matches:
            try:
                match = json.loads(match)
                result.append(match)
            except Exception as e:
                pass
        return result
