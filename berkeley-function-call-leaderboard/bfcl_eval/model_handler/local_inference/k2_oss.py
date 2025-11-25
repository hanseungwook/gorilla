import json

from overrides import override

from bfcl_eval.model_handler.local_inference.k2 import K2Handler


class K2OSSHandler(K2Handler):
    @override
    def _format_prompt(self, messages, function):
        formatted_prompt = ""
        tools = function or []
        first_message = messages[0] if messages else None

        tools_header = (
            "\n# Tools\nYou may call one or more functions to assist with the user query.\n\n"
            "You are provided with function signatures within <tools></tools> XML tags:\n<tools>"
        )
        tools_footer = (
            '\n</tools>\n\nFor each function call, return a json object with function name and arguments within <tool_call></tool_call> XML tags:\n<tool_call>\n{"name": <function-name>, "arguments": <args-json-object>}\n</tool_call>You may use multiple <tool_call> blocks if multiple function calls are needed to fully address the request.<|im_end|>\n'
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
                formatted_prompt += (
                    system_content + "\n\n" if system_content else "\n\n"
                )
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

