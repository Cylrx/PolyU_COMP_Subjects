from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Iterable, Optional
import io
import base64
from PIL import Image
from colorama import Fore

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI

from src.config import cfg

import warnings

# ignore Gemini python version warning
warnings.simplefilter(action='ignore', category=FutureWarning)

def extract_code_block(text: str) -> str: 
    if "```" not in text: 
        return text
    pattern = r"```(?:[a-z]*)?\n(.*?)```"
    return re.search(pattern, text, re.DOTALL).group(1)


class ChatSession:
    _MODES = {"conversation", "one-shot"}

    @dataclass(slots=True)
    class Turn:
        role: str
        content: str

    def __init__(
        self, 
        model: str, 
        *, 
        mode: str = "one-shot", 
        sys_prompt: Optional[str] = None,
        verbose: bool = False,
    ):
        if mode not in self._MODES:
            raise ValueError(mode)
        
        if model.startswith(("gpt", "o1", "o3", "o4")):
            self._llm = ChatOpenAI(
                model=model, 
                temperature=cfg.llm["temperature"], 
                max_tokens=cfg.llm["max_tokens"], 
                api_key=os.environ["OPENAI_API_KEY"]
            )
        elif model.startswith("gemini"):
            self._llm = ChatGoogleGenerativeAI(
                model=model,
                temperature=cfg.llm["temperature"],
                max_output_tokens=cfg.llm["max_tokens"],
                api_key=os.environ["GOOGLE_API_KEY"]
            )
        else:
            raise ValueError(f"Unsupported model: {model}")

        self._mode = mode
        self._log: list[BaseMessage] = []
        self._last_prompt: HumanMessage | None = None
        self._sys_prompt = sys_prompt
        self._verbose = verbose
        if sys_prompt: 
            assert isinstance(sys_prompt, str)
            sys_msg = SystemMessage(content=sys_prompt)
            self._log.append(sys_msg)


    def set_mode(self, mode: str) -> None:
        if mode not in self._MODES:
            raise ValueError(mode)
        if mode == self._mode:
            return
        self._mode = mode
        if mode == "one-shot":
            self.clear()


    def _extract_content(self, content: str | list[str | dict]) -> str:
        """Extract text content from string or list of blocks (Gemini may return list with 'extras')."""
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            text_parts = []
            for part in content:
                if isinstance(part, str):
                    text_parts.append(part)
                elif isinstance(part, dict) and part.get("type") == "text":
                    text_parts.append(part.get("text", ""))
            return "".join(text_parts)
        return str(content)

    def ask(self, content: str, images: Optional[Iterable[Image.Image]] = None) -> str:
        if images:
            parts = [{"type": "text", "text": content}]
            for img in images:
                assert isinstance(img, Image.Image)
                buf = io.BytesIO()
                img.save(buf, format="PNG")
                b64 = base64.b64encode(buf.getvalue()).decode("utf-8")
                parts.append({
                    "type": "image_url", 
                    "image_url": {"url": f"data:image/png;base64,{b64}"}
                })
            prompt = HumanMessage(content=parts)
        else:
            prompt = HumanMessage(content=content)
        if self._verbose:
            head = "=" * 32 + " Human Message " + "=" * 32
            print(f"{head}\n\n{content}\n")
 

        self._last_prompt = prompt
        if self._mode == "conversation":
            self._log.append(prompt)
            reply: AIMessage = self._llm.invoke(self._log)
            self._log.append(reply)
        else:
            assert (
                len(self._log) == 0 or  # no sys prompt
                (len(self._log) == 1 and  # one sys prompt only
                isinstance(self._log[0], SystemMessage))
            )
            reply: AIMessage = self._llm.invoke(self._log + [prompt])
        assert isinstance(reply, AIMessage)
        content = self._extract_content(reply.content)
        if self._verbose:
            head = "=" * 32 + " AI Message " + "=" * 32
            print(f"{Fore.CYAN}{head}\n\n{content}{Fore.RESET}")
        return content


    def retry(self) -> str:
        if not self._last_prompt:
            raise RuntimeError("No message to retry. Call `ask` first.")
        if self._mode == "conversation":
            if not self._log or not isinstance(self._log[-1], AIMessage):
                raise RuntimeError("No response to retry.")
            self._log.pop()
            reply: AIMessage = self._llm.invoke(self._log)
            self._log.append(reply)
        else:
            ctx = self._log + [self._last_prompt]
            reply = self._llm.invoke(ctx)
        return self._extract_content(reply.content)

    def inject_exchange(self, human_content: str, ai_content: str) -> None:
        """Inject a human-AI exchange into the log (for priming context)."""
        self._log.append(HumanMessage(content=human_content))
        self._log.append(AIMessage(content=ai_content))

    def history(self) -> Iterable[Turn]:
        for msg in self._log:
            if isinstance(msg, HumanMessage):
                yield self.Turn("user", self._extract_content(msg.content))
            elif isinstance(msg, AIMessage):
                yield self.Turn("ai", self._extract_content(msg.content))

    def clear(self) -> None:
        self._log.clear()
        if self._sys_prompt:
            self._log.append(SystemMessage(content=self._sys_prompt))
        self._last_prompt = None


# Example Usage
# llm = ChatSession('gpt=4o', mode="conversation", sys_prompt=prompts.constraints['sys'])
# prompt: str = prompts.constraints['coarse']
# template: Template = Template(prompt)
# prompt = template.substitute(
#     design="",
#     asset_list_str="",
# )
# response: str = llm.ask(prompt)