import io
import base64
from types import SimpleNamespace
from PIL import Image
import openai


class OpenAIAsGenerativeModel:
    """
    Pretends to be a Google GenerativeModel while using an OpenAI-compatible Chat Completions API.
    - Accepts 'contents' as a list of Google-style prompt parts (strings or dicts with {mime_type, data})
    - Returns an object with .text and .candidates (each candidate has .content and .finish_reason int)
    """

    def __init__(self, model: str, api_key: str | None = None, base_url: str | None = None):
        # Works with OpenAI and any OpenAI-compatible endpoint 
        self.client = openai.OpenAI(api_key=api_key, base_url=base_url)
        self.model = model

    # ---------------------- public API ----------------------
    def generate_content(self, contents, generation_config=None, safety_settings=None):
        """
        contents: list of prompt parts (e.g., ["hello", {"mime_type": "image/png", "data": b"..."}])
        generation_config: optional dict; common keys mapped: temperature, top_p, max_output_tokens, stop,
                           presence_penalty, frequency_penalty
        safety_settings: ignored for OpenAI backends (kept for interface compatibility)
        """
        messages = self._prompt_parser(contents)
        params = self._map_gen_config(generation_config)
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            stream=False,
            **params,
        )

        # Build Gemini-like response
        finish_map = {"stop": 1, "length": 0, "tool_calls": 2, "content_filter": 3}
        candidates = []
        for ch in resp.choices:
            text = (ch.message.content or "")
            text = self._fix_json_format(text)
            fr = finish_map.get(ch.finish_reason, -1)
            candidates.append(SimpleNamespace(content=text, finish_reason=fr))

        first_text = candidates[0].content if candidates else ""
        final_response = SimpleNamespace(text=first_text, candidates=candidates)
        return final_response

    # ---------------------- helpers ----------------------
    def _fix_json_format(self, response_text: str) -> str:
        # Trim ``` and stray "json" tags for downstream JSON parsing convenience
        if "```" in response_text:
            response_text = response_text.replace("```", "")
        # be conservative: only strip leading 'json' fences, not legitimate content
        if response_text.lstrip().startswith("json"):
            response_text = response_text.lstrip()[4:]
        return response_text

    def _map_gen_config(self, cfg):
        if not cfg:
            return {}

        out = {}
        if getattr(cfg, "temperature", None) is not None:
            out["temperature"] = cfg.temperature
        if getattr(cfg, "top_p", None) is not None:
            out["top_p"] = cfg.top_p
        if getattr(cfg, "max_output_tokens", None) is not None:
            out["max_tokens"] = cfg.max_output_tokens
        if getattr(cfg, "presence_penalty", None) is not None:
            out["presence_penalty"] = cfg.presence_penalty
        if getattr(cfg, "frequency_penalty", None) is not None:
            out["frequency_penalty"] = cfg.frequency_penalty
        if getattr(cfg, "stop_sequences", None):
            out["stop"] = cfg.stop_sequences

        return out

    def _to_data_url(self, mime_type: str, data_bytes: bytes) -> str:
        b64 = base64.b64encode(data_bytes).decode("ascii")
        return f"data:{mime_type};base64,{b64}"

    def _pil_to_data_url(self, img: Image.Image, mime_type: str = "image/png") -> str:
        buf = io.BytesIO()
        fmt = "PNG" if mime_type.lower().endswith("png") else "JPEG"
        img.save(buf, format=fmt)
        return self._to_data_url(mime_type, buf.getvalue())

    def _prompt_parser(self, genai_parts):
        """
        Convert Google-style prompt parts to OpenAI chat format:
          messages = [
            {"role": "user",
             "content": [
               {"type": "text", "text": "..."},
               {"type": "image_url", "image_url": {"url": "data:image/png;base64,..." }}
             ]}
          ]
        """
        user_content = []

        for part in genai_parts:
            # plain text
            if isinstance(part, str):
                user_content.append({"type": "text", "text": part})
                continue

            # PIL Image
            if isinstance(part, Image.Image):
                url = self._pil_to_data_url(part, "image/png")
                user_content.append({"type": "image_url", "image_url": {"url": url}})
                continue

            # dict parts: expect {"mime_type": "...", "data": b"..."} for images
            if isinstance(part, dict):
                mime = part.get("mime_type", "")
                data = part.get("data", None)

                if isinstance(data, (bytes, bytearray)) and mime.startswith("image/"):
                    try:
                        # if data is raw bytes, use directly
                        url = self._to_data_url(mime, data)
                        user_content.append({"type": "image_url", "image_url": {"url": url}})
                        continue
                    except Exception as _:
                        pass

                # fallback: stringify any other dict
                user_content.append({"type": "text", "text": str(part)})
                continue

            # everything else -> string
            user_content.append({"type": "text", "text": str(part)})

        return [{"role": "user", "content": user_content}]


