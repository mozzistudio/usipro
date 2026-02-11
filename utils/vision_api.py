"""Claude Vision API wrapper for image analysis."""

import base64
import json
import logging
import time

import anthropic

logger = logging.getLogger(__name__)


class VisionAPI:
    """Wrapper around Claude Vision API with retry and rate limiting."""

    def __init__(self, config: dict):
        self.client = anthropic.Anthropic()
        self.model = config["api"]["model"]
        self.max_tokens = config["api"]["max_tokens"]
        self.timeout = config["api"]["timeout"]
        self.max_retries = config["api"]["max_retries"]
        self.retry_delay = config["api"]["retry_delay"]
        self.rate_limit_delay = config["api"]["rate_limit_delay"]
        self._last_call_time = 0.0

    def _rate_limit(self):
        """Enforce minimum delay between API calls."""
        elapsed = time.time() - self._last_call_time
        if elapsed < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - elapsed)

    def analyze_image(self, image_bytes: bytes, prompt: str, media_type: str = "image/png") -> dict:
        """Send an image to Claude Vision and get a structured JSON response.

        Args:
            image_bytes: Raw image bytes (PNG or JPEG).
            prompt: The analysis prompt to send with the image.
            media_type: MIME type of the image.

        Returns:
            Parsed JSON dict from Claude's response.
        """
        image_b64 = base64.standard_b64encode(image_bytes).decode("utf-8")

        for attempt in range(1, self.max_retries + 1):
            self._rate_limit()
            try:
                logger.debug("Vision API call attempt %d/%d", attempt, self.max_retries)
                response = self.client.messages.create(
                    model=self.model,
                    max_tokens=self.max_tokens,
                    messages=[
                        {
                            "role": "user",
                            "content": [
                                {
                                    "type": "image",
                                    "source": {
                                        "type": "base64",
                                        "media_type": media_type,
                                        "data": image_b64,
                                    },
                                },
                                {
                                    "type": "text",
                                    "text": prompt,
                                },
                            ],
                        }
                    ],
                    timeout=self.timeout,
                )
                self._last_call_time = time.time()

                text = response.content[0].text
                return self._parse_json_response(text)

            except anthropic.RateLimitError:
                logger.warning("Rate limited, waiting %ds (attempt %d/%d)",
                               self.retry_delay * 2, attempt, self.max_retries)
                time.sleep(self.retry_delay * 2)
            except anthropic.APIStatusError as e:
                logger.warning("API error %s (attempt %d/%d): %s",
                               e.status_code, attempt, self.max_retries, e.message)
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay)
            except anthropic.APITimeoutError:
                logger.warning("API timeout (attempt %d/%d)", attempt, self.max_retries)
                if attempt == self.max_retries:
                    raise
                time.sleep(self.retry_delay)

        raise RuntimeError("Vision API call failed after all retries")

    def _parse_json_response(self, text: str) -> dict:
        """Extract JSON from Claude's response text."""
        # Try direct parse first
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass

        # Look for JSON in code blocks
        for marker_start, marker_end in [("```json", "```"), ("```", "```")]:
            start = text.find(marker_start)
            if start != -1:
                start += len(marker_start)
                end = text.find(marker_end, start)
                if end != -1:
                    try:
                        return json.loads(text[start:end].strip())
                    except json.JSONDecodeError:
                        continue

        # Try to find any JSON object in the text
        brace_start = text.find("{")
        if brace_start != -1:
            depth = 0
            for i in range(brace_start, len(text)):
                if text[i] == "{":
                    depth += 1
                elif text[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(text[brace_start:i + 1])
                        except json.JSONDecodeError:
                            break

        logger.error("Failed to parse JSON from response: %s", text[:500])
        raise ValueError(f"Could not parse JSON from API response: {text[:200]}")
