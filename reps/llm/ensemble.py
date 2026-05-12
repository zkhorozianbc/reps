"""
Model ensemble for LLMs
"""

import asyncio
import logging
import random
from typing import Dict, List, Optional, Tuple

from reps.llm.base import LLMInterface
from reps.llm.openai_compatible import OpenAICompatibleLLM
from reps.llm.anthropic import AnthropicLLM
from reps.config import LLMModelConfig

logger = logging.getLogger(__name__)


class LLMEnsemble:
    """Ensemble of LLMs"""

    def __init__(self, models_cfg: List[LLMModelConfig]):
        self.models_cfg = models_cfg

        # Initialize models from the configuration
        self.models = [self._create_model(model_cfg) for model_cfg in models_cfg]

        # Extract and normalize model weights
        self.weights = [model.weight for model in models_cfg]
        total = sum(self.weights)
        self.weights = [w / total for w in self.weights]

        # Set up random state for deterministic model selection
        self.random_state = random.Random()
        # Initialize with seed from first model's config if available
        if (
            models_cfg
            and hasattr(models_cfg[0], "random_seed")
            and models_cfg[0].random_seed is not None
        ):
            self.random_state.seed(models_cfg[0].random_seed)
            logger.debug(
                f"LLMEnsemble: Set random seed to {models_cfg[0].random_seed} for deterministic model selection"
            )

        # Only log if we have multiple models or this is the first ensemble
        if len(models_cfg) > 1 or not hasattr(logger, "_ensemble_logged"):
            logger.info(
                f"Initialized LLM ensemble with models: "
                + ", ".join(
                    f"{model.name} (weight: {weight:.2f})"
                    for model, weight in zip(models_cfg, self.weights)
                )
            )
            logger._ensemble_logged = True

    @staticmethod
    def _create_model(model_cfg: LLMModelConfig) -> LLMInterface:
        """Dispatch model creation based on provider attribute.

        Priority:
        1. ``model_cfg.init_client`` callable (fully custom)
        2. ``model_cfg.provider == "anthropic"`` -> AnthropicLLM
        3. ``model_cfg.provider in {"openai", "openrouter", None, ""}`` ->
           OpenAICompatibleLLM (one class handles both via ``api_base``)
        4. Anything else raises ``ValueError``.
        """
        if model_cfg.init_client:
            return model_cfg.init_client(model_cfg)
        provider = getattr(model_cfg, "provider", None)
        if provider == "anthropic":
            return AnthropicLLM(model_cfg)
        if provider in ("openai", "openrouter", None, ""):
            return OpenAICompatibleLLM(model_cfg)
        raise ValueError(f"Unknown provider: {provider!r}")

    async def generate(self, prompt: str, **kwargs) -> str:
        """Generate text using a randomly selected model based on weights"""
        model = self._select_model(**kwargs)
        result = await model.generate(prompt, **kwargs)
        if hasattr(model, "last_usage"):
            self._last_usage = model.last_usage
        return result

    async def generate_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a system message and conversational context"""
        model = self._select_model(**kwargs)
        result = await model.generate_with_context(system_message, messages, **kwargs)
        # Capture token usage from the model that just ran
        if hasattr(model, "last_usage"):
            self._last_usage = model.last_usage
        # Capture reasoning output if the provider surfaced any
        self._last_reasoning = getattr(model, "last_reasoning", None)
        self._last_model_name = getattr(model, "model", None)
        return result

    @property
    def last_usage(self) -> Dict[str, int]:
        """Token usage from the last API call across any model in the ensemble."""
        return getattr(self, "_last_usage", {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0})

    @property
    def last_reasoning(self) -> Optional[str]:
        """Reasoning output (if any) from the last API call."""
        return getattr(self, "_last_reasoning", None)

    @property
    def last_model_name(self) -> Optional[str]:
        """Name of the model actually selected for the last API call."""
        return getattr(self, "_last_model_name", None)

    def _select_model(self, **kwargs) -> LLMInterface:
        """Select a model -- by name override if provided, otherwise by weighted sampling.

        Raises `ValueError` if `model=` is passed but not found in the
        ensemble. Previously this silently fell back to weighted sampling,
        which masked real config mistakes (e.g. summarizer requesting a
        model that wasn't in the worker ensemble would quietly run on the
        worker primary instead — a 404 later). Callers that genuinely
        want "use this model if present, else sample" should check
        `self.models` explicitly before passing `model=`.
        """
        model_override = kwargs.pop("model", None)
        if model_override:
            override_aliases = self._model_aliases(model_override)
            for cfg, m in zip(self.models_cfg, self.models):
                aliases = self._model_aliases(getattr(m, "model", None))
                aliases.update(self._model_aliases(getattr(cfg, "name", None)))
                if aliases & override_aliases:
                    logger.info(f"Using model override: {model_override}")
                    return m
            # No match found — fail loudly so the caller sees its config
            # error rather than a cryptic 404 three layers down.
            available = [getattr(m, "model", "?") for m in self.models]
            raise ValueError(
                f"Model override {model_override!r} not found in ensemble. "
                f"Available models: {available}. "
                f"Either add the model to `llm.models` or build a dedicated "
                f"LLM client for out-of-ensemble use."
            )
        return self._sample_model()

    @staticmethod
    def _model_aliases(model_name: Optional[str]) -> set[str]:
        """Return equivalent names for override matching across provider prefixes."""
        if not model_name:
            return set()
        raw = str(model_name)
        aliases = {raw}
        if "/" in raw:
            aliases.add(raw.split("/", 1)[1])
        else:
            for provider in ("anthropic", "openai", "openrouter"):
                aliases.add(f"{provider}/{raw}")
        return aliases

    def _sample_model(self) -> LLMInterface:
        """Sample a model from the ensemble based on weights"""
        index = self.random_state.choices(range(len(self.models)), weights=self.weights, k=1)[0]
        sampled_model = self.models[index]
        logger.info(f"Sampled model: {vars(sampled_model)['model']}")
        return sampled_model

    async def generate_multiple(self, prompt: str, n: int, **kwargs) -> List[str]:
        """Generate multiple texts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for _ in range(n)]
        return await asyncio.gather(*tasks)

    async def parallel_generate(self, prompts: List[str], **kwargs) -> List[str]:
        """Generate responses for multiple prompts in parallel"""
        tasks = [self.generate(prompt, **kwargs) for prompt in prompts]
        return await asyncio.gather(*tasks)

    async def generate_all_with_context(
        self, system_message: str, messages: List[Dict[str, str]], **kwargs
    ) -> str:
        """Generate text using a all available models and average their returned metrics"""
        responses = []
        for model in self.models:
            responses.append(await model.generate_with_context(system_message, messages, **kwargs))
        return responses
