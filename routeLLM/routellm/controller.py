from collections import defaultdict
from dataclasses import dataclass
from types import SimpleNamespace
from typing import Any, Optional

import pandas as pd
from litellm import acompletion, completion
from tqdm import tqdm

from routellm.routers.routers import ROUTER_CLS

# Default config for routers augmented using golden label data from GPT-4.
# This is exactly the same as config.example.yaml.
GPT_4_AUGMENTED_CONFIG = {
    "sw_ranking": {
        "arena_battle_datasets": [
            "lmsys/lmsys-arena-human-preference-55k",
            "routellm/gpt4_judge_battles",
        ],
        "arena_embedding_datasets": [
            "routellm/arena_battles_embeddings",
            "routellm/gpt4_judge_battles_embeddings",
        ],
    },
    "causal_llm": {"checkpoint_path": "routellm/causal_llm_gpt4_augmented"},
    "bert": {"checkpoint_path": "routellm/bert_gpt4_augmented"},
    "mf": {"checkpoint_path": "routellm/mf_gpt4_augmented"},
}


class RoutingError(Exception):
    pass


@dataclass
class ModelPair:
    strong: str
    weak: str


@dataclass
class ModelBackend:
    """Connection settings for one routed completion backend."""

    model: str
    api_base: Optional[str]
    api_key: Optional[str]


class Controller:
    def __init__(
        self,
        routers: list[str],
        strong_model: str,
        weak_model: str,
        strong_model_backend: Optional[str] = None,
        weak_model_backend: Optional[str] = None,
        strong_model_api_base: Optional[str] = None,
        weak_model_api_base: Optional[str] = None,
        strong_model_api_key: Optional[str] = None,
        weak_model_api_key: Optional[str] = None,
        embedding_api_base: Optional[str] = None,
        embedding_api_key: Optional[str] = None,
        embedding_model: Optional[str] = None,
        config: Optional[dict[str, dict[str, Any]]] = None,
        api_base: Optional[str] = None,
        api_key: Optional[str] = None,
        progress_bar: bool = False,
    ):
        # Router implementations score the names in ``model_pair``.  Some
        # routers (notably MF) require these exact, training-time model names.
        self.model_pair = ModelPair(strong=strong_model, weak=weak_model)
        # LiteLLM may need a provider-qualified name for the actual completion
        # request, e.g. ``openai/chatglm3-6b`` for an OpenAI-compatible proxy.
        # Keep it separate from the router-scoring names above.
        self.backend_model_pair = ModelPair(
            strong=strong_model_backend or strong_model,
            weak=weak_model_backend or weak_model,
        )
        self.strong_backend = ModelBackend(
            model=self.backend_model_pair.strong,
            api_base=strong_model_api_base or api_base,
            api_key=strong_model_api_key or api_key,
        )
        self.weak_backend = ModelBackend(
            model=self.backend_model_pair.weak,
            api_base=weak_model_api_base or api_base,
            api_key=weak_model_api_key or api_key,
        )
        self.routers = {}
        self.api_base = api_base
        self.api_key = api_key
        self.model_counts = defaultdict(lambda: defaultdict(int))
        self.progress_bar = progress_bar
        self.embedding_api_base = embedding_api_base
        self.embedding_api_key = embedding_api_key
        self.embedding_model = embedding_model

        if config is None:
            config = GPT_4_AUGMENTED_CONFIG

        router_pbar = None
        if progress_bar:
            router_pbar = tqdm(routers)
            tqdm.pandas()

        for router in routers:
            if router_pbar is not None:
                router_pbar.set_description(f"Loading {router}")
            router_config = dict(config.get(router, {}))
            if router == "mf":
                if self.embedding_api_base is not None:
                    router_config["embedding_api_base"] = self.embedding_api_base
                if self.embedding_api_key is not None:
                    router_config["embedding_api_key"] = self.embedding_api_key
                if self.embedding_model is not None:
                    router_config["embedding_model"] = self.embedding_model
            self.routers[router] = ROUTER_CLS[router](**router_config)

        # Some Python magic to match the OpenAI Python SDK
        self.chat = SimpleNamespace(
            completions=SimpleNamespace(
                create=self.completion, acreate=self.acompletion
            )
        )

    def _validate_router_threshold(
        self, router: Optional[str], threshold: Optional[float]
    ):
        if router is None or threshold is None:
            raise RoutingError("Router or threshold unspecified.")
        if router not in self.routers:
            raise RoutingError(
                f"Invalid router {router}. Available routers are {list(self.routers.keys())}."
            )
        if not 0 <= threshold <= 1:
            raise RoutingError(
                f"Invalid threshold {threshold}. Threshold must be a float between 0.0 and 1.0."
            )

    def _parse_model_name(self, model: str):
        _, router, threshold = model.split("-", 2)
        try:
            threshold = float(threshold)
        except ValueError as e:
            raise RoutingError(f"Threshold {threshold} must be a float.") from e
        if not model.startswith("router"):
            raise RoutingError(
                f"Invalid model {model}. Model name must be of the format 'router-[router name]-[threshold]."
            )
        return router, threshold

    def _get_routed_model_for_completion(
        self, messages: list, router: str, threshold: float
    ):
        # Look at the last turn for routing.
        # Our current routers were only trained on first turn data, so more research is required here.
        prompt = messages[-1]["content"]
        routed_model = self.routers[router].route(prompt, threshold, self.model_pair)

        self.model_counts[router][routed_model] += 1

        if routed_model == self.model_pair.strong:
            return self.backend_model_pair.strong
        if routed_model == self.model_pair.weak:
            return self.backend_model_pair.weak
        raise RoutingError(f"Router returned an unknown model: {routed_model}")

    def _get_routed_backend_for_completion(
        self, messages: list, router: str, threshold: float
    ) -> ModelBackend:
        """Route a request and return both its model name and connection settings."""
        # Keep the public string-returning helper above for existing callers.
        routed_model = self._get_routed_model_for_completion(messages, router, threshold)
        if routed_model == self.strong_backend.model:
            return self.strong_backend
        if routed_model == self.weak_backend.model:
            return self.weak_backend
        raise RoutingError(f"No backend configured for routed model: {routed_model}")

    # Mainly used for evaluations
    def batch_calculate_win_rate(
        self,
        prompts: pd.Series,
        router: str,
    ):
        self._validate_router_threshold(router, 0)
        router_instance = self.routers[router]
        if router_instance.NO_PARALLEL and self.progress_bar:
            return prompts.progress_apply(router_instance.calculate_strong_win_rate)
        elif router_instance.NO_PARALLEL:
            return prompts.apply(router_instance.calculate_strong_win_rate)
        else:
            return prompts.parallel_apply(router_instance.calculate_strong_win_rate)

    def route(self, prompt: str, router: str, threshold: float):
        self._validate_router_threshold(router, threshold)

        return self.routers[router].route(prompt, threshold, self.model_pair)

    # Matches OpenAI's Chat Completions interface, but also supports optional router and threshold args
    # If model name is present, attempt to parse router and threshold using it, otherwise, use the router and threshold args
    def completion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        backend = self._get_routed_backend_for_completion(
            kwargs["messages"], router, threshold
        )
        kwargs["model"] = backend.model
        return completion(
            api_base=backend.api_base,
            api_key=backend.api_key,
            drop_params=True,
            **kwargs,
        )

    # Matches OpenAI's Async Chat Completions interface, but also supports optional router and threshold args
    async def acompletion(
        self,
        *,
        router: Optional[str] = None,
        threshold: Optional[float] = None,
        **kwargs,
    ):
        if "model" in kwargs:
            router, threshold = self._parse_model_name(kwargs["model"])

        self._validate_router_threshold(router, threshold)
        backend = self._get_routed_backend_for_completion(
            kwargs["messages"], router, threshold
        )
        kwargs["model"] = backend.model
        return await acompletion(
            api_base=backend.api_base,
            api_key=backend.api_key,
            drop_params=True,
            **kwargs,
        )
