"""Offline smoke test: uses only RouteLLM's built-in random router."""

from routellm.controller import Controller


def main() -> None:
    controller = Controller(
        routers=["random"],
        strong_model="strong-model",
        weak_model="weak-model",
    )
    prompt = "Return a short greeting."
    assert controller.route(prompt, "random", 0.0) == "strong-model"
    assert controller.route(prompt, "random", 1.0) == "weak-model"

    backend_controller = Controller(
        routers=["random"],
        strong_model="strong-model",
        weak_model="weak-model",
        strong_model_backend="openai/strong-model",
        weak_model_backend="openai/weak-model",
    )
    messages = [{"role": "user", "content": prompt}]
    assert (
        backend_controller._get_routed_model_for_completion(messages, "random", 0.0)
        == "openai/strong-model"
    )
    assert (
        backend_controller._get_routed_model_for_completion(messages, "random", 1.0)
        == "openai/weak-model"
    )

    split_backend_controller = Controller(
        routers=["random"],
        strong_model="strong-model",
        weak_model="weak-model",
        strong_model_backend="deepseek/deepseek-flash",
        weak_model_backend="ollama/qwen3:1.7b",
        strong_model_api_base="https://api.deepseek.example/v1",
        weak_model_api_base="http://127.0.0.1:11434",
        strong_model_api_key="strong-key",
        weak_model_api_key="ollama",
    )
    strong_backend = split_backend_controller._get_routed_backend_for_completion(
        messages, "random", 0.0
    )
    weak_backend = split_backend_controller._get_routed_backend_for_completion(
        messages, "random", 1.0
    )
    assert (strong_backend.model, strong_backend.api_base, strong_backend.api_key) == (
        "deepseek/deepseek-flash",
        "https://api.deepseek.example/v1",
        "strong-key",
    )
    assert (weak_backend.model, weak_backend.api_base, weak_backend.api_key) == (
        "ollama/qwen3:1.7b",
        "http://127.0.0.1:11434",
        "ollama",
    )
    print("PASS: random routing and separate backend model connections work offline.")


if __name__ == "__main__":
    main()
