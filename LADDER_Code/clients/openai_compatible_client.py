"""
OpenAI-Compatible API Client

适配所有兼容 OpenAI API 格式的国内大模型厂商：
  - 阿里云百炼 (DashScope):   base_url = https://dashscope.aliyuncs.com/compatible-mode/v1
  - DeepSeek:                base_url = https://api.deepseek.com
  - 智谱AI (ZhipuAI):        base_url = https://open.bigmodel.cn/api/paas/v4
  - 月之暗面 (Moonshot):      base_url = https://api.moonshot.cn/v1
  - 硅基流动 (SiliconFlow):   base_url = https://api.siliconflow.cn/v1
"""

import os
from typing import Dict, Any, List
from openai import OpenAI
from .config import load_env_manually, get_env_path


# 预设的国内 API 厂商配置
DOMESTIC_PROVIDERS = {
    "dashscope": {
        "base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "env_key": "DASHSCOPE_API_KEY",
        "models": [
            "qwen3-235b-a22b",
            "qwen-max",
            "qwen-plus",
            "qwen-turbo",
            "qwen3-32b",
            "qwen3-8b",
        ],
        "default_model": "qwen-plus",
    },
    "deepseek": {
        "base_url": "https://api.deepseek.com",
        "env_key": "DEEPSEEK_API_KEY",
        "models": ["deepseek-chat", "deepseek-reasoner"],
        "default_model": "deepseek-chat",
    },
    "zhipu": {
        "base_url": "https://open.bigmodel.cn/api/paas/v4",
        "env_key": "ZHIPU_API_KEY",
        "models": ["glm-4-plus", "glm-4-flash", "glm-4", "glm-4-air"],
        "default_model": "glm-4-flash",
    },
    "moonshot": {
        "base_url": "https://api.moonshot.cn/v1",
        "env_key": "MOONSHOT_API_KEY",
        "models": ["moonshot-v1-8k", "moonshot-v1-32k", "moonshot-v1-128k"],
        "default_model": "moonshot-v1-32k",
    },
    "siliconflow": {
        "base_url": "https://api.siliconflow.cn/v1",
        "env_key": "SILICONFLOW_API_KEY",
        "models": [
            "Qwen/Qwen3-235B-A22B",
            "Qwen/Qwen3-32B",
            "Qwen/Qwen3-8B",
            "deepseek-ai/DeepSeek-V3",
            "deepseek-ai/DeepSeek-R1",
            "Pro/zai-org/GLM-4.7",
        ],
        "default_model": "Qwen/Qwen3-235B-A22B",
    },
    "apinebula": {
        "base_url": "https://apinebula.com/v1",
        "env_key": "APINEBULA_API_KEY",
        "models": [
            "gpt-5.4",
            "gpt-5.4-mini",
            "gpt-5.5",
            "gpt-5.5-openai-compact",
        ],
        "default_model": "gpt-5.4-mini",
    },
}


class OpenAICompatibleClient:
    """
    通用的 OpenAI 兼容 API 客户端。

    通过传入 provider 别名或 base_url + api_key 来适配任意厂商。
    只需 client 有 chat_messages() 方法即可被 adapters.py 适配为 LangGraph 可用的 LLM。
    """

    def __init__(
        self,
        api_key: str = None,
        base_url: str = None,
        model: str = "qwen-plus",
        temperature: float = 1.0,
        max_completion_tokens: int = 8000,
        provider: str = None,
    ):
        """
        初始化客户端。

        Args:
            api_key: API key。为 None 时尝试从 .env 或环境变量读取
            base_url: API 端点地址。为 None 时从 DOMESTIC_PROVIDERS 查找
            model: 模型名称
            temperature: 采样温度，论文使用 1.0
            max_completion_tokens: 最大输出 token 数
            provider: 预设厂商别名 (dashscope/deepseek/zhipu/moonshot/siliconflow)
        """
        self.temperature = temperature
        self.max_completion_tokens = max_completion_tokens

        # ----- 根据 provider 别名自动填 base_url 和 api_key -----
        if provider and provider in DOMESTIC_PROVIDERS:
            cfg = DOMESTIC_PROVIDERS[provider]
            if base_url is None:
                base_url = cfg["base_url"]
            # 如果 model 还是默认值，换成该厂商的默认模型
            if model == "qwen-plus":
                self.model = cfg["default_model"]
            else:
                self.model = model
            if api_key is None:
                env_config = load_env_manually(get_env_path())
                api_key = env_config.get(cfg["env_key"]) or os.environ.get(
                    cfg["env_key"]
                )
        else:
            self.model = model

        # 兜底：尝试从环境变量直接读
        if not api_key:
            env_config = load_env_manually(get_env_path())
            for env_name in [
                "OPENAI_API_KEY",
                "DASHSCOPE_API_KEY",
                "DEEPSEEK_API_KEY",
                "ZHIPU_API_KEY",
                "MOONSHOT_API_KEY",
                "SILICONFLOW_API_KEY",
                "APINEBULA_API_KEY",
            ]:
                api_key = env_config.get(env_name) or os.environ.get(env_name)
                if api_key:
                    break

        if not api_key:
            raise ValueError(
                "API key not found. Please set the appropriate key in .env file.\n"
                f"Supported providers: {list(DOMESTIC_PROVIDERS.keys())}\n"
                "Or set one of: DASHSCOPE_API_KEY, DEEPSEEK_API_KEY, ZHIPU_API_KEY, "
                "MOONSHOT_API_KEY, SILICONFLOW_API_KEY"
            )

        self.client = OpenAI(api_key=api_key, base_url=base_url)
        self.base_url = base_url or "(default)"

        print(f"OpenAI Compatible client initialized")
        print(f"   Provider: {provider or 'custom'}")
        print(f"   Base URL: {self.base_url}")
        print(f"   Model: {self.model}")
        print(f"   Temperature: {temperature}")

    # ─── 核心接口 ───────────────────────────────────────────
    # adapters.py 只需要 chat_messages(messages: List[dict]) -> str

    def chat_messages(self, messages: List[dict]) -> str:
        """
        发送 raw chat messages，返回模型回复文本。

        这是 adapters.py 所需的核心接口。
        """
        response = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
            max_completion_tokens=self.max_completion_tokens,
        )
        return (response.choices[0].message.content or "").strip()

    # ─── 兼容旧版调用方式 ───────────────────────────────────

    def answer_question(
        self,
        question: str,
        system_prompt: str,
        attack_prefix: str = "",
        context: Dict[str, Any] = None,
    ) -> str:
        """
        回答单个问题（兼容 openai_client.OpenAIAnswerer 的调用方式）。

        Args:
            question: 问题文本
            system_prompt: 系统提示词
            attack_prefix: 越狱攻击前缀
            context: 额外上下文

        Returns:
            模型回复文本
        """
        user_prompt = f"{attack_prefix}\n\nQuestion: {question}"

        if context:
            user_prompt += (
                f"\n\nKnowledge context: {context.get('Knowledge_name', '')}\n"
            )

        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ]

        try:
            print(f"Calling model: {self.model}")
            response = self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                temperature=self.temperature,
                max_completion_tokens=self.max_completion_tokens,
            )
            answer = response.choices[0].message.content.strip()

            if hasattr(response, "usage") and response.usage:
                usage = response.usage
                print(
                    f"  Token usage: input={usage.prompt_tokens}, "
                    f"output={usage.completion_tokens}"
                )

            return answer
        except Exception as e:
            error_msg = f"API call failed: {str(e)}"
            print(f"  Error: {error_msg}")
            return error_msg

    # ─── 辅助方法 ───────────────────────────────────────────

    def test_connection(self) -> bool:
        """测试 API 连接是否正常。"""
        try:
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": "Hello"}],
                max_tokens=10,
            )
            print(f"Connection test successful for model: {self.model}")
            return True
        except Exception as e:
            print(f"Connection test failed: {e}")
            return False


# ─── 便利工厂函数 ───────────────────────────────────────────


def create_domestic_client(
    provider: str = "dashscope", model: str = None, **kwargs
) -> OpenAICompatibleClient:
    """
    快速创建国内 API 客户端。

    Args:
        provider: 厂商别名 (dashscope/deepseek/zhipu/moonshot/siliconflow)
        model: 模型名，None 则使用该厂商的默认模型
        **kwargs: 传给 OpenAICompatibleClient 的额外参数

    Returns:
        OpenAICompatibleClient 实例
    """
    return OpenAICompatibleClient(
        model=model or DOMESTIC_PROVIDERS.get(provider, {}).get("default_model", ""),
        provider=provider,
        **kwargs,
    )


if __name__ == "__main__":
    print("Testing OpenAI Compatible client...")
    print("Available providers:", list(DOMESTIC_PROVIDERS.keys()))
    print()
    print("Usage:")
    print("  from clients.openai_compatible_client import create_domestic_client")
    print("  client = create_domestic_client('deepseek')")
    print("  client = create_domestic_client('dashscope', model='qwen-max')")
