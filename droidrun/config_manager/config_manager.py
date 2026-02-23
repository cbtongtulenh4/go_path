from __future__ import annotations

from dataclasses import asdict, dataclass, field, fields
from typing import Any, Dict, List, Optional

import yaml

from droidrun.config_manager.path_resolver import PathResolver
from droidrun.config_manager.safe_execution import SafeExecutionConfig



# ---------- Config Schema ----------
@dataclass
class LLMProfile:
    """LLM profile configuration."""

    provider: str = "GoogleGenAI"
    model: str = "gemini-2.5-pro"
    temperature: float = 0.2
    base_url: Optional[str] = None
    api_base: Optional[str] = None
    kwargs: Dict[str, Any] = field(default_factory=dict)

    def to_load_llm_kwargs(self) -> Dict[str, Any]:
        """Convert profile to kwargs for load_llm function."""
        result = {
            "model": self.model,
            "temperature": self.temperature,
        }
        # Add optional URL parameters
        if self.base_url:
            result["base_url"] = self.base_url
        if self.api_base:
            result["api_base"] = self.api_base
        # Merge additional kwargs
        result.update(self.kwargs)
        return result


@dataclass
class FastAgentConfig:
    vision: bool = False
    codeact: bool = False
    parallel_tools: bool = True
    system_prompt: str = "config/prompts/codeact/tools_system.jinja2"
    user_prompt: str = "config/prompts/codeact/tools_user.jinja2"
    safe_execution: bool = False
    execution_timeout: float = 50.0


@dataclass
class ManagerConfig:
    vision: bool = False
    system_prompt: str = "config/prompts/manager/system.jinja2"
    stateless: bool = False


@dataclass
class ExecutorConfig:
    vision: bool = False
    system_prompt: str = "config/prompts/executor/system.jinja2"


@dataclass
class ScripterConfig:
    enabled: bool = True
    max_steps: int = 10
    execution_timeout: float = 30.0
    system_prompt: str = "config/prompts/scripter/system.jinja2"
    safe_execution: bool = False


@dataclass
class AppCardConfig:
    """App card configuration."""

    enabled: bool = True
    mode: str = "local"  # local | server | composite
    app_cards_dir: str = "config/app_cards"
    server_url: Optional[str] = None
    server_timeout: float = 2.0
    server_max_retries: int = 2


@dataclass
class AgentConfig:
    name: str = "droidrun"
    max_steps: int = 15
    reasoning: bool = False
    streaming: bool = True
    after_sleep_action: float = 1.0
    wait_for_stable_ui: float = 0.3
    use_normalized_coordinates: bool = False

    fast_agent: FastAgentConfig = field(default_factory=FastAgentConfig)
    manager: ManagerConfig = field(default_factory=ManagerConfig)
    executor: ExecutorConfig = field(default_factory=ExecutorConfig)
    scripter: ScripterConfig = field(default_factory=ScripterConfig)
    app_cards: AppCardConfig = field(default_factory=AppCardConfig)

    def get_fast_agent_system_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.fast_agent.system_prompt, must_exist=True))

    def get_fast_agent_user_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.fast_agent.user_prompt, must_exist=True))

    def get_manager_system_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.manager.system_prompt, must_exist=True))

    def get_executor_system_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.executor.system_prompt, must_exist=True))

    def get_scripter_system_prompt_path(self) -> str:
        return str(PathResolver.resolve(self.scripter.system_prompt, must_exist=True))


@dataclass
class DeviceConfig:
    """Device-related configuration."""

    serial: Optional[str] = None
    use_tcp: bool = False
    platform: str = "android"  # "android" or "ios"





@dataclass
class LoggingConfig:
    """Logging configuration."""

    debug: bool = False
    rich_text: bool = False


def _default_disabled_tools() -> List[str]:
    return ["click_at", "click_area", "long_press_at"]


@dataclass
class ToolsConfig:
    """Tools configuration."""

    disabled_tools: List[str] = field(default_factory=_default_disabled_tools)
    stealth: bool = False


@dataclass
class CredentialsConfig:
    """Credentials configuration."""

    enabled: bool = False
    file_path: str = "config/credentials.yaml"


@dataclass
class DroidrunConfig:
    """Complete DroidRun configuration schema."""

    agent: AgentConfig = field(default_factory=AgentConfig)
    llm_profiles: Dict[str, LLMProfile] = field(default_factory=dict)
    device: DeviceConfig = field(default_factory=DeviceConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    tools: ToolsConfig = field(default_factory=ToolsConfig)
    credentials: CredentialsConfig = field(default_factory=CredentialsConfig)
    safe_execution: SafeExecutionConfig = field(default_factory=SafeExecutionConfig)
    external_agents: Dict[str, Dict[str, Any]] = field(default_factory=dict)


    def __post_init__(self):
        """Ensure default profiles exist."""
        if not self.llm_profiles:
            self.llm_profiles = self._default_profiles()

    @staticmethod
    def _default_profiles() -> Dict[str, LLMProfile]:
        """Get default agent specific LLM profiles."""
        return {
            "manager": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-2.5-pro",
                temperature=0.2,
                kwargs={},
            ),
            "executor": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-2.5-pro",
                temperature=0.1,
                kwargs={},
            ),
            "fast_agent": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-2.5-pro",
                temperature=0.2,
                kwargs={},
            ),
            "text_manipulator": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-2.5-pro",
                temperature=0.3,
                kwargs={},
            ),
            "app_opener": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-2.5-pro",
                temperature=0.0,
                kwargs={},
            ),
            "scripter": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-2.5-flash",
                temperature=0.1,
                kwargs={},
            ),
            "structured_output": LLMProfile(
                provider="GoogleGenAI",
                model="gemini-2.5-flash",
                temperature=0.0,
                kwargs={},
            ),
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert config to dictionary."""
        result = asdict(self)
        # Convert LLMProfile objects to dicts
        result["llm_profiles"] = {
            name: asdict(profile) for name, profile in self.llm_profiles.items()
        }
        # safe_execution is already converted by asdict
        return result

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "DroidrunConfig":
        """Create config from dictionary."""
        # Parse LLM profiles
        llm_profiles = {}
        for name, profile_data in data.get("llm_profiles", {}).items():
            llm_profiles[name] = LLMProfile(**profile_data)

        # Parse agent config with sub-configs
        agent_data = data.get("agent", {})

        fast_agent_data = agent_data.get("fast_agent", {})
        fast_agent_config = (
            FastAgentConfig(**fast_agent_data) if fast_agent_data else FastAgentConfig()
        )

        manager_data = agent_data.get("manager", {})
        manager_config = (
            ManagerConfig(**manager_data) if manager_data else ManagerConfig()
        )

        executor_data = agent_data.get("executor", {})
        executor_config = (
            ExecutorConfig(**executor_data) if executor_data else ExecutorConfig()
        )

        script_data = agent_data.get("scripter", {})
        scripter_config = (
            ScripterConfig(**script_data) if script_data else ScripterConfig()
        )

        app_cards_data = agent_data.get("app_cards", {})
        app_cards_config = (
            AppCardConfig(**app_cards_data) if app_cards_data else AppCardConfig()
        )

        agent_config = AgentConfig(
            name=agent_data.get("name", "droidrun"),
            max_steps=agent_data.get("max_steps", 15),
            reasoning=agent_data.get("reasoning", False),
            streaming=agent_data.get("streaming", False),
            after_sleep_action=agent_data.get("after_sleep_action", 1.0),
            wait_for_stable_ui=agent_data.get("wait_for_stable_ui", 0.3),
            use_normalized_coordinates=agent_data.get(
                "use_normalized_coordinates", False
            ),
            fast_agent=fast_agent_config,
            manager=manager_config,
            executor=executor_config,
            scripter=scripter_config,
            app_cards=app_cards_config,
        )

        safe_exec_data = data.get("safe_execution", {})
        safe_execution_config = (
            SafeExecutionConfig(**safe_exec_data)
            if safe_exec_data
            else SafeExecutionConfig()
        )

        # External agents config - just pass through as-is
        external_agents = data.get("external_agents", {})



        def get_valid_kwargs(config_cls, data_dict):
            valid_keys = {f.name for f in fields(config_cls)}
            return {k: v for k, v in data_dict.items() if k in valid_keys}

        return cls(
            agent=agent_config,
            llm_profiles=llm_profiles,
            device=DeviceConfig(**get_valid_kwargs(DeviceConfig, data.get("device", {}))),
            logging=LoggingConfig(**get_valid_kwargs(LoggingConfig, data.get("logging", {}))),
            tools=ToolsConfig(**get_valid_kwargs(ToolsConfig, data.get("tools", {}))),
            credentials=CredentialsConfig(**get_valid_kwargs(CredentialsConfig, data.get("credentials", {}))),
            safe_execution=safe_execution_config,
            external_agents=external_agents,
        )

    @classmethod
    def from_yaml(cls, path: str) -> "DroidrunConfig":
        """
        Load config from YAML file.

        Args:
            path: Path to config file (relative to CWD or absolute)

        Returns:
            DroidrunConfig instance

        Raises:
            FileNotFoundError: If file doesn't exist
            Exception: If file can't be parsed
        """
        with open(path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        return cls.from_dict(data)
