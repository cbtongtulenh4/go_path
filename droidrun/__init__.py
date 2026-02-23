"""
Droidrun - A framework for controlling Android devices through LLM agents.
"""

import logging
from importlib.metadata import version

__version__ = version("droidrun")

_logger = logging.getLogger("droidrun")
_logger.addHandler(logging.StreamHandler())
_logger.setLevel(logging.INFO)
_logger.propagate = False

# Import main classes for easier access
from droidrun.agent import ResultEvent
from droidrun.agent.droid import DroidAgent
from droidrun.agent.utils.llm_picker import load_llm

# Import configuration classes
from droidrun.config_manager import (
    # Agent configs
    AgentConfig,
    AppCardConfig,
    FastAgentConfig,
    CredentialsConfig,
    # Feature configs
    DeviceConfig,
    DroidrunConfig,
    ExecutorConfig,
    LLMProfile,
    LoggingConfig,
    ManagerConfig,
    SafeExecutionConfig,
    ScripterConfig,
    ToolsConfig,
)

from droidrun.tools import AndroidDriver, DeviceDriver, RecordingDriver

# Make main components available at package level
__all__ = [
    # Agent
    "DroidAgent",
    "load_llm",
    "ResultEvent",
    # Tools / Drivers
    "DeviceDriver",
    "AndroidDriver",
    "RecordingDriver",
    # Configuration
    "DroidrunConfig",
    "AgentConfig",
    "FastAgentConfig",
    "ManagerConfig",
    "ExecutorConfig",
    "ScripterConfig",
    "AppCardConfig",
    "DeviceConfig",
    "LoggingConfig",
    "ToolsConfig",
    "CredentialsConfig",
    "SafeExecutionConfig",
    "LLMProfile",
]
