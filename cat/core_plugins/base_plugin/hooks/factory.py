
from cat import (
    AgenticWorkflowConfig,
    AuthHandlerConfig,
    ChunkerSettings,
    ContextRetrieverSettings,
    EmbedderSettings,
    FileManagerConfig,
    LLMSettings,
    VectorDatabaseSettings,
    hook,
)
from cat.core_plugins.base_plugin.embedders.configs import EmbedderFakeConfig
from cat.core_plugins.base_plugin.file_managers.configs import LocalFileManagerConfig


@hook(priority=0)
def factory_allowed_ingestions(allowed, lizard):
    """Hook to extend the list of supported ingestion (re-embed) engines.

    The ``efficient_ingestion`` core plugin registers its
    ``EfficientIngestionConfiguration`` here. No-op default.
    """
    return allowed


@hook(priority=0)
def factory_allowed_llms(allowed: list[LLMSettings], cat) -> list:
    """
    Hook to extend support of LLMs.

    Args:
        allowed: List of LLMSettings classes
        cat: CheshireCat instance

    Returns:
        list of allowed LLMSettings classes for the allowed language models
    """
    return allowed


@hook(priority=0)
def factory_allowed_embedders(allowed: list[EmbedderSettings], lizard) -> list:
    """Hook to extend list of supported embedders.

    Args:
        allowed: List of EmbedderSettings classes
        lizard: BillTheLizard instance

    Returns:
        list of allowed EmbedderSettings classes for the allowed embedders
    """
    return allowed + [EmbedderFakeConfig]


@hook(priority=0)
def factory_allowed_auth_handlers(allowed: list[AuthHandlerConfig], cat) -> list:
    """Hook to extend list of supported auth handlers.

    Args:
        allowed: List of AuthHandlerConfig classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of AuthHandlerConfig classes for the allowed auth_handlers
    """
    return allowed


@hook(priority=0)
def factory_allowed_file_managers(allowed: list[FileManagerConfig], cat) -> list:
    """Hook to extend list of supported file managers.

    Args:
        allowed: List of FileManagerConfig classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of FileManagerConfig classes for the allowed file managers
    """
    return allowed + [LocalFileManagerConfig]


@hook(priority=0)
def factory_allowed_chunkers(allowed: list[ChunkerSettings], cat) -> list:
    """Hook to extend list of supported chunkers.

    Args:
        allowed: List of ChunkerSettings classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of ChunkerSettings classes for the allowed chunkers
    """
    return allowed


@hook(priority=0)
def factory_allowed_context_retrievers(allowed: list[ContextRetrieverSettings], cat) -> list:
    """Hook to extend list of supported context retrievers.

    Args:
        allowed: List of ContextRetrieverSettings classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of ContextRetrieverSettings classes for the allowed chunkers
    """
    return allowed


@hook(priority=0)
def factory_allowed_vector_databases(allowed: list[VectorDatabaseSettings], cat) -> list:
    """Hook to extend list of supported vector databases.

    Args:
        allowed: List of VectorDatabaseSettings classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of VectorDatabaseSettings classes for the allowed vector databases
    """
    return allowed


@hook(priority=0)
def factory_allowed_agentic_workflows(allowed: list[AgenticWorkflowConfig], cat) -> list:
    """Hook to extend list of supported agentic workflows.

    Args:
        allowed: List of AgenticWorkflowConfig classes
        cat: Cheshire Cat instance

    Returns:
        supported: List of AgenticWorkflowConfig classes for the allowed agentic_workflows
    """
    return allowed
