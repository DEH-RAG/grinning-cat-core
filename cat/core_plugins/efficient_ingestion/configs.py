"""Factory configuration for the efficient ingestion engine.

Registers ``EfficientIngestionConfiguration`` in the ``ingestion`` category
through the ``factory_allowed_ingestions`` hook, so the engine can be selected
(replaceable class pattern, like the embedders).
"""


from cat.core_plugins.efficient_ingestion.reembed import EfficientIngestionEngine
from cat.services.factory.ingestion import BaseIngestionEngine
from cat.services.factory.models import BaseFactoryConfigModel


class EfficientIngestionConfiguration(BaseFactoryConfigModel):
    """Configuration of the efficient re-embed engine (category ``ingestion``)."""

    reembed_max_concurrency: int = 5

    @classmethod
    def pyclass(cls) -> type:
        return EfficientIngestionEngine

    @classmethod
    def base_class(cls) -> type:
        return BaseIngestionEngine
