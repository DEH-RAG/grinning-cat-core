from pydantic import BaseModel, Field


class SentimentResult(BaseModel):
    sentiment: str = Field(description="The detected sentiment, one of positive, negative or neutral.")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence score between 0 and 1.")
    reason: str = Field(description="Human-readable reason supporting the classification.")


class ExtractedEntity(BaseModel):
    name: str = Field(description="Entity name as found in the text.")
    entity_type: str = Field(description="Entity type, e.g. person, organization, location.")
    mentions: int = Field(ge=0, description="Number of mentions of this entity in the text.")