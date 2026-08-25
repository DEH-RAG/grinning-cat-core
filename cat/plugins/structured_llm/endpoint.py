from pydantic import BaseModel

from cat import endpoint
from cat.auth.connection import AuthorizedInfo
from cat.auth.permissions import AuthPermission, AuthResource, check_permissions
from cat.looking_glass.stray_cat import StrayCat
from cat.plugins.structured_llm.models import SentimentResult
from cat.plugins.structured_llm.structured_llm import StructuredLLM, StructuredLLMError


class ExtractRequest(BaseModel):
    text: str


@endpoint.post("/structured/extract")
async def structured_extract(
    request: ExtractRequest,
    info: AuthorizedInfo = check_permissions(
        AuthResource.AGENTIC_WORKFLOW, AuthPermission.READ
    ),
):
    cat: StrayCat = await StrayCat.from_cat(
        user_data=info.user, cat=info.cheshire_cat
    )

    try:
        result = await StructuredLLM().run(
            prompt=request.text,
            llm=cat.large_language_model,
            output_model=SentimentResult,
        )
        return {
            "json": result.text,
            "model": result.model.model_dump() if result.model else None,
        }
    except StructuredLLMError as e:
        return {"error": str(e), "raw": e.raw_text}
