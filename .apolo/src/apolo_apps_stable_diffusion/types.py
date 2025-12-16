from pydantic import ConfigDict, Field

from apolo_app_types.protocols.common import (
    AbstractAppFieldType,
    AppInputs,
    HuggingFaceModel,
    IngressHttp,
    Preset,
    SchemaExtraMetadata,
    SchemaMetaType,
)
from apolo_app_types.protocols.common.hugging_face import HF_SCHEMA_EXTRA
from apolo_app_types.protocols.stable_diffusion import StableDiffusionOutputs


class StableDiffusionParams(AbstractAppFieldType):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="Stable Diffusion",
            description="Configure the deployment settings"
            " and model selection for Stable Diffusion.",
        ).as_json_schema_extra(),
    )

    replica_count: int = Field(
        default=1,
        gt=0,
        json_schema_extra=SchemaExtraMetadata(
            title="Replica Count",
            description="Set the number of replicas to "
            "deploy for handling concurrent image generation requests.",
        ).as_json_schema_extra(),
    )
    hugging_face_model: HuggingFaceModel = Field(
        ...,
        json_schema_extra=HF_SCHEMA_EXTRA.model_copy(
            update={
                "meta_type": SchemaMetaType.INLINE,
            }
        ).as_json_schema_extra(),
    )


class StableDiffusionInputs(AppInputs):
    ingress_http: IngressHttp
    preset: Preset
    stable_diffusion: StableDiffusionParams


class SDModel(AbstractAppFieldType):
    model_config = ConfigDict(
        protected_namespaces=(),
        json_schema_extra=SchemaExtraMetadata(
            title="Stable Diffusion Model",
            description="Stable Diffusion Model hosted in application.",
            meta_type=SchemaMetaType.INTEGRATION,
        ).as_json_schema_extra(),
    )
    name: str
    files: str


__all__ = ["StableDiffusionParams", "StableDiffusionInputs", "StableDiffusionOutputs"]
