import pytest
from apolo_app_types_fixtures.constants import (
    APP_ID,
    APP_SECRETS_NAME,
    CPU_POOL,
    DEFAULT_NAMESPACE,
)
from apolo_apps_stable_diffusion.inputs_processor import StableDiffusionInputsProcessor

from apolo_app_types import HuggingFaceModel, HuggingFaceToken, StableDiffusionInputs
from apolo_app_types.helm.apps.common import _get_match_expressions
from apolo_app_types.protocols.common import ApoloSecret, IngressHttp, Preset
from apolo_app_types.protocols.stable_diffusion import StableDiffusionParams


@pytest.mark.asyncio
async def test_values_sd_generation(setup_clients, mock_get_preset_cpu):
    apolo_client = setup_clients
    processor = StableDiffusionInputsProcessor(client=apolo_client)
    helm_params = await processor.gen_extra_values(
        input_=StableDiffusionInputs(
            preset=Preset(
                name="cpu-large",
            ),
            ingress_http=IngressHttp(),
            stable_diffusion=StableDiffusionParams(
                replica_count=1,
                stablestudio=None,
                hugging_face_model=HuggingFaceModel(
                    model_hf_name="test",
                    hf_token=HuggingFaceToken(
                        token_name="test-token-name", token=ApoloSecret(key="test3")
                    ),
                ),
            ),
        ),
        app_name="sd",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )

    assert helm_params["preset_name"] == "cpu-large"
    assert helm_params["api"]["resources"]["requests"]["cpu"] == "4000.0m"
    tolerations = helm_params["api"]["tolerations"]
    assert len(tolerations) == 3
    assert {
        "effect": "NoSchedule",
        "key": "platform.neuromation.io/job",
        "operator": "Exists",
    } in tolerations
    assert {
        "effect": "NoExecute",
        "key": "node.kubernetes.io/not-ready",
        "operator": "Exists",
        "tolerationSeconds": 300,
    } in tolerations
    assert {
        "effect": "NoExecute",
        "key": "node.kubernetes.io/unreachable",
        "operator": "Exists",
        "tolerationSeconds": 300,
    } in tolerations
    match_expressions = helm_params["api"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    assert match_expressions == _get_match_expressions([CPU_POOL])
    assert (
        "--docs --cors-origins=* --lowvram"
        in helm_params["api"]["env"]["COMMANDLINE_ARGS"]
    )
    assert helm_params["api"]["resources"]["requests"].get("nvidia.com/gpu") is None
    assert helm_params["api"]["resources"]["limits"].get("nvidia.com/gpu") is None

    # Verify Stable Diffusion gets ONLY auth middleware (no strip headers)
    assert (
        helm_params["ingress"]["annotations"][
            "traefik.ingress.kubernetes.io/router.middlewares"
        ]
        == "platform-platform-control-plane-ingress-auth@kubernetescrd"
    )


@pytest.mark.asyncio
async def test_values_sd_generation_with_gpu(setup_clients, mock_get_preset_gpu):
    apolo_client = setup_clients
    processor = StableDiffusionInputsProcessor(client=apolo_client)
    helm_params = await processor.gen_extra_values(
        input_=StableDiffusionInputs(
            preset=Preset(
                name="gpu-small",
            ),
            ingress_http=IngressHttp(),
            stable_diffusion=StableDiffusionParams(
                replica_count=1,
                stablestudio=None,
                hugging_face_model=HuggingFaceModel(
                    model_hf_name="test",
                    hf_token=HuggingFaceToken(
                        token_name="test-token-name", token=ApoloSecret(key="test3")
                    ),
                ),
            ),
        ),
        app_name="sd",
        namespace=DEFAULT_NAMESPACE,
        app_secrets_name=APP_SECRETS_NAME,
        app_id=APP_ID,
    )
    assert helm_params["preset_name"] == "gpu-small"
    assert helm_params["api"]["resources"]["requests"]["cpu"] == "2000.0m"
    assert helm_params["api"]["resources"]["requests"]["nvidia.com/gpu"] == "1"
    tolerations = helm_params["api"]["tolerations"]
    assert len(tolerations) == 4
    assert {
        "effect": "NoSchedule",
        "key": "platform.neuromation.io/job",
        "operator": "Exists",
    } in tolerations
    assert {
        "effect": "NoExecute",
        "key": "node.kubernetes.io/not-ready",
        "operator": "Exists",
        "tolerationSeconds": 300,
    } in tolerations
    assert {
        "effect": "NoExecute",
        "key": "node.kubernetes.io/unreachable",
        "operator": "Exists",
        "tolerationSeconds": 300,
    } in tolerations
    assert {
        "effect": "NoSchedule",
        "key": "nvidia.com/gpu",
        "operator": "Exists",
    } in tolerations
    match_expressions = helm_params["api"]["affinity"]["nodeAffinity"][
        "requiredDuringSchedulingIgnoredDuringExecution"
    ]["nodeSelectorTerms"][0]["matchExpressions"]
    assert match_expressions == _get_match_expressions(["gpu_pool"])
