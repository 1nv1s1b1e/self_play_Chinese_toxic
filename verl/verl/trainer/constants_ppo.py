# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import os

from ray._private.runtime_env.constants import RAY_JOB_CONFIG_JSON_ENV_VAR

PPO_RAY_RUNTIME_ENV = {
    "env_vars": {
        "TOKENIZERS_PARALLELISM": "true",
        "NCCL_DEBUG": "WARN",
        "VLLM_LOGGING_LEVEL": "WARN",
        "VLLM_ALLOW_RUNTIME_LORA_UPDATING": "true",
        "CUDA_DEVICE_MAX_CONNECTIONS": "1",
        # To prevent hanging or crash during synchronization of weights between actor and rollout
        # in disaggregated mode. See:
        # https://docs.vllm.ai/en/latest/usage/troubleshooting.html?h=nccl_cumem_enable#known-issues
        # https://github.com/vllm-project/vllm/blob/c6b0a7d3ba03ca414be1174e9bd86a97191b7090/vllm/worker/worker_base.py#L445
        "NCCL_CUMEM_ENABLE": "0",
        # TODO: disable compile cache due to cache corruption issue
        # https://github.com/vllm-project/vllm/issues/31199
        "VLLM_DISABLE_COMPILE_CACHE": "1",
        # Needed for multi-processes colocated on same NPU device
        # https://www.hiascend.com/document/detail/zh/canncommercial/83RC1/maintenref/envvar/envref_07_0143.html
        "HCCL_HOST_SOCKET_PORT_RANGE": "auto",
        "HCCL_NPU_SOCKET_PORT_RANGE": "auto",
    },
}


def get_ppo_ray_runtime_env():
    """
    A filter function to return the PPO Ray runtime environment.
    To avoid repeat of some environment variables that are already set.
    """
    working_dir = (
        json.loads(os.environ.get(RAY_JOB_CONFIG_JSON_ENV_VAR, "{}")).get("runtime_env", {}).get("working_dir", None)
    )

    runtime_env = {
        "env_vars": PPO_RAY_RUNTIME_ENV["env_vars"].copy(),
        **({"working_dir": None} if working_dir is None else {}),
    }
    for key in list(runtime_env["env_vars"].keys()):
        if os.environ.get(key) is not None:
            runtime_env["env_vars"].pop(key, None)

    # ── 昇腾 NPU / HCCL 必需变量: 不受上方过滤逻辑影响 ──────────────────────
    # 问题根因: get_ppo_ray_runtime_env() 会把已存在于父进程 os.environ 的 key
    # 从 runtime_env 中删除，假设 Ray worker 会继承父进程 env。但在 Ascend/ModelArts
    # 等平台上 Ray worker 不保证继承所有父进程 env，导致 HCCL 安全开关丢失，
    # 从而引发 hcclCommInitRootInfoConfig error code 5。
    # 方案: 在过滤之后强制覆盖写入，保证这些 key 始终出现在 Ray worker env 中。
    _hccl_required = {
        "HCCL_WHITELIST_DISABLE": "1",       # 关闭白名单验证，闭合网络必需
        "HCCL_SECURITY_INFO_DISABLE": "1",   # 关闭安全握手，避免 init 被拒绝
    }
    # 将父进程探测到的 HCCL_IF_IP / HCCL_SOCKET_IFNAME 传递给 Ray worker
    for _k in ("HCCL_IF_IP", "HCCL_SOCKET_IFNAME", "HCCL_CONNECT_TIMEOUT", "HCCL_EXEC_TIMEOUT"):
        _v = os.environ.get(_k, "")
        if _v and _v not in ("127.0.0.1", "lo", ""):
            _hccl_required[_k] = _v
    runtime_env["env_vars"].update(_hccl_required)

    return runtime_env
