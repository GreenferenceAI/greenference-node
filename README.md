# greenference-node

Unified miner agent for the Greenference subnet. Handles inference (vLLM, diffusion), GPU pods, and VMs in a single daemon.

## quick start

### prerequisites

- Linux with NVIDIA GPU
- Docker Engine 20.10+
- NVIDIA Container Toolkit

```bash
# install nvidia container toolkit
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
  sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg

curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
  sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
  sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list

sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
```

### clone

```bash
mkdir greenference-ai && cd greenference-ai
git clone https://github.com/your-org/greenference.git        # protocol
git clone https://github.com/your-org/greenference-node.git   # this repo
cd greenference-node
```

### configure

Create `.env`:

```env
# Identity
GREENFERENCE_MINER_HOTKEY=your-bittensor-hotkey
GREENFERENCE_MINER_PAYOUT_ADDRESS=5F...your-payout-address
GREENFERENCE_MINER_AUTH_SECRET=shared-secret-from-validator
GREENFERENCE_MINER_NODE_ID=my-node-01

# Network (point to validator)
GREENFERENCE_CONTROL_PLANE_URL=http://VALIDATOR_IP:28001
GREENFERENCE_MINER_VALIDATOR_URL=http://VALIDATOR_IP:28002
GREENFERENCE_MINER_API_BASE_URL=http://YOUR_PUBLIC_IP:8007

# Hardware (match your actual specs)
GREENFERENCE_GPU_MODEL=rtx4090
GREENFERENCE_GPU_COUNT=1
GREENFERENCE_VRAM_GB_PER_GPU=24
GREENFERENCE_CPU_CORES=16
GREENFERENCE_MEMORY_GB=64

# Backends
GREENFERENCE_INFERENCE_BACKEND=docker
GREENFERENCE_POD_BACKEND=process
GREENFERENCE_SUPPORTED_WORKLOAD_KINDS=inference,pod

# HuggingFace (for gated models like Llama)
HF_TOKEN=hf_your_token_here

# SSH for pod access
GREENFERENCE_SSH_HOST=YOUR_PUBLIC_IP
GREENFERENCE_SSH_PORT_RANGE_START=30000
GREENFERENCE_SSH_PORT_RANGE_END=31000
```

### run

```bash
docker compose up -d
```

### verify

```bash
curl http://localhost:8007/readyz
```

### firewall

```
8007/tcp         - node-agent API
30000-31000/tcp  - SSH port range for pod tenants
```

## how it works

1. Agent registers with the control plane (GPU specs, supported workload kinds)
2. Sends heartbeats every second
3. Polls for lease assignments
4. When assigned a workload:
   - **Inference**: pulls vLLM or diffusion Docker image, starts container on GPU
   - **Pod**: starts Docker container with SSH access
   - **VM**: starts Firecracker microVM (if supported)
5. Reports status back to control plane
6. Validator probes your node for scoring

## inference runtimes

| Runtime | Docker Image | Models |
|---|---|---|
| vLLM | `vllm/vllm-openai:v0.7.3` | Llama 3, Mistral, Qwen, Phi |
| vLLM Vision | `vllm/vllm-openai:v0.7.3` | LLaVA, Qwen2.5-VL, Phi-3.5-vision |
| Diffusion | `ghcr.io/greenference/diffusion:latest` | SDXL, FLUX, Stable Diffusion 3 |

Images are pulled automatically. Models are cached in `~/.cache/huggingface`.

## updating

```bash
git pull
docker compose restart
```

## environment variables reference

| Variable | Default | Description |
|---|---|---|
| `GREENFERENCE_MINER_HOTKEY` | required | Bittensor hotkey |
| `GREENFERENCE_CONTROL_PLANE_URL` | required | Control plane URL |
| `GREENFERENCE_MINER_VALIDATOR_URL` | required | Validator URL |
| `GREENFERENCE_MINER_API_BASE_URL` | `http://127.0.0.1:8007` | Self-advertised URL |
| `GREENFERENCE_MINER_AUTH_SECRET` | required | Shared HMAC secret |
| `GREENFERENCE_GPU_MODEL` | `rtx4090` | GPU model name |
| `GREENFERENCE_GPU_COUNT` | `1` | Number of GPUs |
| `GREENFERENCE_VRAM_GB_PER_GPU` | `24` | VRAM per GPU in GB |
| `GREENFERENCE_CPU_CORES` | `32` | CPU cores |
| `GREENFERENCE_MEMORY_GB` | `128` | System RAM in GB |
| `GREENFERENCE_INFERENCE_BACKEND` | `docker` | `docker`, `process`, or `fallback` |
| `GREENFERENCE_POD_BACKEND` | `process` | `process`, `stub`, or `k8s` |
| `GREENFERENCE_VM_BACKEND` | `stub` | `stub` or `firecracker` |
| `GREENFERENCE_SUPPORTED_WORKLOAD_KINDS` | `inference,pod,vm` | CSV of workload types |
| `GREENFERENCE_AUTH_MODE` | `hmac` | `hmac` (dev) or `hotkey` (production) |
| `GREENFERENCE_SECURITY_TIER` | auto-detect | `standard`, `cpu_tee`, `cpu_gpu_attested` |
| `HF_TOKEN` | none | HuggingFace token for gated models |
| `GREENFERENCE_SSH_HOST` | `0.0.0.0` | Public IP for pod SSH |
| `GREENFERENCE_SSH_PORT_RANGE_START` | `30000` | SSH port pool start |
| `GREENFERENCE_SSH_PORT_RANGE_END` | `31000` | SSH port pool end |

## directory structure

```
greenference-node/
  services/node-agent/src/     # agent source code
  images/diffusion/            # diffusion server Docker image source
  docker-compose.yml           # production docker compose
```
