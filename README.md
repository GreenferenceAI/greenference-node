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

```bash
cp .env.example .env
```

Edit `.env` — fill in your validator IP, hotkey, hardware specs, and HF token. See `.env.example` for all options with comments.

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

## environment variables

See `.env.example` for the full list with comments. Key ones:

| Variable | What to set |
|---|---|
| `GREENFERENCE_CONTROL_PLANE_URL` | Validator IP + port 28001 |
| `GREENFERENCE_MINER_VALIDATOR_URL` | Validator IP + port 28002 |
| `GREENFERENCE_MINER_HOTKEY` | Your Bittensor hotkey |
| `GREENFERENCE_GPU_MODEL` / `GPU_COUNT` / `VRAM_GB_PER_GPU` | Your actual hardware |
| `GREENFERENCE_INFERENCE_BACKEND` | `docker` (production) |
| `HF_TOKEN` | HuggingFace token for gated models |
| `GREENFERENCE_SSH_HOST` | Your public IP (for pod SSH) |

## directory structure

```
greenference-node/
  services/node-agent/src/     # agent source code
  images/diffusion/            # diffusion server Docker image source
  docker-compose.yml           # production docker compose
```
