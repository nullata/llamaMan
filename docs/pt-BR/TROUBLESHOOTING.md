# Solução de problemas

A tabela completa sintoma → correção. O [README](../../README.md#troubleshooting) mantém as nove linhas mais comuns do primeiro uso; todo o resto está aqui.

| Sintoma | Correção |
|---|---|
| Instância presa em **starting** | Consulte os logs pelo botão Logs. Comuns: OOM, erro de digitação no caminho do modelo, GGUF corrompido, imagem não baixada |
| _"Docker image not found"_ | Baixe a imagem correspondente: `docker pull ghcr.io/ggml-org/llama.cpp:server-cuda` (ou `server-rocm` / `server-sycl` / `server`) |
| _"Docker API error"_ no lançamento | Garanta que `/var/run/docker.sock` está montado (padrão no `docker-compose.yml`) |
| Sem GPU / erro CUDA | Garanta que o NVIDIA Container Toolkit está instalado e que `docker run --gpus all` funciona no host |
| Sem GPU / erro ROCm | Garanta que `/dev/kfd` e `/dev/dri` existem e que seu usuário está em `video`/`render` |
| Sem GPU / erro Intel Arc | Garanta que `/dev/dri` é acessível e que seu usuário está em `video`/`render` |
| A imagem Vulkan falha com _"Unable to find group render"_ | Defina `GPU_TYPE=vulkan` para que o llamaMan anexe os GIDs render/video do host numericamente (o group_add por nome falha em imagens cujo /etc/group não traz uma entrada `render`) |
| Estatísticas de GPU indisponíveis | NVIDIA: descomente o bloco `deploy.resources.reservations`. AMD/Intel: garanta que `/sys/class/drm:ro` está montado |
| Fabricante de GPU errado detectado | Defina `GPU_TYPE=cuda`/`rocm`/`intel` para sobrescrever |
| Instância presa em **starting** rodando bare-metal | O container está healthy, mas o llamaman não o alcança. Defina `LLAMAMAN_IN_DOCKER=false`/`true` explicitamente se a autodetecção estiver errada para o seu ambiente |
| Strata: _"Docker image 'strata:latest' not found"_ | Baixe ou construa uma imagem Strata em Settings → Docker Images (ou `docker build -t strata .` num clone do repositório Strata). Veja [Strata](STRATA.md) |
| Modelos Strata não listados | Defina `STRATA_ENABLED=true`; o node precisa ser NVIDIA (`GPU_TYPE=cuda` se a detecção falhar). `GET /api/engines` mostra o motivo |
| Instância Strata presa em **starting** | Normal por minutos na primeira inicialização (download + pack). O card mostra a etapa; o log mostra o setup do Strata. "setup failed" significa que o setup dele parou - leia a linha `[X]` do log |
| O modal de estatísticas está vazio | Ative **Settings → App Settings → Request recording** (veja [Gravação de requisições e estatísticas](CONFIGURATION.md#request-recording--stats)) |
| O lançamento falha com erro de GPU/CDI num host sem passthrough de GPU | Defina **GPU Layers** como `0` para CPU-only sem dispositivo GPU anexado, ou corrija o runtime de GPU |
| Conflito de porta | O formulário sugere automaticamente uma porta livre; ajuste se necessário |
| Modelo não aparece no OpenWebUI | Garanta `OLLAMA_BASE_URL=http://llamaman:42069`. Verifique se `/api/tags` retorna modelos |
| OpenWebUI recebe 401 | `require_auth` está ativado (padrão). Crie uma chave API e defina `OPENAI_API_KEYS` no OpenWebUI |
| Cliente OpenAI recebe 503 quando os modelos estão cheios | O teto `LLAMAMAN_MAX_MODELS` foi atingido e a API OpenAI não pode desalojar por padrão. Aumente o teto ou ative os interruptores de desalojo: [Desalojo de modelos](MODELS.md#model-eviction) |
| Containers não limpos após um stop | O llamaMan remove containers no stop. Órfãos após um crash: `docker ps --filter name=llamaman-`, ou reinicie o llamaMan (a adoção de órfãos roda na inicialização) |
| O cliente (Hermes / OpenWebUI / etc.) reporta a janela de contexto treinada em vez do teto do preset | Atualize para a 1.1.2+. O `/api/ps` agora inclui `context_length` definido ao ctx runtime com o qual a instância foi lançada, e o `model_info["<arch>.context_length"]` do `/api/show` é sobrescrito com o teto efetivo |
| Um cliente que consulta um nome de queue-group recebe o próprio ctx padrão (p. ex. 128k) em vez do ctx real do deployment | O `/api/show` e o `/v1/models` agora respondem por um queue-group em qualquer node em cluster — mesmo um sem membro local — usando o ctx runtime do peer do estado compartilhado. Grupos com vários membros reportam o mínimo entre os membros ativos para que o valor anunciado seja seguro para qualquer alvo de dispatch. Veja [Clustering](CLUSTERING.md) |
