# Dockerized CAI Container

Uses the Python-based [CAI](https://github.com/aliasrobotics/cai) (Cybersecurity AI).
Dockerfile sets up the environment with:
- Kali
- Python3.12
- SSHD
- CAI-Framework


Exposes an SSH service on port `22`.

Copies over a `.env` file.


## Setup

1. Follow the [CAI Setup .env File](https://github.com/aliasrobotics/cai?tab=readme-ov-file#nut_and_bolt-setup-env-file) step to create a `.env` file with this structure:
   ```env
   OPENAI_API_KEY="sk-1234"
   OLLAMA=""
   ALIAS_API_KEY="<sk-your-key>"
   CAI_STEAM=False
   ```
   or see the `.env.example` file.
2. Build the Docker image:
   ```bash
   docker build -t kali-cai .
   ```
3. Run the Docker container with the required capabilities:
   ```bash
   docker run --name kali-cai --cap-add=NET_RAW --cap-add=NET_ADMIN -d kali-cai
   ```
4. SSH into the container:
   ```bash
   docker inspect --format='{{range .NetworkSettings.Networks}}{{.IPAddress}}{{end}}' kali-cai
   ssh root@<ip>
   ```
5. CAI may already be running, but if not, start it with `cai`.
6. Select a model `/model o3-mini`.
7. Select an agent `/agent redteam_agent`. 
   1. The agent is what gives the model tool access (and a system prompt). Running CAI without an agent / pattern specified will simply call the LLM and not perform actions.


Example Prompts:
```
Find open ports on example.com
> Performs port scan using nmap
```

You may copy the logs from CAI to your working directory with:

`docker cp kali_cai:/root/logs ./logs`