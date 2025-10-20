# AI Wargames - UNCW Cybersecurity Conference 2025

Thanks for attending our AI Wargames talk at the UNCW Cybersecurity Conference!

This repository contains the presentation slides in PowerPoint or PDF format along with source code examples from all of our demos.

These examples are provided for academic & educational purposes only and should not be used with systems you don't own or are unauthorized to test.

For best results, use a safely isolated, easy to restore test environment using Docker and/or virtual machines.

## Contents
- [Presentation Slides](https://docs.google.com/presentation/d/1bFOIhxXMUZ6wjz8ahWWiBUeX7MMt7iPcWM-7ymi86sQ/edit?usp=sharing)
- [Slides in PDF format](slides.pdf)
- [cai_src](./cai_src) - Dockerized CAI container with Kali Linux and SSH access.
- [red_team_agent.ipynb](./red_team_agent.ipynb) - Jupyter notebook demonstrating the use of the Metasploit MCP server with a red team agent. Uses Langchain & Langsmith.

## Firing up Metasploit MCP

Follow instructions at [MetasploitMCP on GitHub](https://github.com/GH05TCREW/MetasploitMCP)

```bash
export MSF_PASSWORD=MyPassword
export MSF_SERVER=127.0.0.1
export MSF_PORT=55553
export MSF_SSL=false

msfconsole

msfrpcd -P MyPassword -S -a 127.0.0.1 -p 55553

# cd to MetasploitMCP folder

python3 -m venv .vnv

source .vnv/bin/activate

pip install -r requirements.txt

python3 MetasploitMCP.py --transport http --host 0.0.0.0 --port 8085
```

## Stopping the Metasploit RPC if you need to restart it
```bash
ps -aux | grep msfrpcd
kill -9 [PID]
```
