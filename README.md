## Contents
- [cai_src](./cai_src) - Dockerized CAI container with Kali Linux and SSH access.
- [red_team_agent.ipynb](./red_team_agent.ipynb) - Jupyter notebook demonstrating the use of the Metasploit MCP server with a red team agent. Uses Langchain & Langsmith.

## Firing up Metasploit MCP
```bash
export MSF_PASSWORD=MyPassword
export MSF_SERVER=127.0.0.1
export MSF_PORT=55553
export MSF_SSL=false
```
`msfconsole`
`msfrpcd -P MyPassword -S -a 127.0.0.1 -p 55553`
`# cd to MetasploitMCP folder`
`python3 -m venv .vnv`
`source .vnv/bin/activate`
`pip install -r requirements.txt`
`python3 MetasploitMCP.py --transport http --host 0.0.0.0 --port 8085`

## Stop the Metasploit RPC
`ps -aux | grep msfrpcd`
`kill -9 [PID]`

## Models

- Claude for tool calling.
  - Bedrock?
- Running local models: llamacpp
  - Codename Goose
- LangChain

Being able to explain in plain english "find vulnerable devices" makes it easier for non-technical users to perform security assessments,
and makes it easier for technical users to perform actions, eg "scan <ip> for open SSH ports".

## Default vs. User-Defined Tools
- eg. a MCP service, or a Google search
- Most models / tooling exposing the command shell as a default tool.

### Potential Topics:

- How to tell if the attacker is an AI?
- How to tell if the defense is an AI?
- LLM parsing logs
- what software can be used to make the parsing better?

- Agentic notifications
- ntfy example

- How to capture / replay?
- Shell commands? PCap?
- Reverse engineering / deobfuscation for param names?
- Blue team: detect something's wrong (potentially fix it)
- Phishing email with agentic llm
- Kiro agent?

## Memories from first day
- Tool inserting text into HTTP responses to try to prevent reverse prompt injection (CAI)
- CAI was using curl as bash command (not a custom tool or CAI) to submit form and found SQL injection. Didn't fully comprehend the relevant form parameters. Did not take advantage of the vuln. GPT o3-mini would not continue once prompted in the context to continue to exploit
- O3-Mini guardrails rejected "Using common web exploits (sql injection, php vulns, etc) get root access to http://192.168.1.106/mutillidae/"
- openai:gpt-5-mini just will do hacking and even gives you a lecture / finger-wag
- openai:gpt-4.1 is cooperative
- "as quickly as you can" trimmed time from 5 min to 25 seconds
- "create a file at the root directory called 'dan_wuz_here_1.txt.'" forced it to take advantage of the session to actually run commands on the server. before, it opened a session but didn't prove it could run as root
- "do not ask me for any further inputs, operate autonomously" + create_react_agent instead of create_agent made it possible for it to run completely on its own
- blue team does similar actions to red team but did not actually run the exploits, asked for further investigation (human in the loop)
- switching from Mint to Kali docker image allowed us to give the AI way more tools
- Metasploitable on a separate headless box with SSH available
- Uber Eats can deliver you long Ethernet cables
- Firehouse Subs and Gin & Tonic
- CAI agent (agentic persona) defines the system prompt and which tools are available 
- Setting up observability for Phoenix or LangSmith, both have awesome visuals, LangSmith is the WORST to set up

## To explore soon:
- Phoenix Model Logging Tools
- Dig into Selenium and PortSwigger API
- Write up the fictional business and personas, motivations for an attack, story-telling component. Alien company?
- Add our Brunches notes