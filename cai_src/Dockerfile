# Modified from @tcrisall's dockerfile: https://github.com/tcrisall/cai/blob/a479194f936a927d4894ddec1ed7b1abf186c1f6/Dockerfile#L1
FROM kalilinux/kali-rolling as base-stage

# Install dependencies
RUN apt update && apt install -y \
    kali-linux-headless \
    iputils-ping \
    git \
    python3-pip \
    python3-venv \
    openssh-server \
    vim \
    && apt-get clean \
    && rm -rf /var/lib/apt/lists/*

# Bring in the entire CAI project
WORKDIR /app
COPY . .

# Create the virtual environment
RUN python3 -m venv cai_env

ENV PATH="/app/cai_env/bin:$PATH"

# Install the package from the local directory
#RUN source cai_env/bin/activate && pip install cai-framework
RUN pip install cai-framework

RUN echo 'export PATH=/app/cai_env/bin:$PATH\n\
cai\n\
' >> /etc/profile.d/99-setup-cai.sh

COPY .env .env

# Setup SSH
RUN mkdir -p /var/run/sshd
RUN echo 'root:toor' | chpasswd
# Allow root login via SSH (optional, for convenience; consider security implications)
RUN sed -i 's/#PermitRootLogin prohibit-password/PermitRootLogin yes/' /etc/ssh/sshd_config

# Disable PAM to avoid issues with some container environments
RUN sed -i 's/usepam yes/usepam no/' /etc/ssh/sshd_config

# Expose SSH port
EXPOSE 22

# Start SSH server
CMD ["/usr/sbin/sshd", "-D"]