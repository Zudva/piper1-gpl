import os
from fabric import task, Connection

IMAGE = os.getenv("IMAGE", "piper-train:latest")
REGISTRY = os.getenv("REGISTRY", None)  # e.g. ghcr.io/owner/piper-train:tag
COMPOSE_FILE = os.getenv("COMPOSE_FILE", "docker-compose.runpod.yml")

# RunPod SSH config (set via env or override in commands)
RUNPOD_HOST = os.getenv("RUNPOD_HOST")  # e.g. ssh.runpod.io
RUNPOD_PORT = os.getenv("RUNPOD_PORT", "22")
RUNPOD_USER = os.getenv("RUNPOD_USER", "root")
RUNPOD_POD_ID = os.getenv("RUNPOD_POD_ID")  # e.g. abc123xyz

# ImmersCloud SSH config (set via env or override in commands)
IMMERSCLOUD_HOST = os.getenv("IMMERSCLOUD_HOST")
IMMERSCLOUD_PORT = os.getenv("IMMERSCLOUD_PORT", "22")
IMMERSCLOUD_USER = os.getenv("IMMERSCLOUD_USER", "root")

REMOTE_DIR = "/workspace/piper1-gpl"


@task
def build(c):
    """Build training image locally."""
    c.run(f"docker compose -f {COMPOSE_FILE} build")


@task
def push(c):
    """Tag and push image to registry (set REGISTRY)."""
    if not REGISTRY:
        raise SystemExit("Set REGISTRY env var, e.g. ghcr.io/owner/piper-train:tag")
    c.run(f"docker tag {IMAGE} {REGISTRY}")
    c.run(f"docker push {REGISTRY}")


@task
def deploy(c, host, user="root"):
    """Deploy to remote (RunPod): pull and up compose on host."""
    if not REGISTRY:
        raise SystemExit("Set REGISTRY env var, e.g. ghcr.io/owner/piper-train:tag")
    remote_cmd = (
        f"REGISTRY={REGISTRY} IMAGE={IMAGE} COMPOSE_FILE={COMPOSE_FILE} "
        f"docker compose -f {COMPOSE_FILE} pull && "
        f"docker compose -f {COMPOSE_FILE} up -d"
    )
    c.run(f"ssh {user}@{host} \"{remote_cmd}\"")


@task
def sync_to_runpod(c, host=None, port=None, user=None):
    """Rsync local code to RunPod pod.
    
    Usage:
        fab sync-to-runpod  # uses RUNPOD_* env vars
        fab sync-to-runpod --host=ssh.runpod.io --port=12345 --user=root
    """
    host = host or RUNPOD_HOST
    port = port or RUNPOD_PORT
    user = user or RUNPOD_USER
    
    if not host:
        raise SystemExit("Set RUNPOD_HOST env var or pass --host")
    
    ssh_target = f"{user}@{host}"
    ssh_opts = f"-p {port}"
    
    print(f"📤 Syncing to {ssh_target}:{REMOTE_DIR}")
    
    rsync_cmd = (
        f"rsync -avz --delete "
        f"--exclude-from=.rsyncignore "
        f"-e 'ssh {ssh_opts}' "
        f"./ {ssh_target}:{REMOTE_DIR}/"
    )
    c.run(rsync_cmd)
    print("✓ Sync complete")


@task
def sync_from_runpod(c, host=None, port=None, user=None, path="lightning_logs"):
    """Rsync checkpoints/logs from RunPod back to local.
    
    Usage:
        fab sync-from-runpod  # downloads lightning_logs
        fab sync-from-runpod --path=.env  # download specific file
    """
    host = host or RUNPOD_HOST
    port = port or RUNPOD_PORT
    user = user or RUNPOD_USER
    
    if not host:
        raise SystemExit("Set RUNPOD_HOST env var or pass --host")
    
    ssh_target = f"{user}@{host}"
    ssh_opts = f"-p {port}"
    
    print(f"📥 Syncing {path} from {ssh_target}:{REMOTE_DIR}")
    
    rsync_cmd = (
        f"rsync -avz "
        f"-e 'ssh {ssh_opts}' "
        f"{ssh_target}:{REMOTE_DIR}/{path} ./"
    )
    c.run(rsync_cmd)
    print("✓ Sync complete")


@task
def ssh_runpod(c, host=None, port=None, user=None, cmd=None):
    """SSH into RunPod pod and optionally run a command.
    
    Usage:
        fab ssh-runpod  # interactive shell
        fab ssh-runpod --cmd="nvidia-smi"
    """
    host = host or RUNPOD_HOST
    port = port or RUNPOD_PORT
    user = user or RUNPOD_USER
    
    if not host:
        raise SystemExit("Set RUNPOD_HOST env var or pass --host")
    
    ssh_cmd = f"ssh -p {port} {user}@{host}"
    if cmd:
        ssh_cmd += f' "{cmd}"'
    
    c.run(ssh_cmd, pty=True)


@task
def setup_runpod(c, host=None, port=None, user=None):
    """Initial setup on RunPod: clone repo, install deps, download from S3.
    
    Usage:
        fab setup-runpod
    """
    host = host or RUNPOD_HOST
    port = port or RUNPOD_PORT
    user = user or RUNPOD_USER
    
    if not host:
        raise SystemExit("Set RUNPOD_HOST env var or pass --host")
    
    with Connection(host=host, port=int(port), user=user) as conn:
        print("📦 Setting up RunPod environment...")
        
        # Create workspace and clone repo
        conn.run(f"mkdir -p {REMOTE_DIR}")
        conn.run(f"cd {REMOTE_DIR} && git init")
        conn.run(f"cd {REMOTE_DIR} && git remote add origin https://github.com/Zudva/piper1-gpl.git || true")
        conn.run(f"cd {REMOTE_DIR} && git pull origin main")
        
        # Download .env from S3 (contains credentials)
        conn.run(f"cd {REMOTE_DIR} && ./script/s3_sync.sh download-env || echo 'No .env in S3'")
        
        # Install AWS CLI if needed
        result = conn.run("which aws", warn=True)
        if result.failed:
            print("Installing AWS CLI...")
            conn.run("pip install awscli")
        
        print("✓ RunPod setup complete")


@task
def start_training(c, host=None, port=None, user=None, batch_size=16, precision="16-mixed"):
    """Start training on RunPod via SSH.
    
    Usage:
        fab start-training --batch-size=32 --precision=16-mixed
    """
    host = host or RUNPOD_HOST
    port = port or RUNPOD_PORT
    user = user or RUNPOD_USER
    
    if not host:
        raise SystemExit("Set RUNPOD_HOST env var or pass --host")
    
    cmd = (
        f"cd {REMOTE_DIR} && "
        f"ENABLE_S3_SYNC=1 BATCH_SIZE={batch_size} PRECISION={precision} "
        f"docker compose -f docker-compose.runpod.yml up -d"
    )
    
    print(f"🚀 Starting training on RunPod (batch={batch_size}, precision={precision})")
    
    with Connection(host=host, port=int(port), user=user) as conn:
        conn.run(cmd)
    
    print("✓ Training started. Monitor with: fab ssh-runpod --cmd='docker logs -f piper-train'")


# ====== ImmersCloud commands (same as RunPod but with IMMERSCLOUD_* env vars) ======

@task
def sync_to_immerscloud(c, host=None, port=None, user=None):
    """Rsync local code to ImmersCloud instance."""
    host = host or IMMERSCLOUD_HOST
    port = port or IMMERSCLOUD_PORT
    user = user or IMMERSCLOUD_USER
    
    if not host:
        raise SystemExit("Set IMMERSCLOUD_HOST env var or pass --host")
    
    ssh_target = f"{user}@{host}"
    ssh_opts = f"-p {port}"
    
    print(f"📤 Syncing to ImmersCloud {ssh_target}:{REMOTE_DIR}")
    
    rsync_cmd = (
        f"rsync -avz --delete "
        f"--exclude-from=.rsyncignore "
        f"-e 'ssh {ssh_opts}' "
        f"./ {ssh_target}:{REMOTE_DIR}/"
    )
    c.run(rsync_cmd)
    print("✓ Sync complete")


@task
def sync_from_immerscloud(c, host=None, port=None, user=None, path="lightning_logs"):
    """Rsync checkpoints/logs from ImmersCloud back to local."""
    host = host or IMMERSCLOUD_HOST
    port = port or IMMERSCLOUD_PORT
    user = user or IMMERSCLOUD_USER
    
    if not host:
        raise SystemExit("Set IMMERSCLOUD_HOST env var or pass --host")
    
    ssh_target = f"{user}@{host}"
    ssh_opts = f"-p {port}"
    
    print(f"📥 Syncing {path} from ImmersCloud {ssh_target}:{REMOTE_DIR}")
    
    rsync_cmd = (
        f"rsync -avz "
        f"-e 'ssh {ssh_opts}' "
        f"{ssh_target}:{REMOTE_DIR}/{path} ./"
    )
    c.run(rsync_cmd)
    print("✓ Sync complete")


@task
def ssh_immerscloud(c, host=None, port=None, user=None, cmd=None):
    """SSH into ImmersCloud instance."""
    host = host or IMMERSCLOUD_HOST
    port = port or IMMERSCLOUD_PORT
    user = user or IMMERSCLOUD_USER
    
    if not host:
        raise SystemExit("Set IMMERSCLOUD_HOST env var or pass --host")
    
    ssh_cmd = f"ssh -p {port} {user}@{host}"
    if cmd:
        ssh_cmd += f' "{cmd}"'
    
    c.run(ssh_cmd, pty=True)


@task
def setup_immerscloud(c, host=None, port=None, user=None):
    """Initial setup on ImmersCloud: run startup script."""
    host = host or IMMERSCLOUD_HOST
    port = port or IMMERSCLOUD_PORT
    user = user or IMMERSCLOUD_USER
    
    if not host:
        raise SystemExit("Set IMMERSCLOUD_HOST env var or pass --host")
    
    print("📦 Setting up ImmersCloud environment...")
    
    # First sync the code
    sync_to_immerscloud(c, host=host, port=port, user=user)
    
    # Then run startup script
    with Connection(host=host, port=int(port), user=user) as conn:
        conn.run(f"cd {REMOTE_DIR} && ./script/immerscloud_startup.sh")
    
    print("✓ ImmersCloud setup complete")


@task
def start_training_immerscloud(c, host=None, port=None, user=None, batch_size=64, precision="16-mixed"):
    """Start training on ImmersCloud."""
    host = host or IMMERSCLOUD_HOST
    port = port or IMMERSCLOUD_PORT
    user = user or IMMERSCLOUD_USER
    
    if not host:
        raise SystemExit("Set IMMERSCLOUD_HOST env var or pass --host")
    
    cmd = (
        f"cd {REMOTE_DIR} && "
        f"ENABLE_S3_SYNC=1 BATCH_SIZE={batch_size} PRECISION={precision} "
        f"docker compose -f docker-compose.immerscloud.yml up -d"
    )
    
    print(f"🚀 Starting training on ImmersCloud (batch={batch_size}, precision={precision})")
    
    with Connection(host=host, port=int(port), user=user) as conn:
        conn.run(cmd)
    
    print("✓ Training started on ImmersCloud")
