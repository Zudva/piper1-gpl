#!/usr/bin/env python3
"""
RunPod API integration for automated pod creation and management.

Usage:
    # Set API key
    export RUNPOD_API_KEY="your_api_key_here"
    
    # Create L40S pod
    python script/runpod_api.py create --gpu L40S --name piper-training
    
    # List pods
    python script/runpod_api.py list
    
    # Get SSH command
    python script/runpod_api.py ssh POD_ID
    
    # Stop pod
    python script/runpod_api.py stop POD_ID
    
    # Terminate pod
    python script/runpod_api.py terminate POD_ID
"""

import os
import sys
import json
import requests
import argparse
from typing import Optional, Dict, Any


RUNPOD_API_URL = "https://api.runpod.io/graphql"
RUNPOD_API_KEY = os.getenv("RUNPOD_API_KEY")

# GPU templates (можно получить через API или из документации)
GPU_TYPES = {
    "L4": "NVIDIA L4",
    "L40S": "NVIDIA L40S", 
    "RTX4090": "NVIDIA GeForce RTX 4090",
    "RTX5090": "NVIDIA GeForce RTX 5090",
    "A100": "NVIDIA A100 80GB PCIe",
    "H100": "NVIDIA H100 PCIe",
}


def graphql_request(query: str, variables: Optional[Dict[str, Any]] = None) -> Dict:
    """Execute GraphQL request to RunPod API."""
    if not RUNPOD_API_KEY:
        raise ValueError("RUNPOD_API_KEY environment variable not set")
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {RUNPOD_API_KEY}",
    }
    
    payload = {"query": query}
    if variables:
        payload["variables"] = variables
    
    response = requests.post(RUNPOD_API_URL, json=payload, headers=headers)
    response.raise_for_status()
    
    data = response.json()
    if "errors" in data:
        raise Exception(f"GraphQL errors: {data['errors']}")
    
    return data["data"]


def create_pod(
    gpu_type: str = "L40S",
    gpu_count: int = 1,
    name: str = "piper-training",
    container_disk: int = 150,
    volume_disk: int = 100,
    docker_image: str = "ghcr.io/zudva/piper-train:latest",
    env_vars: Optional[Dict[str, str]] = None,
    spot: bool = True,
) -> Dict:
    """Create a new RunPod pod."""
    
    # Prepare environment variables
    env_list = []
    if env_vars:
        for key, value in env_vars.items():
            env_list.append({"key": key, "value": value})
    
    query = """
    mutation CreatePod($input: PodInput!) {
      podCreate(input: $input) {
        id
        name
        runtime {
          uptimeInSeconds
          ports {
            ip
            isIpPublic
            privatePort
            publicPort
            type
          }
        }
        machine {
          gpuDisplayName
        }
      }
    }
    """
    
    variables = {
        "input": {
            "cloudType": "COMMUNITY" if spot else "SECURE",
            "gpuCount": gpu_count,
            "gpuTypeId": gpu_type,  # Нужно получить реальный ID
            "name": name,
            "containerDiskInGb": container_disk,
            "volumeInGb": volume_disk,
            "dockerArgs": f"-v /workspace -p 22:22 -p 6006:6006",
            "imageName": docker_image,
            "env": env_list,
            "ports": "22/tcp,6006/http",
        }
    }
    
    data = graphql_request(query, variables)
    return data["podCreate"]


def list_pods() -> list:
    """List all pods."""
    query = """
    query GetPods {
      myself {
        pods {
          id
          name
          runtime {
            uptimeInSeconds
            ports {
              ip
              isIpPublic
              privatePort
              publicPort
              type
            }
          }
          machine {
            gpuDisplayName
          }
        }
      }
    }
    """
    
    data = graphql_request(query)
    return data["myself"]["pods"]


def get_pod(pod_id: str) -> Dict:
    """Get pod details."""
    query = """
    query GetPod($podId: String!) {
      pod(input: {podId: $podId}) {
        id
        name
        runtime {
          uptimeInSeconds
          ports {
            ip
            isIpPublic
            privatePort
            publicPort
            type
          }
        }
        machine {
          gpuDisplayName
          cpuCores
          memoryInGb
          diskInGb
        }
      }
    }
    """
    
    data = graphql_request(query, {"podId": pod_id})
    return data["pod"]


def stop_pod(pod_id: str) -> Dict:
    """Stop a running pod."""
    query = """
    mutation StopPod($podId: String!) {
      podStop(input: {podId: $podId}) {
        id
        name
      }
    }
    """
    
    data = graphql_request(query, {"podId": pod_id})
    return data["podStop"]


def terminate_pod(pod_id: str) -> Dict:
    """Terminate (delete) a pod."""
    query = """
    mutation TerminatePod($podId: String!) {
      podTerminate(input: {podId: $podId})
    }
    """
    
    data = graphql_request(query, {"podId": pod_id})
    return data


def get_ssh_command(pod_id: str) -> str:
    """Get SSH command for connecting to pod."""
    pod = get_pod(pod_id)
    
    # Find SSH port
    ssh_port = None
    ssh_ip = None
    for port in pod["runtime"]["ports"]:
        if port["privatePort"] == 22:
            ssh_port = port["publicPort"]
            ssh_ip = port["ip"]
            break
    
    if not ssh_port or not ssh_ip:
        raise Exception("SSH port not found for pod")
    
    return f"ssh root@{ssh_ip} -p {ssh_port}"


def main():
    parser = argparse.ArgumentParser(description="RunPod API CLI")
    subparsers = parser.add_subparsers(dest="command", help="Command to run")
    
    # Create pod
    create_parser = subparsers.add_parser("create", help="Create a new pod")
    create_parser.add_argument("--gpu", default="L40S", choices=list(GPU_TYPES.keys()),
                              help="GPU type")
    create_parser.add_argument("--count", type=int, default=1, help="Number of GPUs")
    create_parser.add_argument("--name", default="piper-training", help="Pod name")
    create_parser.add_argument("--container-disk", type=int, default=150,
                              help="Container disk in GB")
    create_parser.add_argument("--volume-disk", type=int, default=100,
                              help="Volume disk in GB")
    create_parser.add_argument("--image", default="ghcr.io/zudva/piper-train:latest",
                              help="Docker image")
    create_parser.add_argument("--spot", action="store_true", default=True,
                              help="Use spot instance (cheaper)")
    create_parser.add_argument("--on-demand", action="store_true",
                              help="Use on-demand instance (more stable)")
    
    # List pods
    subparsers.add_parser("list", help="List all pods")
    
    # Get pod details
    get_parser = subparsers.add_parser("get", help="Get pod details")
    get_parser.add_argument("pod_id", help="Pod ID")
    
    # Get SSH command
    ssh_parser = subparsers.add_parser("ssh", help="Get SSH command")
    ssh_parser.add_argument("pod_id", help="Pod ID")
    
    # Stop pod
    stop_parser = subparsers.add_parser("stop", help="Stop a pod")
    stop_parser.add_argument("pod_id", help="Pod ID")
    
    # Terminate pod
    terminate_parser = subparsers.add_parser("terminate", help="Terminate a pod")
    terminate_parser.add_argument("pod_id", help="Pod ID")
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == "create":
            # Load env vars from .env if exists
            env_vars = {}
            if os.path.exists(".env"):
                with open(".env") as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith("#") and "=" in line:
                            key, value = line.split("=", 1)
                            env_vars[key.strip()] = value.strip()
            
            spot = not args.on_demand
            pod = create_pod(
                gpu_type=args.gpu,
                gpu_count=args.count,
                name=args.name,
                container_disk=args.container_disk,
                volume_disk=args.volume_disk,
                docker_image=args.image,
                env_vars=env_vars,
                spot=spot,
            )
            print(f"✅ Pod created: {pod['id']}")
            print(f"Name: {pod['name']}")
            print(f"GPU: {pod['machine']['gpuDisplayName']}")
            print(f"\nGet SSH command: python {sys.argv[0]} ssh {pod['id']}")
        
        elif args.command == "list":
            pods = list_pods()
            if not pods:
                print("No pods found")
                return
            
            print(f"{'ID':<30} {'Name':<20} {'GPU':<30} {'Uptime':<10}")
            print("-" * 100)
            for pod in pods:
                uptime = pod["runtime"]["uptimeInSeconds"] if pod["runtime"] else 0
                uptime_str = f"{uptime // 3600}h {(uptime % 3600) // 60}m"
                gpu = pod["machine"]["gpuDisplayName"] if pod["machine"] else "N/A"
                print(f"{pod['id']:<30} {pod['name']:<20} {gpu:<30} {uptime_str:<10}")
        
        elif args.command == "get":
            pod = get_pod(args.pod_id)
            print(json.dumps(pod, indent=2))
        
        elif args.command == "ssh":
            ssh_cmd = get_ssh_command(args.pod_id)
            print(f"\nSSH command:")
            print(ssh_cmd)
            print(f"\nOr set env vars:")
            pod = get_pod(args.pod_id)
            for port in pod["runtime"]["ports"]:
                if port["privatePort"] == 22:
                    print(f"export RUNPOD_HOST={port['ip']}")
                    print(f"export RUNPOD_PORT={port['publicPort']}")
                    break
        
        elif args.command == "stop":
            pod = stop_pod(args.pod_id)
            print(f"✅ Pod stopped: {pod['id']}")
        
        elif args.command == "terminate":
            terminate_pod(args.pod_id)
            print(f"✅ Pod terminated: {args.pod_id}")
    
    except Exception as e:
        print(f"❌ Error: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
