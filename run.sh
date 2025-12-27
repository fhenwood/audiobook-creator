#!/bin/bash
# ============================================================================
# Audiobook Creator - Startup Script
# ============================================================================
# This script automatically detects your hardware and starts the appropriate
# Docker Compose configuration.
#
# Usage:
#   ./run.sh          # Start the application
#   ./run.sh stop     # Stop the application
#   ./run.sh logs     # View logs
#   ./run.sh restart  # Restart the application
# ============================================================================

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Script directory
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

# ============================================================================
# Hardware Detection Functions
# ============================================================================

detect_nvidia_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        if nvidia-smi &> /dev/null; then
            # Get GPU info
            GPU_NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null | head -n1)
            GPU_MEMORY=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits 2>/dev/null | head -n1)
            echo "$GPU_NAME:$GPU_MEMORY"
            return 0
        fi
    fi
    return 1
}

check_nvidia_container_toolkit() {
    if docker info 2>/dev/null | grep -q "nvidia"; then
        return 0
    fi
    # Alternative check
    if command -v nvidia-container-cli &> /dev/null; then
        return 0
    fi
    return 1
}

get_system_ram() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        free -g | awk '/^Mem:/{print $2}'
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        sysctl -n hw.memsize | awk '{print int($1/1024/1024/1024)}'
    else
        echo "16"  # Default assumption
    fi
}

get_cpu_cores() {
    if [[ "$OSTYPE" == "linux-gnu"* ]]; then
        nproc
    elif [[ "$OSTYPE" == "darwin"* ]]; then
        sysctl -n hw.ncpu
    else
        echo "8"  # Default assumption
    fi
}

# ============================================================================
# Environment Setup
# ============================================================================

setup_env() {
    if [[ ! -f ".env" ]]; then
        echo -e "${YELLOW}📝 No .env file found. Creating from template...${NC}"
        
        if [[ -f ".env_sample" ]]; then
            cp .env_sample .env
        else
            echo -e "${RED}❌ .env_sample not found!${NC}"
            exit 1
        fi
    fi
    
    # Detect hardware and suggest settings
    local GPU_INFO=$(detect_nvidia_gpu)
    local RAM=$(get_system_ram)
    local CORES=$(get_cpu_cores)
    
    echo -e "${BLUE}🖥️  System Detection:${NC}"
    echo "   CPU Cores: $CORES"
    echo "   RAM: ${RAM}GB"
    
    if [[ -n "$GPU_INFO" ]]; then
        local GPU_NAME=$(echo "$GPU_INFO" | cut -d: -f1)
        local GPU_MEM=$(echo "$GPU_INFO" | cut -d: -f2)
        echo -e "   GPU: ${GREEN}$GPU_NAME${NC} (${GPU_MEM}MB VRAM)"
        
        # Suggest GPU layers based on VRAM
        if [[ "$GPU_MEM" -ge 20000 ]]; then
            SUGGESTED_LAYERS=99
            SUGGESTED_TTS_PARALLEL=8
            SUGGESTED_LLM_PARALLEL=4
        elif [[ "$GPU_MEM" -ge 12000 ]]; then
            SUGGESTED_LAYERS=99
            SUGGESTED_TTS_PARALLEL=4
            SUGGESTED_LLM_PARALLEL=2
        elif [[ "$GPU_MEM" -ge 8000 ]]; then
            SUGGESTED_LAYERS=60
            SUGGESTED_TTS_PARALLEL=2
            SUGGESTED_LLM_PARALLEL=1
        else
            SUGGESTED_LAYERS=30
            SUGGESTED_TTS_PARALLEL=1
            SUGGESTED_LLM_PARALLEL=1
        fi
        
        # Update .env with detected settings if they're at defaults
        if grep -q "GPU_LAYERS=95" .env 2>/dev/null; then
            sed -i.bak "s/GPU_LAYERS=95/GPU_LAYERS=$SUGGESTED_LAYERS/" .env 2>/dev/null || \
            sed -i '' "s/GPU_LAYERS=95/GPU_LAYERS=$SUGGESTED_LAYERS/" .env
        fi
        
        HAS_GPU=true
    else
        echo -e "   GPU: ${YELLOW}None detected (CPU mode)${NC}"
        HAS_GPU=false
        
        # Set CPU mode in .env
        if grep -q "GPU_LAYERS=" .env 2>/dev/null; then
            sed -i.bak "s/GPU_LAYERS=.*/GPU_LAYERS=0/" .env 2>/dev/null || \
            sed -i '' "s/GPU_LAYERS=.*/GPU_LAYERS=0/" .env
        fi
    fi
    
    # Check NVIDIA container toolkit
    if [[ "$HAS_GPU" == true ]]; then
        if check_nvidia_container_toolkit; then
            echo -e "   NVIDIA Container Toolkit: ${GREEN}Installed${NC}"
            CAN_USE_GPU=true
        else
            echo -e "   NVIDIA Container Toolkit: ${RED}Not installed${NC}"
            echo -e "${YELLOW}   ⚠️  GPU detected but container toolkit not found.${NC}"
            echo -e "${YELLOW}   Run the installation commands from README.md to enable GPU support.${NC}"
            CAN_USE_GPU=false
        fi
    fi
    
    echo ""
}

# ============================================================================
# Docker Compose Commands
# ============================================================================

get_compose_files() {
    local GPU_INFO=$(detect_nvidia_gpu)
    
    if [[ -n "$GPU_INFO" ]] && check_nvidia_container_toolkit; then
        echo "-f docker-compose.yaml -f docker-compose.gpu.yaml"
    else
        echo "-f docker-compose.yaml -f docker-compose.cpu.yaml"
    fi
}

start_services() {
    setup_env
    
    local COMPOSE_FILES=$(get_compose_files)
    
    echo -e "${GREEN}🚀 Starting Audiobook Creator...${NC}"
    echo -e "   Using: docker compose $COMPOSE_FILES"
    echo ""
    
    docker compose $COMPOSE_FILES up -d
    
    echo ""
    echo -e "${GREEN}✅ Services started!${NC}"
    echo ""
    echo -e "🌐 Web UI: ${BLUE}http://localhost:7860${NC}"
    echo -e "🔊 TTS API: ${BLUE}http://localhost:8880${NC}"
    echo ""
    echo -e "${YELLOW}📥 First run? Models are downloading in the background.${NC}"
    echo -e "   Watch progress: ${BLUE}./run.sh logs${NC}"
}

stop_services() {
    local COMPOSE_FILES=$(get_compose_files)
    
    echo -e "${YELLOW}🛑 Stopping Audiobook Creator...${NC}"
    docker compose $COMPOSE_FILES down
    echo -e "${GREEN}✅ Services stopped.${NC}"
}

restart_services() {
    stop_services
    echo ""
    start_services
}

show_logs() {
    local COMPOSE_FILES=$(get_compose_files)
    local SERVICE="${1:-}"
    
    if [[ -n "$SERVICE" ]]; then
        docker compose $COMPOSE_FILES logs -f "$SERVICE"
    else
        docker compose $COMPOSE_FILES logs -f
    fi
}

show_status() {
    local COMPOSE_FILES=$(get_compose_files)
    
    echo -e "${BLUE}📊 Service Status:${NC}"
    docker compose $COMPOSE_FILES ps
}

rebuild_services() {
    local COMPOSE_FILES=$(get_compose_files)
    
    echo -e "${YELLOW}🔨 Rebuilding containers...${NC}"
    docker compose $COMPOSE_FILES build --no-cache
    echo -e "${GREEN}✅ Rebuild complete.${NC}"
    echo ""
    echo -e "Run ${BLUE}./run.sh${NC} to start the updated services."
}

# ============================================================================
# Main Entry Point
# ============================================================================

show_help() {
    echo "Audiobook Creator - Startup Script"
    echo ""
    echo "Usage: ./run.sh [command]"
    echo ""
    echo "Commands:"
    echo "  (none)     Start the application (default)"
    echo "  stop       Stop all services"
    echo "  restart    Restart all services"
    echo "  logs       View logs (Ctrl+C to exit)"
    echo "  logs <svc> View logs for specific service"
    echo "  status     Show service status"
    echo "  rebuild    Rebuild Docker images"
    echo "  help       Show this help message"
    echo ""
    echo "Services: audiobook_creator, llm_server, orpheus_llama"
}

case "${1:-start}" in
    start)
        start_services
        ;;
    stop)
        stop_services
        ;;
    restart)
        restart_services
        ;;
    logs)
        show_logs "${2:-}"
        ;;
    status)
        show_status
        ;;
    rebuild)
        rebuild_services
        ;;
    help|--help|-h)
        show_help
        ;;
    *)
        echo -e "${RED}Unknown command: $1${NC}"
        echo ""
        show_help
        exit 1
        ;;
esac
