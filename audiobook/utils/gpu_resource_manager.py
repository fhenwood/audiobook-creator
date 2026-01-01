"""
Audiobook Creator - GPU Resource Manager

This program is free software: you can redistribute it and/or modify
it under the terms of the GNU General Public License as published by
the Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

This program is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with this program.  If not, see <https://www.gnu.org/licenses/>.

Manages GPU resources by starting/stopping LLM containers on-demand.
This keeps GPU memory free when not actively using specific services.

Strategy:
- LLM server (gpt-oss ~12GB): Only loaded during emotion tag processing
- Orpheus LLM (~5GB): Only loaded during TTS generation
- Each service is started just before use and stopped immediately after
- This maximizes available VRAM for each task
"""

import os
import time
import threading
import requests
from typing import Optional, Tuple
from contextlib import contextmanager


# Container configuration from environment
from audiobook.config import settings

# Configuration from Pydantic settings
LLM_SERVER_HOST = settings.llm_server_host
LLM_SERVER_PORT = settings.llm_server_port
ORPHEUS_LLAMA_HOST = settings.orpheus_llama_host
ORPHEUS_LLAMA_PORT = settings.orpheus_llama_port

# Container names are derived from hostnames in docker-compose
LLM_CONTAINER_NAME = LLM_SERVER_HOST
ORPHEUS_CONTAINER_NAME = ORPHEUS_LLAMA_HOST

# Docker API endpoint (when docker socket is mounted)
DOCKER_SOCKET = "/var/run/docker.sock"

# Timeout settings
CONTAINER_START_TIMEOUT = 120  # seconds to wait for container to be healthy
HEALTH_CHECK_INTERVAL = 2  # seconds between health checks
IDLE_TIMEOUT = settings.gpu_idle_timeout  # seconds before stopping idle containers


class GPUResourceManager:
    """
    Singleton manager for GPU resources.
    Starts LLM containers on-demand and stops them immediately when done.
    
    Key optimization: Only one heavy LLM model in VRAM at a time.
    - Emotion tagging: Load LLM server (~12GB), process, then unload
    - TTS generation: Load Orpheus LLM (~5GB), generate, then unload
    """
    
    _instance = None
    _lock = threading.Lock()
    
    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance
    
    def __init__(self):
        if self._initialized:
            return
        self._initialized = True
        self._llm_users = 0  # Count of active LLM users
        self._orpheus_users = 0  # Count of active Orpheus users
        self._llm_idle_timer: Optional[threading.Timer] = None
        self._orpheus_idle_timer: Optional[threading.Timer] = None
        self._docker_available = self._check_docker_available()
        
        if not self._docker_available:
            print("⚠️ Docker socket not available - GPU resource management disabled")
            print("   Containers will run continuously (default behavior)")
        else:
            print("✅ Docker socket available - GPU resource management enabled")
            print(f"   Idle timeout: {IDLE_TIMEOUT} seconds")
            print("   Strategy: Load only the model needed for each task")
    
    def _check_docker_available(self) -> bool:
        """Check if Docker socket is available."""
        if not os.path.exists(DOCKER_SOCKET):
            return False
        try:
            import docker
            self._docker_client = docker.from_env()
            self._docker_client.ping()
            return True
        except ImportError:
            print("⚠️ Docker SDK not installed. Run: pip install docker")
            return False
        except Exception as e:
            print(f"⚠️ Docker connection failed: {e}")
            return False
    
    def _get_container(self, name: str):
        """Get a container by name prefix."""
        if not self._docker_available:
            return None
        try:
            containers = self._docker_client.containers.list(all=True)
            for container in containers:
                if name in container.name:
                    return container
            return None
        except Exception as e:
            print(f"Error finding container {name}: {e}")
            return None
    
    def _is_container_running(self, name: str) -> bool:
        """Check if a container is running."""
        container = self._get_container(name)
        if container:
            return container.status == "running"
        return False
    
    def _start_container(self, name: str) -> bool:
        """Start a container if not running."""
        container = self._get_container(name)
        if container is None:
            print(f"⚠️ Container {name} not found")
            return False
        
        if container.status == "running":
            return True
        
        try:
            print(f"🚀 Starting container: {container.name}")
            container.start()
            return True
        except Exception as e:
            print(f"❌ Failed to start container {name}: {e}")
            return False
    
    
    def log_gpu_stats(self, context: str = ""):
        """Log current GPU memory usage via nvidia-smi."""
        try:
            import subprocess
            # Query memory.total, memory.used, memory.free
            result = subprocess.check_output(
                ["nvidia-smi", "--query-gpu=memory.total,memory.used,memory.free", "--format=csv,noheader,nounits"],
                encoding="utf-8"
            )
            # Parse first GPU (assuming single GPU for now)
            lines = result.strip().split('\n')
            if lines:
                total, used, free = map(int, lines[0].split(','))
                pct = (used / total) * 100
                print(f"📊 [GPU] {context}: {used}MiB / {total}MiB ({pct:.1f}%) | Free: {free}MiB")
        except Exception:
            pass # Silent failure for logging

    def _stop_container(self, name: str) -> bool:
        """Stop a container if running."""
        container = self._get_container(name)
        if container is None:
            return False
        
        if container.status != "running":
            return True
            
        self.log_gpu_stats(f"Pre-Stop {name}")
        
        try:
            print(f"🛑 Stopping container: {container.name}")
            container.stop(timeout=10)
            self.log_gpu_stats(f"Post-Stop {name}")
            return True
        except Exception as e:
            print(f"❌ Failed to stop container {name}: {e}")
            return False
    
    def _wait_for_service(self, host: str, port: str, timeout: int = CONTAINER_START_TIMEOUT) -> bool:
        """Wait for a service to be ready."""
        url = f"http://{host}:{port}/health"
        start_time = time.time()
        
        while time.time() - start_time < timeout:
            try:
                response = requests.get(url, timeout=5)
                if response.status_code == 200:
                    return True
            except requests.exceptions.RequestException:
                pass
            time.sleep(HEALTH_CHECK_INTERVAL)
        
        return False
    
    # =========================================================================
    # LLM Server Management (for emotion tagging)
    # =========================================================================
    
    def acquire_llm(self) -> Tuple[bool, str]:
        """
        Acquire the LLM server for emotion tag processing.
        Starts the container if needed. Call release_llm() when done.
        
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            # Cancel any pending idle shutdown
            if self._llm_idle_timer:
                self._llm_idle_timer.cancel()
                self._llm_idle_timer = None
            
            self._llm_users += 1
            
            if not self._docker_available:
                return True, "Docker management not available, assuming LLM server is running"
            
            if self._is_container_running("llm_server"):
                return True, "LLM server already running"
            
            # Start LLM server
            if self._start_container("llm_server"):
                print(f"⏳ Waiting for LLM server to be ready...")
                if self._wait_for_service(LLM_SERVER_HOST, LLM_SERVER_PORT):
                    print(f"✅ LLM server ready (~12GB VRAM)")
                    return True, "LLM server started"
                else:
                    self._llm_users -= 1
                    return False, "LLM server failed to start within timeout"
            else:
                self._llm_users -= 1
                return False, "Failed to start LLM server container"
    
    def release_llm(self, immediate: bool = False):
        """
        Release the LLM server after emotion tag processing.
        If no other users, schedules shutdown after idle timeout.
        
        Args:
            immediate: If True, stop immediately without waiting for idle timeout
        """
        with self._lock:
            self._llm_users = max(0, self._llm_users - 1)
            
            if self._llm_users == 0 and self._docker_available:
                if immediate:
                    print("🛑 Stopping LLM server immediately to free VRAM...")
                    self._stop_container("llm_server")
                elif IDLE_TIMEOUT > 0:
                    # Start idle timer
                    if self._llm_idle_timer:
                        self._llm_idle_timer.cancel()
                    
                    print(f"⏱️ LLM server idle timer started ({IDLE_TIMEOUT}s)")
                    self._llm_idle_timer = threading.Timer(IDLE_TIMEOUT, self._llm_idle_shutdown)
                    self._llm_idle_timer.daemon = True
                    self._llm_idle_timer.start()
    
    def _llm_idle_shutdown(self):
        """Called when LLM idle timeout expires."""
        with self._lock:
            if self._llm_users == 0:
                print("💤 LLM server idle timeout - stopping to free ~12GB VRAM...")
                self._stop_container("llm_server")
    
    # =========================================================================
    # Orpheus LLM Management (for TTS token generation)
    # =========================================================================
    
    def acquire_orpheus(self) -> Tuple[bool, str]:
        """
        Acquire the Orpheus LLM for TTS generation.
        Starts the container if needed. Call release_orpheus() when done.
        
        Returns:
            Tuple of (success, message)
        """
        with self._lock:
            # Cancel any pending idle shutdown
            if self._orpheus_idle_timer:
                self._orpheus_idle_timer.cancel()
                self._orpheus_idle_timer = None
            
            self._orpheus_users += 1
            
            if not self._docker_available:
                return True, "Docker management not available, assuming Orpheus LLM is running"
            
            if self._is_container_running("orpheus_llama"):
                return True, "Orpheus LLM already running"
            
            # Start Orpheus LLM
            if self._start_container("orpheus_llama"):
                print(f"⏳ Waiting for Orpheus LLM to be ready...")
                if self._wait_for_service(ORPHEUS_LLAMA_HOST, ORPHEUS_LLAMA_PORT):
                    print(f"✅ Orpheus LLM ready (~5GB VRAM)")
                    return True, "Orpheus LLM started"
                else:
                    self._orpheus_users -= 1
                    return False, "Orpheus LLM failed to start within timeout"
            else:
                self._orpheus_users -= 1
                return False, "Failed to start Orpheus LLM container"
    
    def release_orpheus(self, immediate: bool = False):
        """
        Release the Orpheus LLM after TTS generation.
        If no other users, schedules shutdown after idle timeout.
        
        Args:
            immediate: If True, stop immediately without waiting for idle timeout
        """
        with self._lock:
            self._orpheus_users = max(0, self._orpheus_users - 1)
            
            if self._orpheus_users == 0 and self._docker_available:
                if immediate:
                    print("🛑 Stopping Orpheus LLM immediately to free VRAM...")
                    self._stop_container("orpheus_llama")
                elif IDLE_TIMEOUT > 0:
                    # Start idle timer
                    if self._orpheus_idle_timer:
                        self._orpheus_idle_timer.cancel()
                    
                    print(f"⏱️ Orpheus LLM idle timer started ({IDLE_TIMEOUT}s)")
                    self._orpheus_idle_timer = threading.Timer(IDLE_TIMEOUT, self._orpheus_idle_shutdown)
                    self._orpheus_idle_timer.daemon = True
                    self._orpheus_idle_timer.start()
    
    def _orpheus_idle_shutdown(self):
        """Called when Orpheus idle timeout expires."""
        with self._lock:
            if self._orpheus_users == 0:
                print("💤 Orpheus LLM idle timeout - stopping to free ~5GB VRAM...")
                self._stop_container("orpheus_llama")

    # =========================================================================
    # VibeVoice Management (In-process, but needs to stop others)
    # =========================================================================

    def acquire_vibevoice(self) -> Tuple[bool, str]:
        """
        Acquire resources for VibeVoice (running in this process).
        Running VibeVoice requires stopping ALL other GPU containers to free ~24GB VRAM.
        """
        with self._lock:
            if not self._docker_available:
                return True, "Docker not available, proceeding without container management"

            # Force stop other services
            print("🛑 Stopping all other GPU services to make room for VibeVoice (~24GB)...")
            msg_success, msg_text = self.stop_llm_services()
            if msg_success:
                 return True, f"Environment prepared for VibeVoice: {msg_text}"
            return False, f"Failed to stop services: {msg_text}"

    def release_vibevoice(self):
        """
        Release VibeVoice resources.
        Since VibeVoice runs in-process, this mainly signals we are done.
        We can't really 'unload' it easily unless we restart the process or use del/gc,
        which is handled by VibeVoiceEngine.shutdown().
        """
        # Nothing specific to do here for Docker
        pass

    # =========================================================================
    # Maya Management (In-process, needs safe VRAM headroom)
    # =========================================================================

    def acquire_maya(self) -> Tuple[bool, str]:
        """
        Acquire resources for Maya 1 (running in this process).
        Requires ~8-12GB VRAM.
        To be safe, we stop other heavy services.
        """
        with self._lock:
            if not self._docker_available:
                return True, "Docker not available, proceeding without container management"

            # Force stop other known heavy GPU services
            print("🛑 Stopping other GPU services to make room for Maya (~12GB)...")
            msg_success, msg_text = self.stop_llm_services()
            if msg_success:
                 return True, f"Environment prepared for Maya: {msg_text}"
            return False, f"Failed to stop services: {msg_text}"

    def release_maya(self):
        """Release Maya resources."""
        pass

    # =========================================================================
    # Voice Analyzer Management (In-process, needs safe VRAM headroom)
    # =========================================================================

    def acquire_voice_analyzer(self) -> Tuple[bool, str]:
        """
        Acquire resources for Voice Analyzer (Qwen2-Audio, in-process).
        Requires ~16GB VRAM.
        To be safe, we stop all other GPU services.
        """
        with self._lock:
            if not self._docker_available:
                return True, "Docker not available, proceeding without container management"

            # Force stop other GPU services
            print("🛑 Stopping other GPU services to make room for Voice Analyzer (~16GB)...")
            msg_success, msg_text = self.stop_llm_services()
            if msg_success:
                 return True, f"Environment prepared for Voice Analyzer: {msg_text}"
            return False, f"Failed to stop services: {msg_text}"

    def release_voice_analyzer(self):
        """Release Voice Analyzer resources."""
        pass

    # =========================================================================
    # Whisper Verification Management (In-process, needs safe VRAM headroom)
    # =========================================================================

    def acquire_whisper(self) -> Tuple[bool, str]:
        """
        Acquire resources for Whisper Verification (faster-whisper, in-process).
        Requires ~2-4GB VRAM for small/medium, more for large.
        We stop other heavy services to be safe.
        """
        with self._lock:
            if not self._docker_available:
                return True, "Docker not available, proceeding without container management"

            # Force stop other GPU services
            print("🛑 Stopping other GPU services to make room for Whisper...")
            msg_success, msg_text = self.stop_llm_services()
            if msg_success:
                 return True, f"Environment prepared for Whisper: {msg_text}"
            return False, f"Failed to stop services: {msg_text}"

    def release_whisper(self):
        """Release Whisper resources."""
        pass


    
    # =========================================================================
    # Combined operations (for backward compatibility and manual control)
    # =========================================================================
    
    def start_llm_services(self, need_llm: bool = True, need_orpheus: bool = True) -> Tuple[bool, str]:
        """Start specified LLM services."""
        messages = []
        
        if need_llm:
            success, msg = self.acquire_llm()
            if not success:
                return False, msg
            messages.append(msg)
        
        if need_orpheus:
            success, msg = self.acquire_orpheus()
            if not success:
                return False, msg
            messages.append(msg)
        
        return True, "; ".join(messages) if messages else "No services requested"
    
    def stop_llm_services(self) -> Tuple[bool, str]:
        """Stop all LLM services to free GPU memory."""
        if not self._docker_available:
            return True, "Docker management not available"
        
        stopped = []
        
        # Force stop both services
        self._llm_users = 0
        self._orpheus_users = 0
        
        if self._llm_idle_timer:
            self._llm_idle_timer.cancel()
            self._llm_idle_timer = None
        
        if self._orpheus_idle_timer:
            self._orpheus_idle_timer.cancel()
            self._orpheus_idle_timer = None
        
        if self._is_container_running("llm_server"):
            if self._stop_container("llm_server"):
                stopped.append("llm_server (~12GB freed)")
        
        if self._is_container_running("orpheus_llama"):
            if self._stop_container("orpheus_llama"):
                stopped.append("orpheus_llama (~5GB freed)")
        
        return True, f"Stopped: {', '.join(stopped)}" if stopped else "No services were running"
    
    def get_status(self) -> dict:
        """Get current status of GPU resources."""
        status = {
            "docker_available": self._docker_available,
            "llm_users": self._llm_users,
            "orpheus_users": self._orpheus_users,
            "idle_timeout": IDLE_TIMEOUT,
        }
        
        if self._docker_available:
            status["llm_server_running"] = self._is_container_running("llm_server")
            status["orpheus_llama_running"] = self._is_container_running("orpheus_llama")
            
            # Estimate VRAM usage
            vram_used = 0
            if status["llm_server_running"]:
                vram_used += 12  # ~12GB for gpt-oss
            if status["orpheus_llama_running"]:
                vram_used += 5  # ~5GB for Orpheus
            status["estimated_vram_gb"] = vram_used
        
        return status
    
    # =========================================================================
    # Context managers for cleaner resource management
    # =========================================================================
    
    @contextmanager
    def llm_context(self):
        """
        Context manager for LLM server usage (emotion tagging).
        Automatically acquires and releases the LLM server.
        
        Usage:
            with gpu_manager.llm_context():
                # Do emotion tagging
                pass
            # LLM server will be released (and stopped if idle)
        """
        success, message = self.acquire_llm()
        if not success:
            raise RuntimeError(f"Failed to acquire LLM server: {message}")
        
        try:
            yield message
        finally:
            self.release_llm(immediate=True)  # Stop immediately to free VRAM for next task
    
    @contextmanager
    def orpheus_context(self):
        """
        Context manager for Orpheus LLM usage (TTS generation).
        Automatically acquires and releases Orpheus LLM.
        
        Usage:
            with gpu_manager.orpheus_context():
                # Do TTS generation
                pass
            # Orpheus LLM will be released (and stopped if idle)
        """
        success, message = self.acquire_orpheus()
        if not success:
            raise RuntimeError(f"Failed to acquire Orpheus LLM: {message}")
        
        try:
            yield message
        finally:
            self.release_orpheus()  # Use idle timeout for TTS (might do multiple samples)
    
    # Legacy methods for backward compatibility
    def acquire_gpu(self, need_llm: bool = True, need_orpheus: bool = True) -> Tuple[bool, str]:
        """Legacy method - use acquire_llm() or acquire_orpheus() instead."""
        return self.start_llm_services(need_llm, need_orpheus)
    
    def release_gpu(self):
        """Legacy method - use release_llm() or release_orpheus() instead."""
        # Don't force stop, just decrement counters
        if self._llm_users > 0:
            self.release_llm()
        if self._orpheus_users > 0:
            self.release_orpheus()


# Singleton instance
gpu_manager = GPUResourceManager()
