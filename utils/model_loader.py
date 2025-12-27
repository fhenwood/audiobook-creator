"""
Audiobook Creator - Model Loader Utility
Copyright (C) 2025 Prakhar Sharma

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
"""

import os
import time
import httpx
from dotenv import load_dotenv

load_dotenv()

# Configuration
LLM_BASE_URL = os.environ.get("LLM_BASE_URL", "http://localhost:8000/v1")
TTS_BASE_URL = os.environ.get("TTS_BASE_URL", "http://localhost:8880/v1")
MAX_RETRIES = 30  # Maximum retries for service health checks
RETRY_INTERVAL = 10  # Seconds between retries


def check_llm_service_health(timeout: int = 10) -> tuple[bool, str]:
    """
    Check if the LLM service is healthy and responding.
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        tuple: (is_healthy, message)
    """
    try:
        # Try to hit the models endpoint to check if service is up
        health_url = f"{LLM_BASE_URL.rstrip('/v1')}/health"
        models_url = f"{LLM_BASE_URL}/models"
        
        with httpx.Client(timeout=timeout) as client:
            # Try health endpoint first
            try:
                response = client.get(health_url)
                if response.status_code == 200:
                    return True, "LLM service is healthy"
            except:
                pass
            
            # Fall back to models endpoint
            try:
                response = client.get(models_url)
                if response.status_code == 200:
                    return True, "LLM service is responding"
            except:
                pass
                
        return False, "LLM service is not responding"
    except Exception as e:
        return False, f"LLM service check failed: {str(e)}"


def check_tts_service_health(timeout: int = 10) -> tuple[bool, str]:
    """
    Check if the Orpheus TTS service is healthy and responding.
    
    Args:
        timeout: Request timeout in seconds
        
    Returns:
        tuple: (is_healthy, message)
    """
    try:
        # Try to hit the voices endpoint to check if TTS service is up
        voices_url = f"{TTS_BASE_URL}/audio/voices"
        
        with httpx.Client(timeout=timeout) as client:
            response = client.get(voices_url)
            if response.status_code == 200:
                return True, "TTS service is healthy"
                
        return False, "TTS service is not responding"
    except Exception as e:
        return False, f"TTS service check failed: {str(e)}"


def wait_for_services(llm: bool = True, tts: bool = True) -> tuple[bool, str]:
    """
    Wait for LLM and/or TTS services to become available.
    Uses exponential backoff with maximum retries.
    
    Args:
        llm: Whether to wait for LLM service
        tts: Whether to wait for TTS service
        
    Returns:
        tuple: (all_healthy, status_message)
    """
    services_status = []
    
    if llm:
        print("⏳ Waiting for LLM service to be ready...")
        llm_ready = False
        for attempt in range(MAX_RETRIES):
            is_healthy, message = check_llm_service_health()
            if is_healthy:
                print(f"✅ LLM service is ready!")
                llm_ready = True
                break
            else:
                print(f"   Attempt {attempt + 1}/{MAX_RETRIES}: {message}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_INTERVAL)
        
        if not llm_ready:
            services_status.append("LLM service failed to start")
    
    if tts:
        print("⏳ Waiting for TTS service to be ready...")
        tts_ready = False
        for attempt in range(MAX_RETRIES):
            is_healthy, message = check_tts_service_health()
            if is_healthy:
                print(f"✅ TTS service is ready!")
                tts_ready = True
                break
            else:
                print(f"   Attempt {attempt + 1}/{MAX_RETRIES}: {message}")
                if attempt < MAX_RETRIES - 1:
                    time.sleep(RETRY_INTERVAL)
        
        if not tts_ready:
            services_status.append("TTS service failed to start")
    
    if services_status:
        return False, "; ".join(services_status)
    
    return True, "All services are ready"


def get_service_status() -> dict:
    """
    Get the current status of all services.
    
    Returns:
        dict: Status of each service
    """
    llm_healthy, llm_message = check_llm_service_health(timeout=5)
    tts_healthy, tts_message = check_tts_service_health(timeout=5)
    
    return {
        "llm": {
            "healthy": llm_healthy,
            "message": llm_message,
            "url": LLM_BASE_URL
        },
        "tts": {
            "healthy": tts_healthy,
            "message": tts_message,
            "url": TTS_BASE_URL
        }
    }


def print_service_status():
    """Print a formatted status of all services."""
    status = get_service_status()
    
    print("\n" + "=" * 50)
    print("📊 Service Status")
    print("=" * 50)
    
    # LLM Status
    llm = status["llm"]
    llm_icon = "✅" if llm["healthy"] else "❌"
    print(f"\n🤖 LLM Service: {llm_icon}")
    print(f"   URL: {llm['url']}")
    print(f"   Status: {llm['message']}")
    
    # TTS Status
    tts = status["tts"]
    tts_icon = "✅" if tts["healthy"] else "❌"
    print(f"\n🔊 TTS Service (Orpheus): {tts_icon}")
    print(f"   URL: {tts['url']}")
    print(f"   Status: {tts['message']}")
    
    print("\n" + "=" * 50)
    
    return all(s["healthy"] for s in status.values())


if __name__ == "__main__":
    # Run as standalone script to check service status
    import sys
    
    all_healthy = print_service_status()
    
    if not all_healthy:
        print("\n⚠️  Some services are not available.")
        print("💡 Make sure docker compose is running:")
        print("   docker compose up -d")
        sys.exit(1)
    else:
        print("\n🎉 All services are ready!")
        sys.exit(0)
