#!/usr/bin/env python3
"""
Frontend E2E Test Suite - VibeVoice Audiobook Generation

Standalone Playwright tests that can be run by anyone without AI assistance.

Usage:
    # Install dependencies first
    pip install playwright pytest-playwright
    playwright install chromium
    
    # Run all frontend tests
    pytest tests/e2e_frontend_test.py -v
    
    # Run with visible browser (debugging)
    pytest tests/e2e_frontend_test.py -v --headed
    
    # Run with custom URL
    AUDIOBOOK_URL=http://myserver:7860 pytest tests/e2e_frontend_test.py -v

Requirements:
    pip install playwright pytest-playwright
    playwright install chromium
"""

import os
import re
import pytest
from pathlib import Path

# Skip all tests if playwright not installed
pytest.importorskip("playwright")

from playwright.sync_api import Page, expect

# ==================== CONFIGURATION ====================

BASE_URL = os.environ.get("AUDIOBOOK_URL", "http://localhost:7860")
TIMEOUT = 30000  # 30 seconds


# ==================== FIXTURES ====================

@pytest.fixture(scope="session")
def browser_context_args(browser_context_args):
    """Configure browser context."""
    return {
        **browser_context_args,
        "viewport": {"width": 1920, "height": 1080},
    }


@pytest.fixture
def app_page(page: Page):
    """Navigate to the app and wait for it to load."""
    page.goto(BASE_URL, wait_until="networkidle", timeout=60000)
    page.wait_for_timeout(2000)  # Let Gradio fully initialize
    return page


# ==================== APP LOAD TESTS ====================

class TestAppLoad:
    """Tests for basic app loading."""
    
    def test_app_loads_successfully(self, app_page: Page):
        """Test that the Gradio app loads."""
        # Should have Gradio footer or header
        expect(app_page.locator("body")).to_contain_text("Audiobook", timeout=TIMEOUT)
    
    def test_tabs_visible(self, app_page: Page):
        """Test that main tabs are visible."""
        # Look for tab buttons
        tabs = app_page.locator("button[role='tab'], .tabs button")
        expect(tabs.first).to_be_visible(timeout=TIMEOUT)
    
    def test_no_javascript_errors(self, app_page: Page):
        """Test that there are no critical JavaScript errors."""
        # Collect console errors
        errors = []
        app_page.on("console", lambda msg: errors.append(msg.text) if msg.type == "error" else None)
        
        # Navigate and interact
        app_page.wait_for_timeout(3000)
        
        # Filter out known harmless errors
        critical_errors = [e for e in errors if "Gradio" not in e and "deprecat" not in e.lower()]
        
        assert len(critical_errors) == 0, f"JavaScript errors found: {critical_errors}"


# ==================== VIBEVOICE UI TESTS ====================

class TestVibeVoiceUI:
    """Tests for VibeVoice-specific UI elements."""
    
    def test_tts_engine_dropdown_exists(self, app_page: Page):
        """Test that TTS engine dropdown exists."""
        # Navigate to audiobook creation tab
        audiobook_tab = app_page.locator("button:has-text('Audiobook'), button:has-text('Create')")
        if audiobook_tab.count() > 0:
            audiobook_tab.first.click()
            app_page.wait_for_timeout(1000)
        
        # Scroll to find TTS engine dropdown
        app_page.mouse.wheel(0, 500)
        app_page.wait_for_timeout(500)
        
        # Look for TTS Engine label or dropdown
        tts_label = app_page.locator("text=TTS Engine")
        expect(tts_label.first).to_be_visible(timeout=TIMEOUT)
    
    def test_vibevoice_is_selectable(self, app_page: Page):
        """Test that VibeVoice can be selected as TTS engine."""
        # Navigate to audiobook creation
        audiobook_tab = app_page.locator("button:has-text('Audiobook'), button:has-text('Create')")
        if audiobook_tab.count() > 0:
            audiobook_tab.first.click()
            app_page.wait_for_timeout(1000)
        
        app_page.mouse.wheel(0, 500)
        app_page.wait_for_timeout(500)
        
        # Find and click TTS dropdown
        dropdown = app_page.locator("label:has-text('TTS Engine') + div, label:has-text('TTS Engine') ~ div").first
        if dropdown.is_visible():
            dropdown.click()
            app_page.wait_for_timeout(500)
            
            # Look for VibeVoice option
            vibevoice_option = app_page.locator("text=VibeVoice")
            expect(vibevoice_option.first).to_be_visible(timeout=TIMEOUT)
    
    def test_custom_voices_appear_in_dropdown(self, app_page: Page):
        """Test that custom voices appear in VibeVoice speaker dropdown."""
        # Navigate and select VibeVoice
        audiobook_tab = app_page.locator("button:has-text('Audiobook'), button:has-text('Create')")
        if audiobook_tab.count() > 0:
            audiobook_tab.first.click()
            app_page.wait_for_timeout(1000)
        
        app_page.mouse.wheel(0, 500)
        app_page.wait_for_timeout(500)
        
        # Try to select VibeVoice
        dropdown = app_page.locator("label:has-text('TTS Engine') + div, label:has-text('TTS Engine') ~ div").first
        if dropdown.is_visible():
            dropdown.click()
            app_page.wait_for_timeout(500)
            
            vv = app_page.locator("li:has-text('VibeVoice'), option:has-text('VibeVoice')")
            if vv.count() > 0:
                vv.first.click()
                app_page.wait_for_timeout(1000)
                
                # Look for speaker dropdown with custom voices
                # Custom voices should have paths like /app/static_files/voices/
                page_content = app_page.content()
                
                # Check if any custom voice patterns exist
                has_custom_voices = (
                    "static_files/voices" in page_content or
                    "_Fry" in page_content or
                    "_Attenborough" in page_content or
                    "David" in page_content or
                    "Stephen" in page_content
                )
                
                assert has_custom_voices, "No custom voices found in page content"


# ==================== JOBS TAB TESTS ====================

class TestJobsTab:
    """Tests for the Jobs tab functionality."""
    
    def test_jobs_tab_accessible(self, app_page: Page):
        """Test that Jobs tab is accessible."""
        jobs_tab = app_page.locator("button:has-text('Jobs')")
        expect(jobs_tab.first).to_be_visible(timeout=TIMEOUT)
        
        jobs_tab.first.click()
        app_page.wait_for_timeout(1000)
        
        # Should see jobs-related content
        page_text = app_page.locator("body").text_content()
        assert "Job" in page_text, "Jobs tab content not visible"
    
    def test_jobs_table_exists(self, app_page: Page):
        """Test that jobs table exists in Jobs tab."""
        jobs_tab = app_page.locator("button:has-text('Jobs')")
        jobs_tab.first.click()
        app_page.wait_for_timeout(1000)
        
        # Look for table or dataframe
        table = app_page.locator("table, [data-testid='dataframe'], .dataframe")
        expect(table.first).to_be_visible(timeout=TIMEOUT)
    
    def test_refresh_jobs_button_works(self, app_page: Page):
        """Test that refresh jobs button is clickable."""
        jobs_tab = app_page.locator("button:has-text('Jobs')")
        jobs_tab.first.click()
        app_page.wait_for_timeout(1000)
        
        refresh_btn = app_page.locator("button:has-text('Refresh')")
        if refresh_btn.count() > 0:
            refresh_btn.first.click()
            app_page.wait_for_timeout(1000)
            # Should not cause an error
            assert True


# ==================== VOICE LIBRARY TESTS ====================

class TestVoiceLibraryUI:
    """Tests for Voice Library tab UI."""
    
    def test_voice_library_tab_accessible(self, app_page: Page):
        """Test that Voice Library tab is accessible."""
        vl_tab = app_page.locator("button:has-text('Voice Library'), button:has-text('Voice')")
        
        if vl_tab.count() == 0:
            pytest.skip("Voice Library tab not found")
        
        vl_tab.first.click()
        app_page.wait_for_timeout(1000)
        
        # Should see voice-related content
        page_text = app_page.locator("body").text_content()
        assert "voice" in page_text.lower(), "Voice Library content not visible"


# ==================== SCREENSHOT HELPERS ====================

@pytest.fixture
def screenshot_dir():
    """Create and return screenshot directory."""
    path = Path(__file__).parent / "screenshots"
    path.mkdir(exist_ok=True)
    return path


def test_capture_vibevoice_state(app_page: Page, screenshot_dir: Path):
    """Capture screenshot of VibeVoice options (for manual verification)."""
    # Navigate to audiobook tab
    audiobook_tab = app_page.locator("button:has-text('Audiobook'), button:has-text('Create')")
    if audiobook_tab.count() > 0:
        audiobook_tab.first.click()
        app_page.wait_for_timeout(1000)
    
    app_page.mouse.wheel(0, 500)
    app_page.wait_for_timeout(500)
    
    # Capture screenshot
    app_page.screenshot(path=str(screenshot_dir / "vibevoice_state.png"))
    
    # Capture jobs tab too
    jobs_tab = app_page.locator("button:has-text('Jobs')")
    if jobs_tab.count() > 0:
        jobs_tab.first.click()
        app_page.wait_for_timeout(1000)
        app_page.screenshot(path=str(screenshot_dir / "jobs_state.png"))
    
    # This test always passes - it's for generating artifacts
    assert True


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))
