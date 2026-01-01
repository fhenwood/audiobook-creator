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
import time

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
    # Use 'load' instead of 'networkidle' because Gradio keeps WebSocket connections open
    # which causes 'networkidle' to never resolve
    page.goto(BASE_URL, wait_until="load", timeout=60000)
    
    # Wait for Gradio to fully initialize by looking for a common element
    page.wait_for_selector("button, h1, .gradio-container", timeout=30000)
    page.wait_for_timeout(2000)  # Extra time for Gradio components to render
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


# ==================== UI FIX VERIFICATION TESTS ====================

class TestUIFixes:
    """Tests specifically targeting recent UI visibility fixes."""
    
    def _click_tab(self, app_page: Page, tab_text: str):
        """Navigate to a tab by clicking it."""
        # Gradio tabs are buttons with role='tab'
        tab = app_page.locator(f"button[role='tab']:has-text('{tab_text}')")
        if tab.count() > 0:
            tab.first.click(force=True)
            app_page.wait_for_timeout(1000)
            return True
        return False
    
    def test_maya_visibility(self, app_page: Page):
        """Test that Maya TTS engine option exists in the app."""
        # Navigate to Create Audiobook tab
        self._click_tab(app_page, "Create Audiobook")
        
        # Check that the page contains Maya as a TTS option
        # The app should have Maya mentioned somewhere
        page_content = app_page.content()
        assert "Maya" in page_content or "maya" in page_content, "Maya TTS option not found in page"
        
        # Check for TTS Engine related content
        tts_elements = app_page.locator("text=/TTS Engine|TTS|Engine/i")
        assert tts_elements.count() > 0, "TTS Engine controls not found"
    
    def test_vibevoice_visibility(self, app_page: Page):
        """Test that VibeVoice TTS engine option exists in the app."""
        # Navigate to Create Audiobook tab
        self._click_tab(app_page, "Create Audiobook")
        
        # Check that the page contains VibeVoice as a TTS option
        page_content = app_page.content()
        assert "VibeVoice" in page_content or "vibevoice" in page_content, "VibeVoice TTS option not found in page"
        
        # Check that voice selection controls exist
        voice_elements = app_page.locator("text=/voice|Voice|speaker|Speaker/i")
        assert voice_elements.count() > 0, "Voice selection controls not found"

    def test_chapter_selection_exists(self, app_page: Page):
        """Test that Chapter Selection functionality exists."""
        # Navigate to Create Audiobook tab
        self._click_tab(app_page, "Create Audiobook")
        app_page.mouse.wheel(0, 1000)  # Scroll down
        app_page.wait_for_timeout(500)
        
        # Check for chapter-related content in the page
        page_content = app_page.content()
        has_chapter_content = any(word in page_content for word in ["Chapter", "chapter", "Extract", "extract"])
        
        assert has_chapter_content, "Chapter selection UI not found in page"




# ==================== JOBS TAB TESTS ====================

class TestJobsTab:
    """Tests for the Jobs tab functionality."""
    
    def _click_element(self, app_page: Page, locator):
        """Click an element, using JavaScript as fallback if normal click fails."""
        try:
            locator.click(force=True, timeout=5000)
        except Exception:
            # Fallback to JavaScript click
            app_page.evaluate("el => el.click()", locator.element_handle())
    
    def test_jobs_tab_accessible(self, app_page: Page):
        """Test that Jobs tab is accessible."""
        jobs_tab = app_page.locator("button:has-text('Jobs')")
        expect(jobs_tab.first).to_be_visible(timeout=TIMEOUT)
        
        self._click_element(app_page, jobs_tab.first)
        app_page.wait_for_timeout(1000)
        
        # Should see jobs-related content
        page_text = app_page.locator("body").text_content()
        assert "Job" in page_text, "Jobs tab content not visible"
    
    def test_jobs_table_exists(self, app_page: Page):
        """Test that jobs table exists in Jobs tab."""
        jobs_tab = app_page.locator("button:has-text('Jobs')")
        self._click_element(app_page, jobs_tab.first)
        app_page.wait_for_timeout(1000)
        
        # Look for job-related content - Gradio renders various ways
        # Check for table, dataframe, or job-related text
        job_content_selectors = [
            "table",
            "[data-testid='dataframe']",
            ".dataframe",
            "text=Job ID",
            "text=Status",
            "text=No jobs",
        ]
        
        found = False
        for selector in job_content_selectors:
            loc = app_page.locator(selector)
            if loc.count() > 0:
                found = True
                break
        
        assert found, "Jobs tab content not properly rendered"


# ==================== MAIN ====================

if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-v"]))

