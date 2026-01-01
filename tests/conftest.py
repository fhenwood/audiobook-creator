import pytest
from playwright.sync_api import sync_playwright

@pytest.fixture(scope="session")
def browser_context():
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True)
        context = browser.new_context()
        yield context
        browser.close()

@pytest.fixture(scope="function")
def page(browser_context):
    page = browser_context.new_page()
    page.goto("http://localhost:7860")
    # Wait for Gradio to load
    page.wait_for_selector("text=Audiobook Creator", timeout=10000)
    yield page
    page.close()
