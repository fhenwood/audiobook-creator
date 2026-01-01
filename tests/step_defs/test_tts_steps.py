import re
from pytest_bdd import scenario, given, when, then, parsers
from playwright.sync_api import Page, expect

# Scenarios
@scenario('../features/tts_generation.feature', 'Generate audio with Maya (Preset)')
def test_maya_preset():
    pass

@scenario('../features/tts_generation.feature', 'Generate audio with VibeVoice')
def test_vibevoice():
    pass

@scenario('../features/voice_cloning.feature', 'Analyze Voice Sample')
def test_analyze_voice():
    pass

@scenario('../features/voice_cloning.feature', 'Generate with Cloned Voice Description')
def test_clone_description():
    pass


# Given Steps
@given('the Audiobook Creator app is open')
def app_open(page: Page):
    # Wait for the main H1 heading to confirm app is loaded
    expect(page.get_by_role("heading", name="Audiobook Creator")).to_be_visible(timeout=20000)

# When Steps
@when(parsers.parse('I select the "{tab_name}" tab'))
def select_tab(page: Page, tab_name: str):
    page.click(f"button:has-text('{tab_name}')")

@when(parsers.parse('I select "{engine_name}" as the TTS Engine'))
def select_engine(page: Page, engine_name: str):
    page.check(f"label:has-text('{engine_name}')")

@when(parsers.parse('I select "{voice_name}" from the voice dropdown'))
def select_voice(page: Page, voice_name: str):
    # This might depend on the specific dropdown implementation in Gradio
    # Assuming standard dropdown
    page.click("input[aria-label='voice_selector']") # Adjust selector if needed
    page.click(f"li:has-text('{voice_name}')")

@when(parsers.parse('I select "{voice_name}" from the VibeVoice dropdown'))
def select_vibevoice(page: Page, voice_name: str):
    page.click("input[aria-label='vibevoice_selector']") 
    page.click(f"li:has-text('{voice_name}')")

@when(parsers.parse('I enter "{text}" into the sample text box'))
def enter_sample_text(page: Page, text: str):
    page.fill("textarea[label='Text to Speak']", text)

@when(parsers.parse('I click the "{button_name}" button'))
def click_button(page: Page, button_name: str):
    page.click(f"button:has-text('{button_name}')")

@when(parsers.parse('I upload the file "{file_path}" to the "{input_label}" input'))
def upload_file(page: Page, file_path: str, input_label: str):
    with page.expect_file_chooser() as fc_info:
        page.click(f"//div[contains(text(), '{input_label}')]//parent::div//input[@type='file']")
    file_chooser = fc_info.value
    file_chooser.set_files(file_path)

@when(parsers.parse('I enter "{text}" into the "Voice Description" box'))
def enter_description(page: Page, text: str):
     page.fill("textarea[label='Voice Description']", text)

# Then Steps
@then(parsers.parse('I should see a success message containing "{message}"'))
def verify_success(page: Page, message: str):
    expect(page.locator("body")).to_contain_text(message)

@then('an audio player should appear with the generated file')
def verify_audio_player(page: Page):
    expect(page.locator("audio")).to_be_visible()

@then('I should see a voice description in the output box')
def verify_description_output(page: Page):
    expect(page.locator("textarea[label='Generated Voice Description']")).to_be_visible()

@then('the description should contain text (non-empty)')
def verify_description_content(page: Page):
    content = page.input_value("textarea[label='Generated Voice Description']")
    assert len(content) > 10
