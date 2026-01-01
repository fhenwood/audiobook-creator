Feature: TTS Generation
    As a user
    I want to generate speech using different TTS engines
    So that I can preview voices before creating an audiobook

    Scenario: Generate audio with Maya (Preset)
        Given the Audiobook Creator app is open
        When I select the "Voice Sampling" tab
        And I select "Maya" as the TTS Engine
        And I select "Maya Professional (Male)" from the voice dropdown
        And I enter "Hello, this is a test of Maya." into the sample text box
        And I click the "Generate Sample" button
        Then I should see a success message containing "Generated sample with maya"
        And an audio player should appear with the generated file

    Scenario: Generate audio with VibeVoice
        Given the Audiobook Creator app is open
        When I select the "Voice Sampling" tab
        And I select "VibeVoice" as the TTS Engine
        And I select "Speaker 1" from the VibeVoice dropdown
        And I enter "Testing VibeVoice generation." into the sample text box
        And I click the "Generate Sample" button
        Then I should see a success message containing "Generated sample with vibevoice"
        And an audio player should appear with the generated file
