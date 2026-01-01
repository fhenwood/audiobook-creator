Feature: Voice Cloning
    As a user
    I want to analyze my voice and generate a clone description
    So that I can use a custom voice for TTS

    Scenario: Analyze Voice Sample
        Given the Audiobook Creator app is open
        When I select the "Voice Analysis & Cloning" tab
        And I upload the file "tests/resources/reference_audio.wav" to the "Reference Voice Sample" input
        And I click the "Analyze Voice" button
        Then I should see a voice description in the output box
        And the description should contain text (non-empty)

    Scenario: Generate with Cloned Voice Description
        Given the Audiobook Creator app is open
        When I select the "Voice Sampling" tab
        And I select "Maya" as the TTS Engine
        And I enter "Custom voice description testing." into the "Voice Description" box
        And I enter "This is my cloned voice speaking." into the sample text box
        And I click the "Generate Sample" button
        Then I should see a success message containing "Generated sample with maya"
        And an audio player should appear with the generated file
