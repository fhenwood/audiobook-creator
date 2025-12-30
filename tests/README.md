# E2E Test Suite for Audiobook Creator

This directory contains end-to-end tests for the Audiobook Creator application.
All tests are **fully standalone** and can be run by anyone without AI assistance.

## Quick Start

```bash
# Install test dependencies
pip install pytest requests playwright pytest-playwright

# Install browser for frontend tests
playwright install chromium

# Run all backend tests
pytest tests/e2e_vibevoice_test.py -v

# Run all frontend tests
pytest tests/e2e_frontend_test.py -v

# Run ALL tests
pytest tests/ -v
```

## Test Files

### `e2e_vibevoice_test.py` - Backend Tests

Tests the application's backend systems:

| Test Class | What It Tests |
|------------|---------------|
| `TestHealthChecks` | App responding, API mounted, static files |
| `TestJobsSystem` | jobs.json validity, job structure, progress |
| `TestVoiceLibrary` | Voice directories exist, audio files present |
| `TestGPU` | nvidia-smi available, GPU not OOM |
| `TestIntegration` | Directories writable, permissions |

### `e2e_frontend_test.py` - Frontend Tests

Tests the application's UI via browser automation:

| Test Class | What It Tests |
|------------|---------------|
| `TestAppLoad` | App loads, tabs visible, no JS errors |
| `TestVibeVoiceUI` | TTS dropdown, VibeVoice selectable, custom voices |
| `TestJobsTab` | Jobs tab accessible, table exists, refresh works |
| `TestVoiceLibraryUI` | Voice Library tab accessible |

### `e2e_architecture_test.py` - Architecture Tests

Tests the new MacWhisper-style API and refactored modules:

| Test Class | What It Tests |
|------------|---------------|
| `TestAPIHealth` | `/api/health`, `/api/engines` |
| `TestVoiceLibraryAPI` | `/api/voice-library` CRUD |
| `TestSettingsAPI` | `/api/settings` GET/PUT |
| `TestJobsAPI` | `/api/jobs` listing |
| `TestModelsAPI` | `/api/models` listing |
| `TestVoicesAPI` | `/api/engines/{engine}/voices` |
| `TestAppModules` | `app.voice_utils`, `app.handlers`, `app.jobs` imports |
| `TestGeneratorPackage` | `audiobook.tts.generator.utils` imports |
| `TestOrpheusEngine` | `OrpheusEngine` class and voices |


## Configuration

### Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `AUDIOBOOK_URL` | `http://localhost:7860` | Base URL of the app |

### Examples

```bash
# Test against a custom server
AUDIOBOOK_URL=http://myserver:7860 pytest tests/ -v

# Run with visible browser (debugging)
pytest tests/e2e_frontend_test.py -v --headed

# Run specific test class
pytest tests/e2e_vibevoice_test.py::TestJobsSystem -v

# Run with verbose output
pytest tests/ -v --tb=long
```

## Writing New Tests

### Backend Test Example

```python
class TestMyFeature:
    def test_something_works(self, session, base_url):
        response = session.get(f"{base_url}/my-endpoint")
        assert response.status_code == 200
```

### Frontend Test Example

```python
class TestMyUI:
    def test_button_clickable(self, app_page: Page):
        button = app_page.locator("button:has-text('My Button')")
        expect(button).to_be_visible()
        button.click()
```

## CI/CD Integration

Add to your GitHub Actions workflow:

```yaml
- name: Install dependencies
  run: |
    pip install pytest requests playwright pytest-playwright
    playwright install chromium

- name: Run E2E tests
  run: pytest tests/ -v --junitxml=test-results.xml
  env:
    AUDIOBOOK_URL: http://localhost:7860
```

## Troubleshooting

### "playwright not found"
```bash
pip install playwright pytest-playwright
playwright install chromium
```

### "Connection refused"
Make sure the app is running:
```bash
docker compose up -d
# Wait for startup
sleep 30
pytest tests/ -v
```

### "GPU tests skipped"
This is normal on CPU-only systems. GPU tests require `nvidia-smi`.

## Screenshots

Frontend tests can capture screenshots for debugging:

```bash
pytest tests/e2e_frontend_test.py::test_capture_vibevoice_state -v
# Screenshots saved to tests/screenshots/
```
