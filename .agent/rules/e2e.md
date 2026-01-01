---
trigger: always_on
---

# Agent Rules: E2E Testing

## Core Principles

1. **No feature is complete without E2E tests** — Every new feature or update must include corresponding E2E tests before being considered done.

2. **Run before reporting** — Always execute E2E tests and confirm they pass before telling the user to test or that a task is complete.

## Rules

### Test Coverage

- Write E2E tests for every new feature, bug fix, or significant update.
- Update existing E2E tests when modifying related functionality.
- Ensure tests cover both happy paths and critical edge cases.

### Test Execution

- Run the full E2E test suite (or relevant subset) after making changes.
- Do not mark a task as complete until all E2E tests pass.
- If tests fail, fix the issue before notifying the user.

### Reporting

- When informing the user that a feature is ready to test:
  - Confirm E2E tests have been written and executed.
  - Report the test results (e.g., "All 12 E2E tests passing").
  - Only then suggest manual testing if needed.

## Example Workflow

1. Implement feature/fix
2. Write or update E2E tests
3. Run E2E tests
4. If tests fail → debug and fix → re-run tests
5. Once all tests pass → inform user the feature is ready