# Classification Core

Shared infrastructure used by all transaction classification engines.

This package contains:

- engine contracts and result models;
- priority orchestration and transaction claiming;
- engine registration and configuration loading;
- transaction-key validation;
- unified Excel/CSV reporting and formatting.

It does not contain income or liability business rules. Those remain inside
their respective engine packages. The project-level entry point is the root
`main.py`.
