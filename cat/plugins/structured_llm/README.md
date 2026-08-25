# Structured JSON LLM

Utility plugin to call the configured LLM and obtain structured JSON output, validated against Pydantic models.

## Purpose

Use the framework's configured LLM to produce structured JSON data that matches a given Pydantic schema, enforced programmatically.

## Usage

This plugin provides a declarative mechanism: define your expected output shape as a Pydantic model and call the configured LLM to fill it in. Responses are validated against the schema.

This plugin does NOT replace the configured conversation AgenticWorkflow.