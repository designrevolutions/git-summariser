# GitHub Repository Summarizer API

This project implements a FastAPI service that analyses a public GitHub repository and generates a structured summary using a Nebius-hosted large language model (LLM).

The service performs repository analysis locally, builds structured context describing the project, and then sends that context to the LLM to generate a concise summary of the repository.

The API returns:

- a short human-readable project summary
- the main detected technologies
- a brief description of the repository structure

---

# Architecture Overview

The service performs the following steps:

1. Validate a public GitHub repository URL.
2. Extract the repository owner and name.
3. Verify that the repository is accessible.
4. Download the repository as a ZIP archive.
5. Extract the repository into a temporary working directory.
6. Traverse the repository while ignoring large dependency/build/cache folders.
7. Collect structural metadata such as:
   - file count
   - file extensions
   - important project files
8. Infer likely technologies using file extensions and marker files.
9. Select a small number of high-signal files (e.g. README, dependency files).
10. Build structured repository context.
11. Send the prepared context to a Nebius-hosted LLM.
12. Parse the LLM response and return a structured API response.

---

# Requirements

- Python **3.10+**
- A **Nebius API key**

---

# Installation

## 1. Clone the repository

```bash
git clone <repository-url>
cd <repository-folder>
```

---

## 2. Create a virtual environment

It is recommended to use a virtual environment to isolate dependencies.

### Linux / macOS

```bash
python -m venv venv
source venv/bin/activate
```

### Windows

```powershell
python -m venv venv
venv\Scripts\activate
```

---

## 3. Install dependencies

Dependencies are listed in `requirements.txt`.

Install them with:

```bash
pip install -r requirements.txt
```

---

# Environment Configuration

The LLM API key must be provided using the environment variable:

```
NEBIUS_API_KEY
```

Create a `.env` file in the project root:

```
NEBIUS_API_KEY=your_api_key_here
```

The application loads this automatically using `python-dotenv`.

⚠️ **Important**

Do **not** commit your `.env` file to version control.

Add this to `.gitignore`:

```
.env
```

You can optionally include a template file:

```
.env.example
```

Example:

```
NEBIUS_API_KEY=your_api_key_here
```

---

# Running the API

Start the FastAPI server using **uvicorn**.

```bash
uvicorn main:app --reload
```

You should see output similar to:

```
Uvicorn running on http://127.0.0.1:8000
```

---

# API Documentation

Once the server is running, open the interactive API documentation:

```
http://127.0.0.1:8000/docs
```

This provides a Swagger UI interface for testing the API.

---

# Testing the API

## Using curl (Linux / macOS)

```bash
curl -X POST "http://127.0.0.1:8000/summarize" \
     -H "Content-Type: application/json" \
     -d '{"github_url": "https://github.com/psf/requests"}'
```

---

## Using PowerShell (Windows)

PowerShell's `curl` command is an alias for `Invoke-WebRequest`, so the syntax differs.

Use:

```powershell
Invoke-RestMethod `
  -Method POST `
  -Uri "http://127.0.0.1:8000/summarize" `
  -ContentType "application/json" `
  -Body '{"github_url": "https://github.com/psf/requests"}'
```

---

# Example Request

```
POST /summarize
```

```json
{
  "github_url": "https://github.com/psf/requests"
}
```

---

# Example Response

```json
{
  "summary": "The Requests library is a Python HTTP client designed to simplify sending HTTP requests.",
  "technologies": ["Python"],
  "structure": "The repository contains the main requests package, documentation, tests, and packaging configuration."
}
```

---

# Notes on Repository Analysis

To keep analysis efficient and focused, the service ignores large dependency or build directories such as:

```
node_modules
venv
__pycache__
build
dist
target
coverage
.git
```

The service prioritises high-signal files such as:

```
README.md
requirements.txt
pyproject.toml
package.json
Cargo.toml
go.mod
Dockerfile
```

These files typically provide the most useful context about the repository.

---

# LLM Integration

The project uses the Nebius Token Factory API with an OpenAI-compatible client.

Model used:

```
deepseek-ai/DeepSeek-R1-0528
```

The LLM receives structured repository context and returns:

- SUMMARY
- TECHNOLOGIES
- STRUCTURE

The API parses this response and returns the structured result.

---

# Logging

The application uses Python's `logging` module to report major execution steps such as:

- URL validation
- repository download
- repository analysis
- LLM request
- response parsing

Logs include timestamps and severity levels to help debugging.

---

# Error Handling

The API returns HTTP 400 responses when errors occur, such as:

- invalid GitHub URLs
- inaccessible repositories
- download failures
- LLM request failures

---

# Temporary File Handling

Repositories are downloaded and extracted into a temporary working directory created with `tempfile`.

This directory is automatically removed after each request.

---

# Summary

This project demonstrates:

- API development with **FastAPI**
- repository analysis using **Python filesystem tools**
- structured prompt building for **LLM summarisation**
- integration with the **Nebius Token Factory API**
- robust error handling and logging