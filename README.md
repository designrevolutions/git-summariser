# AI GitHub Repository Summarizer API

This project implements a FastAPI service that analyses a public GitHub repository and generates a structured summary using a Nebius-hosted large language model (LLM).

The system is designed to perform lightweight static repository analysis locally and delegate higher-level reasoning to an LLM, keeping the architecture simple, efficient, and easily extensible.

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

# Project Structure

```
.
├── main.py
├── requirements.txt
├── README.md
├── .env.example
```

---

# Requirements

- Python **3.10+**
- A **Nebius API key**

---

# Installation

## 1 Clone the repository

```
git clone <repository-url>
cd <repository-folder>
```

---

## 2 Create a virtual environment

### Linux / macOS

```
python -m venv venv
source venv/bin/activate
```

### Windows

```
python -m venv venv
venv\Scripts\activate
```

---

## 3 Install dependencies

Dependencies are defined in **requirements.txt**.

```
pip install -r requirements.txt
```

---

# Environment Configuration

The API key must be provided through an environment variable.

```
NEBIUS_API_KEY
```

Create a `.env` file in the project root:

```
NEBIUS_API_KEY=your_api_key_here
```

The application loads this automatically using **python-dotenv**.

⚠️ Do **not commit** your `.env` file.

Add to `.gitignore`:

```
.env
```

Provide an example file:

`.env.example`

```
NEBIUS_API_KEY=your_api_key_here
```

---

# Running the API

Start the FastAPI server:

```
uvicorn main:app --reload
```

The API runs on:

```
http://127.0.0.1:8000
```

---

# API Documentation

FastAPI automatically provides interactive documentation.

Open:

```
http://127.0.0.1:8000/docs
```

---

# Testing the API

## Linux / macOS

```
curl -X POST "http://127.0.0.1:8000/summarize" \
-H "Content-Type: application/json" \
-d '{"github_url": "https://github.com/psf/requests"}'
```

---

## Windows PowerShell

```
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

```
{
  "github_url": "https://github.com/psf/requests"
}
```

---

# Example Response

```
{
  "summary": "The Requests library is a Python HTTP client designed to simplify sending HTTP requests.",
  "technologies": ["Python"],
  "structure": "The repository contains the main requests package, documentation, tests, and packaging configuration."
}
```

---

# Repository Analysis Notes

To keep analysis efficient the following directories are ignored:

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

High-signal files used for analysis include:

```
README.md
requirements.txt
pyproject.toml
package.json
Cargo.toml
go.mod
Dockerfile
```

---

# LLM Integration

The project uses the **Nebius Token Factory API** with an OpenAI-compatible client.

Default model:

```
deepseek-ai/DeepSeek-R1-0528
```

---

# Model Availability

Models available through Nebius may change over time.

The model used by the system is defined in a single constant:

```
AI_MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528"
```

If a model becomes unavailable, it can be replaced easily without changing the rest of the code.

---

# Logging

The application uses Python's **logging module** to record:

- repository download
- repository analysis
- LLM requests
- responses and errors

---

# Error Handling

The API returns HTTP 400 responses for issues such as:

- invalid GitHub URLs
- inaccessible repositories
- download failures
- LLM request errors

---

# Temporary File Handling

Repositories are downloaded and extracted into temporary directories using `tempfile`.

These directories are automatically removed after processing.

---

# Summary

This project demonstrates:

- FastAPI service development
- repository analysis with Python
- structured LLM prompting
- integration with Nebius Token Factory
- robust logging and error handling
