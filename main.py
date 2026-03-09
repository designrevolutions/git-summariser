#!/usr/bin/env python3
"""
Main FastAPI application for the Nebius assignment project.

This is the first skeleton version of the service.
For now, the /summarize endpoint is just a placeholder so that
I can confirm the API server starts amd accepts requests correctly.

Run:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/docs
        
"""

# from typing import List# I decided not to use this, using the newer syntax instead

from fastapi import FastAPI
from pydantic import BaseModel, Field # Pydantic automatically validates incoming data. If the data is invalid, FastAPI returns an error and the endpoint function is never executed.


class SummariseRequest(BaseModel):
    """
    Request model for the /summarise endpoint.

    Attributes:
        github_url:
            URL of the GitHub repository to analyse. This should be a valid URL pointing to a GitHub repository.

    Example:
        {
            "github_url": "https://github.com/psf/requests"
        }
    """
    
    github_url: str = Field(..., description="Public Github repository URL") # github_url must be a string - Pydantic reads from the type hint above and knows to enforce from that.

class SummariseResponse(BaseModel):
    """
    Successful response returned by the /summarise endpoint.

    Attributes:
        summary:
            Human-readable summary of what the project does.
        technologies:
            Main technologies, languages, and frameworks used in the project.
        structure:
            A brief overview of the project structure.
    """

    summary: str
    technologies: list[str] # This is where you would have used the Typing class and ised List[str] (note the capital L)
    structure: str

# Create the FastAPI app instance - the main entry point for the API server. The parameters passed to FastAPI() are used for automatic API documentation generation (Swagger UI).
app = FastAPI(
    title="GitHub Repository Summariser API",
    description="API to summarise the contents of a GitHub repository.",
    version="0.1.0",
    openapi_tags=[ # I added this after reading up on FastAPI's documentation about API tags. This allows us to group related endpoints together in the API docs.
        {
            "name": "Repository Analysis",
            "description": "Operations related to analysing GitHub repositories."
        }
    ]
)


@app.get("/") # No line spaces below the decorator, otherwise FastAPI won't recognize this function as an endpoint handler.
def root() -> dict[str, str]:
    """
    Root endpoint to confirm the API is running.

    Returns:
        A simple message confirming the API is up.

    Example response:
        {
            "message": "Repository Summariser API is running"
        }
    """

    return {"message": "Hello, Nebius assignment project! The API is running."}

# @app.post("/summarise", response_model=SummariseResponse, tags=["Repository Analysis"])
@app.post("/summarise", response_model=SummariseResponse)
def summarise_repository(request: SummariseRequest) -> SummariseResponse:
    """
    Placeholder implementatio of the repository summarisation endpoint.

    For now, this does not yet:
    - vaidate the Github URL
    - fetch the repository contents
    - analyse the repository to generate a summary, list of technologies, and project structure
    - call the magic sauce 'LLM' to generate the summary

    Instead, it simply returns a placeholder response so that we can verify:
    - request parsing works
    - rsponse formatting works
    - the endpoint is reachable and accepts POST requests

    Args:
        request: 
            The incoming request body containing the GitHub URL to summarise.
    
    Returns:
        A placeholder response with a dummy summary, list of technologies, and project structure.
        
    Example request body:
        {
            "github_url": "https://github.com/psf/requests"
        }
    """

    return SummariseResponse(
        summary=(
            f"Place holder summary for repository at {request.github_url}. "
            "Real repository analysis will be added in the next steps."
        ),
        technologies=["Placeholder"],
        structure="Placeholder project structure description."
    )
