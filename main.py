#!/usr/bin/env python3
"""
Main FastAPI application for the Nebius assignment project.

this is the second skeleton verion of the service.

At this stage, the /summarize endpoint:
- accepts a Github repo URL
- validates that it looks like a public Github repo URL
- extract the repo owner and name
- returns a placeholder response

Run:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/docs
        
"""

from urllib import parse
from urllib.parse import urlparse # This allows us to take a URL and split into different parts that we can dissect - we'd have to do this manually otherwise.

from fastapi import FastAPI, HTTPException # HTTPException is a special exception class provided by FastAPI that allows us to return HTTP error responses with specific status codes and messages when something goes wrong in our endpoint handlers.
from pydantic import BaseModel, Field # Pydantic automatically validates incoming data. If the data is invalid, FastAPI returns an error and the endpoint function is never executed.


class SummarizeRequest(BaseModel):
    """
    Request model for the /summarize endpoint.

    Attributes:
        github_url:
            URL of the GitHub repository to analyse. This should be a valid URL pointing to a GitHub repository.

    Example:
        {
            "github_url": "https://github.com/psf/requests"
        }
    """
    
    github_url: str = Field(..., description="Public Github repository URL") # github_url must be a string - Pydantic reads from the type hint above and knows to enforce from that.

class SummarizeResponse(BaseModel):
    """
    Successful response returned by the /summarize endpoint.

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
    title="GitHub Repository Summarizer API",
    description="API to summarize the contents of a GitHub repository.",
    version="0.1.0",
    openapi_tags=[ # I added this after reading up on FastAPI's documentation about API tags. This allows us to group related endpoints together in the API docs.
        {
            "name": "Repository Analysis",
            "description": "Operations related to analysing GitHub repositories."
        }
    ],
)


def parse_github_repository_url(githurb_url: str) -> tuple[str, str]:
    """
    Validate and parse a Github repository URL.

    This function checks that:
    - the URL uses http or https 
    - the hostname is github.com
    - the path contains at least two segments (owner and repo name)

    Supported examples:
        https://github.com/psf/requests
        http://github.com/psf/requests
        https://github.com/psf/requests/
        https://github.com/psf/requests.git

    Args:
        github_url: 
            The URL of the GitHub repository given by the user.
    
    Returns:
        A tuple containing:
            - repository owner
            - repository name

    Raises:
        ValueError: 
            If the URL is invalid or does not point to a GitHub repository URL.
    """

    parsed_url = urlparse(githurb_url) # We now have an object with all the different parts - all done in one line thanks to the urlparse function!

    # Ensure the scheme is http or https
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")
    
    # Ensure the hostname is github.com
    # if parsed_url.netloc.lower() != "github.com": #netloc and hostname are almost identical, but netloc can include the port number if it's specified in the URL, whereas hostname is just the domain name. In our case, we expect users to input standard GitHub URLs without custom ports, so either would work. I went with hostname for clarity.
    if parsed_url.hostname != "github.com":
        raise ValueError("URL must point to github.com")
    
    # Split the path and remove empty segments
    path_parts = [part for part in parsed_url.path.split("/") if part] # This is a common way to split a URL path into its components while filtering out any empty segments that may occur due to leading/trailing slashes.
    # We need to have the 'if part' in the list comprehension to filter out any empty strings that may result from leading or trailing slashes in the URL path. For example, if the path is "/psf/requests/", splitting by "/" would give us ["", "psf", "requests", ""]. The list comprehension filters out the empty strings, leaving us with ["psf", "requests"].

    # A valid repositor URL should at least look like:
    # /owner/repo
    if len(path_parts) < 2:
        raise ValueError("URL path must contain at least owner and repository name")
    
    owner, repository_name = path_parts[0], path_parts[1]

    # Handle URLs that end with .git (e.g. https://github.com/psf/requests.git)
    if repository_name.endswith(".git"):
        repository_name = repository_name[:-4]
    
    if not owner or not repository_name:
        raise ValueError("Owner and repository name could not be determined") # This is a catch-all validation in case the previous checks missed something. For example, if the URL was "https://github.com/psf/.git", the previous checks would pass but we would end up with an empty repository name after stripping ".git". This check ensures that we have valid values for both owner and repository name. It's for a very edge case - keeping in to make complete.

    return owner, repository_name


@app.get("/") # No line spaces below the decorator, otherwise FastAPI won't recognize this function as an endpoint handler.
def root() -> dict[str, str]:
    """
    Root endpoint to confirm the API is running.

    Returns:
        A simple message confirming the API is up.

    Example response:
        {
            "message": "Repository Summarizer API is running"
        }
    """

    return {"message": "Hello, Nebius assignment project! The API is running."}

# @app.post("/summarize", response_model=SummarizeResponse, tags=["Repository Analysis"])
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_repository(request: SummarizeRequest) -> SummarizeResponse:
    """
    Validate the GitHub URL and return a placeholder summary response.

    At this stage, this endpoint does not yet:
    - download the repository
    - analyse file contents
    - call an LLM

    It does:
    - validate the GitHub URL
    - extract the repository owner and name
    - return a more meaningful placeholder response

    Args:
        request:
            The incoming request body containing the GitHub repository URL.

    Returns:
        A placeholder response containing parsed repository information.

    Raises:
        HTTPException:
            If the GitHub URL is invalid.
    """


    try:
        owner, repository_name = parse_github_repository_url(request.github_url)
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error
        

    return SummarizeResponse(
        summary=(
            f"Place holder summary for repository at {request.github_url}. "
            "Real repository analysis will be added in the next steps."
        ),
        technologies=["Placeholder"],
        structure="Placeholder project structure description."
    )
