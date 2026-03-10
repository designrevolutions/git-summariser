#!/usr/bin/env python3
"""
Main FastAPI application for the Nebius assignment project.

At this stage, the /summarize endpoint:
- accepts a GitHub repository URL
- validates that it looks like a public GitHub repository URL
- extracts the repository owner and name
- downloads the repository ZIP archive
- extracts the repository to a temporary folder
- returns a placeholder response using real extracted repository information

Run:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/docs
"""

from pathlib import Path # This is a convenient way to work with file paths in a platform-independent way. It provides an object-oriented interface for handling filesystem paths, making it easier to read and write code that works across different operating systems.
import shutil # This is a Python module that provides a higher-level interface for file operations, such as copying and removing files and directories. We will use it to extract ZIP files and clean up temporary directories.
import tempfile # This module provides a way to create temporary files and directories. We will use it to create a temporary directory for extracting the downloaded repository ZIP file, which allows us to avoid cluttering the filesystem with intermediate files and ensures that they are cleaned up automatically when we're done.
import zipfile # This module provides tools for working with ZIP archives. We will use it to extract the contents of the downloaded repository ZIP file so that we can analyze the files within the repository.
import requests # This is a popular third-party library for making HTTP requests in Python. We will use it to download the ZIP archive of the GitHub repository. It provides a simple and intuitive API for sending HTTP requests and handling responses, making it easier to interact with web resources compared to using the built-in urllib library.

from urllib.parse import urlparse # This allows us to take a URL and split into different parts that we can dissect - we'd have to do this manually otherwise.

from fastapi import FastAPI, HTTPException # HTTPException is a special exception class provided by FastAPI that allows us to return HTTP error responses with specific status codes and messages when something goes wrong in our endpoint handlers.
from pydantic import BaseModel, Field # Pydantic automatically validates incoming data. If the data is invalid, FastAPI returns an error and the endpoint function is never executed.


class SummarizeRequest(BaseModel): # We use this for making requests to the /summarize endpoint. It defines the expected structure of the request body and allows FastAPI to automatically validate incoming requests against this model. If a request does not conform to this model, FastAPI will return a 422 Unprocessable Entity error with details about what was wrong with the request.
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

class SummarizeResponse(BaseModel): # We use this for receiving responses from the /summarize endpoint. It defines the expected structure of the response body and allows FastAPI to automatically generate API documentation that shows what the response will look like. It also provides type hints for the endpoint function, making it clear what kind of data is being returned.
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
    version="0.3.0", # I've lost track of how many times I've updated the version number already! I don't think I change the ver number before.
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


def build_github_archive_urls_for_zip_download(owner: str, repository_name: str) -> list[str]:
    """
    Build the URLs for downloading the GitHub repository archive.
    Even though we are given the repo URL, we can't be sure what the zip download URL is because it depends on the default branch of the repository. The default branch could be 'main', 'master', or something else. GitHub provides a standard URL format for downloading the repository as a ZIP file, which is:
    https://github.com/{owner}/{repository_name}/archive/refs/heads/{default_branch}.zip
    
    So, in the code: we're going to guess the default branch is 'main' or 'master' or one of the other defaults.

    EDIT: we can actually call the GitHub API to get the default branch name, so we don't have to guess! 
    TODO: This is what I will implement in the next step.

    Args:
        owner:
            The owner of the GitHub repository.
        repository_name:
            The name of the GitHub repository.

    Returns:
        A list of potential URLs to download the repository archive.
    """

    return [
        f"https://github.com/{owner}/{repository_name}/archive/refs/heads/main.zip",
        f"https://github.com/{owner}/{repository_name}/archive/refs/heads/master.zip"
    ]


def download_github_repository_as_zip(
        owner: str, 
        repository_name: str,
        working_directory: Path,
) -> Path:
    """
    Download the GitHub repository as a ZIP file into the supplied working directory.

    This function tries to download the repository archive using potential default branch names until it succeeds or exhausts all options.

    Args:
        owner:
            The owner of the GitHub repository.
        repository_name:
            The name of the GitHub repository.
        working_directory:
            The directory where the downloaded ZIP file will be saved.
            
    """
    archive_urls = build_github_archive_urls_for_zip_download(owner, repository_name)

    # zip_file_path = working_directory / "repository_name.zip" 
    zip_file_path = working_directory / f"{repository_name}.zip"
    
    for archive_url in archive_urls:
        try:
            response = requests.get(archive_url, timeout=30, stream=True) # We set stream=True to download the file in chunks, which is more memory efficient for large files. The timeout parameter is set to 30 seconds to prevent hanging indefinitely if the server does not respond.
        except requests.RequestException as error:
            raise ValueError(
                f"Failed to download Github repository archive from {archive_url}: {error}"
            ) from error
        
        # This is the standard method of using Python requests to download a file from a URL and save it to disk. We check if the response status code is 200 (OK) before writing the content to a file. We write the content in chunks to avoid loading the entire file into memory at once, which is important for large repositories.
        if response.status_code == 200:
            with open(zip_file_path, "wb") as zip_file:
                for chunk in response.iter_content(chunk_size=8192): # We iterate over the response content in chunks of 8192 bytes (8 KB) to efficiently write the file to disk without consuming too much memory.
                    zip_file.write(chunk)

            return zip_file_path
        
        if response.status_code == 404:
            raise ValueError(
                f"Github returned status {response.status_code} for URL {archive_url} while trying to download the repository archive."
            )

    raise ValueError(
        "Could not download repository archive. "
        "Please check the repository URL and try again."
    )




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
