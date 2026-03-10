#!/usr/bin/env python3
"""
Main FastAPI application for the Nebius assignment project.

This version of the service can now download a public GitHub repository,
extract it locally, and perform a basic structural analysis of the project.

The /summarize endpoint currently performs the following steps:

1. Validate the provided GitHub repository URL.
2. Extract the repository owner and name from the URL.
3. Verify that the repository exists on GitHub.
4. Download the repository as a ZIP archive.
5. Extract the repository into a temporary working directory.
6. Traverse the repository while ignoring large dependency directories.
7. Collect basic structural information such as:
   - total file count
   - file extensions used
   - directory names present in the project

This structural metadata will later be used as input to an LLM to generate
a human-readable repository summary.

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
import os


IGNORED_DIRECTORIES = {
    "node_modules",
    ".venv",
    "venv",
    "__pycache__",
    "build",
    "dist",
    "target",
    "vendor",
    "coverage",
    ".next",
    ".nuxt",
    ".git",
}


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


def parse_github_repository_url(github_url: str) -> tuple[str, str]:
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

    parsed_url = urlparse(github_url) # We now have an object with all the different parts - all done in one line thanks to the urlparse function!

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

    # ####==== Start of hack
    # I added this as the code would only fail after tryingt o download bad zips for 30 seconds each time
    # This isn't the best way to check if a repository exists, but it allows us to fail fast if the repository doesn't exist or isn't accessible, without having to wait for multiple failed download attempts. In the next step, we will implement a more robust method of checking repository existence and getting the default branch name using the GitHub API, which will eliminate the need for this hack.
    # I'll fix with using API later
    # Not sure if the code is working! # TODO: check later
    try:
        response = requests.head( # Only get headers, not the full content, to check if the repository exists and is accessible. This is a much faster way to check if the repository exists compared to trying to download the ZIP file, as it doesn't require downloading any content.
            f"https://github.com/{owner}/{repository_name}",
            timeout=10
        )
    except requests.RequestException as error:
        raise ValueError(
            f"Failed to connect to GitHub to verify repository existence: {error}"
        ) from error

    if response.status_code >= 400:
        raise ValueError(
            f"Repository '{owner}/{repository_name}' could not be accessed "
            f"(HTTP status {response.status_code})."
        )
    # ####==== End of hack

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
            # raise ValueError(
            #     f"Github returned status {response.status_code} for URL {archive_url} while trying to download the repository archive."
            # )
            continue # If we get a 404, it likely means that the default branch name in that URL is incorrect (e.g. 'main' vs 'master'), so we can just try the next URL in the list without raising an error immediately. We only raise an error if all options are exhausted and we still haven't successfully downloaded the archive.

    raise ValueError(
        "Could not download repository archive. "
        "Please check the repository URL and try again."
    )


def extract_zip_file(zip_file_path: Path, extract_to_directory: Path) -> Path:
    """
    Extract a ZIP archive and return the extracted repository root folder.

    GitHub archives usually unpack into a single top-level folder such as:
        repository-name-main/

    Args:
        zip_file_path:
            Path to the downloaded ZIP archive.
        extract_to_directory:
            Directory where the ZIP should be extracted.

    Returns:
        Path to the extracted repository root folder.

    Raises:
        ValueError:
            If extraction fails or no extracted repository folder is found.
    """

    try:
        with zipfile.ZipFile(zip_file_path, "r") as zip_file: # This is the standard way to open a ZIP file in Python using the zipfile module. We use a context manager (the with statement) to ensure that the file is properly closed after we're done with it, even if an error occurs during extraction.
            zip_file.extractall(extract_to_directory)
    except zipfile.BadZipFile as error:
        raise ValueError(f"Failed to extract ZIP file {zip_file_path}: {error} - most likely it's not a valid zip archive, else try downloading again.") from error
    
    extract_to_directories = [
        path for path in extract_to_directory.iterdir() if path.is_dir()
    ]

    if not extract_to_directories: # This is a sanity check to ensure that we actually extracted something and that there is a directory in the extracted folder. If the ZIP file was valid but did not contain any directories, this would catch that case.
        raise ValueError(f"No repository folder found in {extract_to_directory}")

    if len(extract_to_directories) == 1:
        return extract_to_directories[0] # This is a sanity check to ensure that we only have one top-level directory in the extracted folder, which is what we expect from GitHub archives. If there are multiple directories, it could indicate that the ZIP file is not structured as expected.

    # Fallback: choose the first directory in sorted order if multiple appear.
    # In normal GitHub ZIP archives this should rarely be needed, but it keeps
    # the function robust.
    return sorted(extract_to_directories)[0]


def download_and_extract_repository(owner: str, repository_name: str) -> tuple[Path, Path]:
    """
    Download and extract a GitHub repository into a temporary directory.

    Args:
        owner:
            GitHub repository owner.
        repository_name:
            GitHub repository name.

    Returns:
        A tuple containing:
            - path to the temporary working directory
            - path to the extracted repository root directory

    Raises:
        ValueError:
            If download or extraction fails.
    """

    # Create a temporary directory to work in. This ensures that we don't clutter the filesystem with intermediate files and that everything is cleaned up automatically when we're done.
    temporary_directory = Path(tempfile.mkdtemp(prefix="repo_summarizer_"))
    print(f"Temporary directory created at: {temporary_directory}") # I just wanted to check the location of where the temporary directory is being created, to make sure it's working as expected. This is especially useful for debugging purposes, as it allows us to inspect the contents of the temporary directory if something goes wrong during the download or extraction process.

    try:
        zip_file_path = download_github_repository_as_zip(
            owner=owner, 
            repository_name=repository_name, 
            working_directory=temporary_directory,
        )

        extracted_repository_path = extract_zip_file(
            zip_file_path=zip_file_path, 
            extract_to_directory=temporary_directory,
            )

        return temporary_directory, extracted_repository_path
    
    except Exception:
        # Clean up the temporary directory if anything goes wrong during download or extraction
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise # This is the standard way to re-raise the original exception after performing cleanup. The 'raise' statement without any arguments will re-raise the last exception that was active in the current scope, allowing us to preserve the original error message and stack trace while ensuring that we clean up any temporary resources we created before the error occurred.


def analyse_repository_structure(repository_path: Path) -> dict[str, str]:
    """
    Traverse the extracted repository and collect basic structural information.

    This function walks the repository directory tree while ignoring large
    dependency folders that would distort analysis.

    Args:
        repository_path:
            Path to the extracted repository root directory.

    Returns:
        Dictionary containing repository metadata useful for summarisation.
    """

    file_count = 0
    extensions = set()
    directories_seen = set()

    for root, dirs, files in os.walk(repository_path):

        # Remove ignored directories so os.walk never enters them
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES] # [:] is a special syntax I've never come across before! It allows us to modify the 'dirs' list in place, which is necessary for os.walk to recognize the changes and skip the ignored directories during traversal.

        for file in files:
            file_count += 1
            suffix = Path(file).suffix.lower() # This is a convenient way to get the file extension from a filename. The suffix property of a Path object returns the file extension, including the leading dot (e.g., '.py' for Python files). We convert it to lowercase to ensure that we treat files with the same extension but different cases (e.g., '.PY' vs '.py') as the same type of file.
            if suffix:
                extensions.add(suffix) # extensions is a set, so it will automatically handle duplicates and only keep unique file extensions.

        directories_seen.add(Path(root).name) # This adds the name of the current directory to the set of directories seen. We use Path(root).name to get just the name of the directory without the full path, which allows us to easily check against our IGNORED_DIRECTORIES set in future iterations.

    return {
        "file_count": file_count,
        "extensions": sorted(extensions),
        "directories": sorted(directories_seen)
    }


# All the action happens below - we've declared fns and classes - now we put everything together.

@app.get("/") # No line spaces below the decorator, otherwise FastAPI won't recognize this function as an endpoint handler.
def root() -> dict[str, str]:
    """
    Root endpoint to confirm the API is running.

    Returns:
        A simple message confirming the API is up.

    Example response:
        {
            "message": "Hello, Nebius assignment project! The API is running."
        }
    """
    print("Root endpoint was called")# I needed this because I was having problems - the server started but the browser showed nothing - same with Curl
    return {"message": "Hello, Nebius assignment project! The API is running."}

# @app.post("/summarize", response_model=SummarizeResponse, tags=["Repository Analysis"])
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_repository(request: SummarizeRequest) -> SummarizeResponse:
    """
    Validate the GitHub URL, download the repository archive, and extract it.

    At this stage, this endpoint still does not yet:
    - traverse repository files
    - analyse project structure
    - detect technologies
    - call an LLM

    It now does:
    - validate the GitHub URL
    - extract the repository owner and name
    - download the repository ZIP archive
    - extract the repository locally
    """

    try:
        owner, repository_name = parse_github_repository_url(request.github_url)

        temporary_directory, extracted_repository_path = download_and_extract_repository(
            owner = owner,
            repository_name = repository_name,
        )
    except ValueError as error:
        raise HTTPException(status_code=400, detail=str(error)) from error

    try:
        # Count top-level items purely as a quick proof that extraction worked.
        # This is not a real analysis of the repository, just a placeholder to
        # show that we can work with the extracted files. In the next steps,
        # we will implement real analysis of the repository structure and contents.
        analysis = analyse_repository_structure(extracted_repository_path)
    finally:
        # Clean up the temporary directory whether the above succeeds or fails.
        # At this stage, we only need the repository long enough to prove that
        # download and extraction worked.
        shutil.rmtree(temporary_directory, ignore_errors=True)

    return SummarizeResponse(
        summary=(
            f"Repository '{owner}/{repository_name}' was downloaded and extracted successfully."
        ),
        technologies=["We'll get to that later"],
        structure=(
            f"Total files analysed: {analysis['file_count']}. "
            f"Detected files extensions: {', '.join(analysis['extensions'][:10])}. " # We limit to the first 10 extensions for brevity, as some repositories may have a large number of different file types. This is just a placeholder to show that we can analyze the repository contents and extract useful information from it. In the next steps, we will implement more detailed analysis of the repository structure and contents to generate a meaningful summary.
            f"Directories found: {', '.join(analysis['directories'][:10])}. "
        ),
    )
