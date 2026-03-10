#!/usr/bin/env python3
"""
Main FastAPI application for the Nebius assignment project.

This version of the service performs basic analysis of a public GitHub
repository and prepares structured context for later LLM summarisation.

The /summarize endpoint currently performs the following steps:

1. Validate the provided GitHub repository URL.
2. Extract the repository owner and name.
3. Verify that the repository exists on GitHub.
4. Download the repository as a ZIP archive.
5. Extract the repository into a temporary working directory.
6. Traverse the repository structure while ignoring large dependency
   directories (e.g. node_modules, .venv, __pycache__).
7. Collect structural metadata including:
   - file count
   - file extensions present
   - directory names
   - important project files
8. Infer likely technologies based on file extensions and key project files.
9. Identify candidate files whose contents may later be passed to the LLM
   (e.g. README.md, dependency files, configuration files).
10. Apply a simple heuristic to adjust analysis behaviour for very large
    repositories.

The collected metadata will later be used to generate a human-readable
summary of the repository using an LLM.

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


IGNORED_DIRECTORIES = { # These are common directories that contain large amounts of files that are not part of the actual project code, but rather dependencies or build artifacts. We want to ignore these directories when analyzing the repository structure to avoid skewing our analysis with irrelevant files.
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

IMPORTANT_FILENAMES = { # These are common filenames that can provide important information about the project, such as its dependencies, build configuration, or documentation. We will use the presence of these files as signals when analyzing the repository structure to help identify the technologies and frameworks used in the project.
    "readme.md",
    "requirements.txt",
    "pyproject.toml",
    "setup.py",
    "setup.cfg",
    "pipfile",
    "package.json",
    "package-lock.json",
    "yarn.lock",
    "pnpm-lock.yaml",
    "cargo.toml",
    "cargo.lock",
    "go.mod",
    "go.sum",
    "pom.xml",
    "build.gradle",
    "build.gradle.kts",
    "settings.gradle",
    "dockerfile",
    "docker-compose.yml",
    "docker-compose.yaml",
    "makefile",
    "cmakelists.txt",
    "composer.json",
    "composer.lock",
    "gemfile",
    "gemfile.lock",
}

EXTENSION_TO_TECHNOLOGY = { # This is a mapping of common file extensions to the programming languages or technologies they are associated with. We will use this mapping to help identify the main technologies used in the project based on the file extensions present in the repository. This is a simple heuristic and may not be 100% accurate, but it can provide useful signals for our analysis.
    ".py": "Python",
    ".js": "JavaScript",
    ".ts": "TypeScript",
    ".tsx": "TypeScript",
    ".jsx": "JavaScript",
    ".java": "Java",
    ".kt": "Kotlin",
    ".rs": "Rust",
    ".go": "Go",
    ".php": "PHP",
    ".rb": "Ruby",
    ".cpp": "C++",
    ".cc": "C++",
    ".cxx": "C++",
    ".c": "C",
    ".cs": "C#",
    ".swift": "Swift",
    ".scala": "Scala",
    ".sh": "Shell",
    ".sql": "SQL",
    ".html": "HTML",
    ".css": "CSS",
}

FILE_TECHNOLOGY_MAP: dict[str, set[str]] = {
    "Python": {"requirements.txt", "pyproject.toml", "setup.py", "Pipfile"},
    "Node.js": {"package.json"},
    "Docker": {"dockerfile", "docker-compose.yml"},
    "Rust": {"cargo.toml"},
    "Go": {"go.mod"},
    "Java": {"pom.xml", "build.gradle"},
    "PHP": {"composer.json"},
    "Ruby": {"gemfile"},
    "CMake": {"cmakelists.txt"},
    "Make": {"makefile"},
}

FILE_PRIORITY_ORDER: dict[str, int] = {
    "readme.md": 0,
    "pyproject.toml": 1,
    "requirements.txt": 2,
    "setup.py": 3,
    "setup.cfg": 4,
    "pipfile": 5,
    "package.json": 6,
    "cargo.toml": 7,
    "go.mod": 8,
    "pom.xml": 9,
    "build.gradle": 10,
    "dockerfile": 11,
    "docker-compose.yml": 12,
    "makefile": 13,
    "cmakelists.txt": 14,
}


MAX_FILES_FOR_BROAD_ANALYSIS = 2000 # This is a threshold for the maximum number of files we will analyze in detail when traversing the repository. If a repository contains more than this number of files, we will switch to a more high-level analysis approach that focuses on key files and directories rather than analyzing every single file. This is to ensure that our analysis remains efficient and does not get bogged down by very large repositories with thousands of files, which could lead to long processing times and increased resource usage.

MAX_CANDIDATE_FILES_TO_READ = 12 # Candidate files are the important files we identify during repository traversal that we want to read and include as context for the LLM when generating the summary. This is a limit on how many of those candidate files we will actually read and include in the LLM input, to ensure that we stay within reasonable token limits for the LLM and avoid overwhelming it with too much information. We will prioritize which files to include based on their importance (e.g., README.md would be more important than a less informative file), but this is just a simple limit to keep our LLM input manageable.
MAX_CHARACTERS_PER_FILE = 4000
MAX_TOTAL_CONTEXT_CHARACTERS = 20000 # This is a limit on the total number of characters we will include from all candidate files combined when providing context to the LLM. This is to ensure that we stay within the token limits of the LLM and provide it with a manageable amount of information to work with when generating the repository summary. We will need to implement logic to prioritize which files and which parts of those files to include in the context based on their importance and relevance, while ensuring that we do not exceed this total character limit.
# We need these max limits to limit the amount we send to the LLM, otherwise we might exceed the token limits of the model and get errors, or we might just overwhelm the model with too much information, which could lead to less coherent summaries. By setting these limits, we can ensure that we provide the LLM with a focused and relevant set of information about the repository while staying within the constraints of the model's input capabilities.

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


def infer_technologies_from_files(
        extensions: set[str], 
        important_files: set[str]
) -> list[str]:
    """
    Infer likely technologies used in the repository based on file extensions
    and important configuration/dependency files.

    Args:
        extensions:
            Set of file extensions found in the repository.
        important_files:
            Set of important filenames found in the repository.

    Returns:
        Sorted list of inferred technologies.
    """

    detected_technologies = set()

    for extension in extensions: # In the part, we see if we can find a match of our extensions to the technologies in our EXTENSION_TO_TECHNOLOGY mapping. This is a simple heuristic that can give us some signals about the main programming languages used in the project based on the file extensions present in the repository. For example, if we see a lot of .py files, it's a strong signal that the project uses Python. If we see .js and .jsx files, it's a strong signal that the project uses JavaScript and possibly React. This is not a perfect method, as some projects may use unconventional file extensions or have mixed technologies, but it can provide useful insights for our analysis.
        technology = EXTENSION_TO_TECHNOLOGY.get(extension)
        if technology:
            detected_technologies.add(technology)

    lower_important_files = {file_name.lower() for file_name in important_files} # Reminder: important_files is a set of filenames found in the repository. We convert them to lowercase to ensure that our checks are case-insensitive, as some repositories may have files with different cases (e.g., "README.md" vs "readme.md"). By normalizing the filenames to lowercase, we can reliably check for the presence of important files regardless of their case in the actual repository.

    # We can also add some heuristics based on important files. For example, if we see a 'package.json' file, it's a strong signal that the project uses Node.js, even if we didn't detect any .js files (e.g. if it's a monorepo with separate frontend/backend folders).
    for technology, markers in FILE_TECHNOLOGY_MAP.items():
        if lower_important_files & markers: # The use of & is checking for an intersection between the set of important files found in the repository and the set of marker files associated with a particular technology. If there is any overlap (i.e., if the intersection of the two sets is not empty), it indicates that at least one of the marker files for that technology is present in the repository, which is a strong signal that the technology is being used. This allows us to infer the presence of certain technologies based on key configuration or dependency files, even if we don't see a large number of source code files with specific extensions.
            detected_technologies.add(technology)

    return sorted(detected_technologies)


def analyse_repository_structure(repository_path: Path) -> dict:
    """
    Traverse the extracted repository and collect structural information.

    This function walks the repository directory tree while ignoring large
    dependency folders and build artefacts that would distort analysis.

    It also:
    - records important project files
    - infers likely technologies
    - applies a simple large-repository heuristic

    Args:
        repository_path:
            Path to the extracted repository root directory.

    Returns:
        Dictionary containing repository metadata useful for summarisation.
    """

    file_count = 0
    extensions_found = set()
    directories_seen = set()
    important_files_found = set()
    candidate_files_to_read = [] # This is where we will collect paths to important files that we want to read and include in the LLM input for summarization. For example, if we find a README.md file, we would add its path to this list so that we can read its contents later and provide it as context to the LLM when generating the repository summary.

    for root, dirs, files in os.walk(repository_path): # The os.walk returns 3 values: the current directory path (root), a list of subdirectories in the current directory (dirs), and a list of files in the current directory (files). We can use these values to traverse the entire directory tree of the repository and collect information about the files and directories present. The os.walk function is a convenient way to perform a recursive traversal of a directory structure, allowing us to easily analyze all files and subdirectories within the repository.

        # Remove ignored directories so os.walk never enters them
        dirs[:] = [d for d in dirs if d not in IGNORED_DIRECTORIES] # [:] is a special syntax I've never come across before! It allows us to modify the 'dirs' list in place, which is necessary for os.walk to recognize the changes and skip the ignored directories during traversal.

        directories_seen.add(Path(root).name) # This adds the name of the current directory to the set of directories seen. We use Path(root).name to get just the name of the directory without the full path, which allows us to easily check against our IGNORED_DIRECTORIES set in future iterations.

        for file_name in files:
            file_count += 1

            suffix = Path(file_name).suffix.lower() # This is a convenient way to get the file extension from a filename. The suffix property of a Path object returns the file extension, including the leading dot (e.g., '.py' for Python files). We convert it to lowercase to ensure that we treat files with the same extension but different cases (e.g., '.PY' vs '.py') as the same type of file.
            if suffix:
                extensions_found.add(suffix) # extensions is a set, so it will automatically handle duplicates and only keep unique file extensions.

            lower_file_name = file_name.lower() # We convert the filename to lowercase to ensure that our checks for important files are case-insensitive, as some repositories may have files with different cases (e.g., "README.md" vs "readme.md"). By normalizing the filenames to lowercase, we can reliably check for the presence of important files regardless of their case in the actual repository.  

            if lower_file_name in IMPORTANT_FILENAMES:
                important_files_found.add(lower_file_name) # This adds the lowercase filename to the set of important files found in the repository. We will use this set later to help infer the technologies used in the project based on the presence of key configuration or dependency files.

                full_file_path = Path(root) / file_name
                relative_file_path = full_file_path.relative_to(repository_path) # This converts the full file path to a relative path with respect to the repository root directory. This is useful for readability and for providing context to the LLM, as it allows us to show the file paths in a way that is relative to the structure of the repository rather than using absolute paths that may be less meaningful.
                
                # candidate_files_to_read.append(str(relative_file_path)) # This adds the full path to the important file to our list of candidate files to read for LLM input. We will later read the contents of these files and include them as context when generating the repository summary with the LLM, as they often contain valuable information about the project, such as its purpose (README.md), dependencies (requirements.txt, package.json), build configuration (pom.xml, build.gradle), and more.
                # We're getting \\ output, using the above - this is normal, it's how JSON treats backslashes in strings. However, it can be a bit messy to read in the output. By using as_posix(), we can convert the path to use forward slashes (/) instead of backslashes (\), which is more standard and easier to read, especially in the context of file paths. This way, the output will show paths like "src/main.py" instead of "src\\main.py", making it clearer and more consistent across different platforms.
                # So I looked up the documentation for pathlib and found that as_posix() is a method that returns the string representation of the path with forward slashes, regardless of the operating system. This is particularly useful when we want to ensure that our file paths are represented in a consistent way, especially when working with APIs or systems that expect POSIX-style paths. By using as_posix(), we can avoid issues with backslashes being treated as escape characters in JSON strings and make our output more readable and standardized.
                candidate_files_to_read.append(relative_file_path.as_posix()) # This adds the full path to the important file to our list of candidate files to read for LLM input. We will later read the contents of these files and include them as context when generating the repository summary with the LLM, as they often contain valuable information about the project, such as its purpose (README.md), dependencies (requirements.txt, package.json), build configuration (pom.xml, build.gradle), and more.

    inferred_technologies = infer_technologies_from_files(
        extensions=extensions_found, 
        important_files=important_files_found,
    )

    analysis_mode= "broad"

    if file_count > MAX_FILES_FOR_BROAD_ANALYSIS:
        analysis_mode = "selective"

    return {
        "file_count": file_count,
        "extensions": sorted(extensions_found),
        "directories": sorted(directories_seen),
        "important_files": sorted(important_files_found),
        "candidate_files_to_read": sorted(candidate_files_to_read),
        "technologies": inferred_technologies,
        "analysis_mode": analysis_mode,
    }


def choose_candidate_files_to_read(
    candidate_files: list[str]
) -> list[str]:
    """
    Choose the most useful candidate files to read for building repository context.

    Files are ranked so that highly informative files such as README.md and
    dependency/configuration files are preferred.

    Args:
        candidate_files:
            Relative file paths of candidate files found during repository analysis.

    Returns:
        A sorted and prioritised list of candidate file paths to read.
    """

    def sort_key(relative_path: str) -> tuple[int, str]:
        file_name = Path(relative_path).name.lower() # We extract the filename from the relative path and convert it to lowercase to ensure that our sorting is case-insensitive, as some repositories may have files with different cases (e.g., "README.md" vs "readme.md"). By normalizing the filenames to lowercase, we can reliably apply our priority sorting based on the presence of important files regardless of their case in the actual repository.

        priority = FILE_PRIORITY_ORDER.get(file_name, 100) # We look up the filename in our FILE_PRIORITY_ORDER mapping to get its priority value. If the filename is not found in the mapping, we assign it a default priority of 100, which means it will be sorted after all known important files. This allows us to prioritize certain key files (e.g., README.md, requirements.txt) that are likely to contain valuable information about the project when selecting which candidate files to read for LLM context.

        return priority, relative_path.lower() # We return a tuple containing the priority and the relative path. The sorting will first be based on priority (with lower values being higher priority), and then by relative path as a tiebreaker for files with the same priority.

    sorted_files: list[str] = sorted(candidate_files, key=sort_key) # We sort the candidate files using the sort_key function defined above, which prioritizes important files based on our FILE_PRIORITY_ORDER mapping. This ensures that when we select the top candidate files to read for LLM context, we are choosing the most informative files that are likely to provide valuable insights about the project.
    # OK, I had trouble getting my head around how this sorted worked - it seems as though this is JUST how it's done in Python - you define a sort key function that takes an item and returns a value (or tuple) that represents the sorting criteria for that item. Then, when you call sorted() with that key function, it will use the values returned by the key function to determine the order of the items in the sorted output. In our case, the sort_key function returns a tuple of (priority, relative_path), so the sorting will first be based on priority (with lower values being higher priority), and then by relative path as a tiebreaker for files with the same priority.
    # Just going with the definition for now and accepting this is how it works.

    return sorted_files[:MAX_CANDIDATE_FILES_TO_READ] # We return only the top candidate files up to our defined limit (MAX_CANDIDATE_FILES_TO_READ) to ensure that we stay within reasonable token limits for the LLM and provide it with a focused set of information about the repository. This allows us to include the most important files in the LLM context while avoiding overwhelming it with too much information, which could lead to less coherent summaries.

# We only will use this fn with the candidate files we identified as important during repository analysis, such as README.md, requirements.txt, package.json, etc. These files often contain valuable information about the project that can help the LLM generate a more accurate and informative summary. By reading the contents of these files and including them as context for the LLM, we can provide it with insights into the project's purpose, dependencies, configuration, and more, which can lead to better summarization results.
def read_repository_text_file( 
        repository_path: Path,
        relative_file_path: str,
        max_characters: int = MAX_CHARACTERS_PER_FILE,
) -> str:
    """
    Read a text file from the repository safely and return trimmed content.

    Args:
        repository_path:
            Path to the extracted repository root directory.
        relative_file_path:
            File path relative to the repository root.
        max_characters:
            Maximum number of characters to return.

    Returns:
        Trimmed file content as text.

    Raises:
        ValueError:
            If the file cannot be read as text.
    """
    
    full_file_path = repository_path / relative_file_path

    try:
        content = full_file_path.read_text(encoding="utf-8") # We attempt to read the file as UTF-8 encoded text, which is a common encoding for text files. If the file is not a valid text file or contains characters that cannot be decoded as UTF-8, this will raise a UnicodeDecodeError, which we catch and re-raise as a ValueError with a more descriptive message. This allows us to handle cases where we might encounter binary files or files with unsupported encodings gracefully, without crashing the entire analysis process.
    except UnicodeDecodeError:
        raise ValueError(
            f"File {relative_file_path} could not be read as text."
        )
    except OSError as error:
        raise ValueError(
            f"An error occurred while reading file {relative_file_path}: {error}"
        ) from error

    return content[:max_characters] # We return only the first max_characters of the file content to ensure that we stay within reasonable token limits for the LLM when providing context. This allows us to include a portion of the file's content that is likely to contain valuable information about the project while avoiding overwhelming the LLM with too much text, which could lead to less coherent summaries. The max_characters limit can be adjusted based on the expected size of important files and the token limits of the LLM we are using.


def build_repository_context(
    repository_path: Path,
    analysis: dict,
) -> str:
    """
    Build a structured text context describing the repository.

    This context is designed to be passed to an LLM in the next step.

    Args:
        repository_path:
            Path to the extracted repository root directory.
        analysis:
            Repository analysis metadata.

    Returns:
        Structured text context containing repository metadata and selected
        file contents.
    """

    chosen_files = choose_candidate_files_to_read(
        analysis["candidate_files_to_read"] # Where did we get analysis["candidate_files_to_read"] from? Search for ***analysis***. We made a fn called analyse_repository_structure(extracted_repository_path). This is the list of candidate files that we identified during the repository analysis phase as potentially important for understanding the project. These files are typically configuration files, dependency files, and documentation files that can provide valuable insights into the project's structure, dependencies, and purpose. By passing this list to the chose_candidate_files_to_read function, we can prioritize and select the most informative files to read and include in the context for the LLM when generating the repository summary.
        # Just to be clear, we have access to analysis because it's passed into the fn.
    )

    context_parts = [
        "REPOSITORY OVERVIEW",  # Section header used to structure the prompt for the LLM.
        f"File count: {analysis['file_count']}",
        f"Analysis mode: {analysis['analysis_mode']}",
        f"Detected technologies: {', '.join(analysis['technologies'] or ['Unknown'])}",
        f"Detected file extensions: {', '.join(analysis['extensions'][:20]) or 'None'}",  
        # Limit to the first 20 extensions for brevity. Large repositories may contain
        # many different file types, and including too many adds noise to the LLM prompt.
        f"Important files found: {', '.join(analysis['important_files'][:20]) or 'None'}",  
        # Limit to the first 20 items to keep the prompt concise. Large repositories
        # may contain many files, and including too many reduces signal for the LLM.
        "",  
        # Blank line used to visually separate sections in the generated LLM prompt.
        "SELECTED FILE CONTENTS",  
        # Header for the section where selected file contents will be appended.
    ]

    total_characters_used = sum(len(part) for part in context_parts)  # We start with the character count of the initial context parts, which includes the repository overview and section headers. This gives us a baseline character count before we add any file contents.
    # Why count here? Why not count at the end? We need to keep track of the total characters used as we add file contents to ensure that we do not exceed the maximum token limits for the LLM. By counting characters as we go, we can stop adding file contents once we reach a certain threshold (e.g., MAX_CHARACTERS_FOR_LLM_CONTEXT) to ensure that our final context remains within the limits that the LLM can handle effectively. If we waited until the end to count, we might end up with a context that is too large and would need to be truncated, which could lead to loss of important information and less coherent summaries.

    for relative_file_path in chosen_files:
        try:
            file_content = read_repository_text_file(
                repository_path=repository_path,
                relative_file_path=relative_file_path,
            )
        except ValueError:
            continue  # If we encounter an error reading a file (e.g., it's not a valid text file), we skip it and move on to the next one. This allows us to handle cases where some of the candidate files may not be readable as text without crashing the entire context-building process.
            # I think this part is clever - simple, but elegant.
            
        section_text = (
            f"\nFILE: {relative_file_path}\n"  # We add a header for each file that includes its relative path, which provides context to the LLM about where this file is located within the repository structure.
            f"{'-' * 40}\n"  # We add a separator line to visually distinguish the content of different files in the context provided to the LLM. This can help the LLM understand that the content is from different sources and may improve its ability to parse and utilize the information effectively when generating the summary.
            f"{file_content}\n"  # We add the actual content of the file,
        )
        
        if total_characters_used + len(section_text) > MAX_TOTAL_CONTEXT_CHARACTERS:
            break  # If adding the next file's content would exceed our defined character limit for the LLM context, we stop adding more files. This ensures that we stay within the token limits of the LLM while still providing it with a rich context that includes important information about the repository.

        context_parts.append(section_text)  # If we haven't exceeded the character limit, we add the section text for this file to our context parts, which will be combined into the final context string for the LLM.
        total_characters_used += len(section_text)  # We update our total character count to include the newly added file content, so that we can continue to check against our character limit as we add more files.

    return "\n".join(context_parts)  # We join all the parts of our context with newline characters to create a single string that will be passed to the LLM. This context includes an overview of the repository as well as the contents of selected important files, providing the LLM with valuable information to generate a meaningful summary of the repository.



####################################################################################
# All the action happens below - we've declared fns and classes - now we put everything together.
####################################################################################


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

    # TODO: remove 2 lines below later when code is working
    # try:
    #     analysis = analyse_repository_structure(extracted_repository_path) # ***analysis*** is a dictionary containing metadata about the repository that we can use for summarization. For now, it includes things like file count, detected file extensions, important files found, inferred technologies, and a simple heuristic for analysis mode based on the size of the repository. This is just a starting point to show that we can analyze the repository contents and extract useful information from it. In the next steps, we will implement more detailed analysis of the repository structure and contents to generate a meaningful summary.
    try:
        analysis = analyse_repository_structure(extracted_repository_path)
        repository_context = build_repository_context(
            repository_path=extracted_repository_path,
            analysis=analysis,
        )
    finally:
        shutil.rmtree(temporary_directory, ignore_errors=True) # Remove the entire directory tree at temporary_directory, which includes the downloaded ZIP file and the extracted repository. We use ignore_errors=True to ensure that we don't raise an exception if there is an issue with deleting the files (e.g., if they were already deleted or if there are permission issues), as we want to make sure that we clean up any temporary resources we created during the process regardless of any issues that may arise during cleanup.

    # TODO: remove the below later when working
    # return SummarizeResponse(
    #     summary=(
    #         f"Repository '{owner}/{repository_name}' was downloaded, extracted and analysed successfully."
    #     ),
    #     technologies=analysis["technologies"] or ["Unknown"], # If we couldn't infer any technologies, we return "Unknown" to indicate that the analysis did not yield any results in that area. This is just a placeholder for now, as our technology inference is still very basic. In the next steps, we will implement more sophisticated analysis to better identify the technologies used in the project, which should lead to more accurate and informative summaries.
    #     structure=(
    #         f"Total files analysed: {analysis['file_count']}. "
    #         f"Detected file extensions: {', '.join(analysis['extensions'][:10])}. " # We limit to the first 10 extensions for brevity, as some repositories may have a large number of different file types. This is just a placeholder to show that we can analyze the repository contents and extract useful information from it. In the next steps, we will implement more detailed analysis of the repository structure and contents to generate a meaningful summary.
    #         f"Important files found: {', '.join(analysis['important_files'][:10]) or 'None'}. "
    #         f"Analysis mode: {analysis['analysis_mode']}."
    #     ),
    # )

    return SummarizeResponse(
        summary=(
            f"Repository '{owner}/{repository_name}' was downloaded, extracted, "
            "analysed, and prepared for LLM summarisation."
        ),
        technologies=analysis["technologies"] or ["Unknown"],
        structure=(
            f"Total files analysed: {analysis['file_count']}. "
            f"Important files read: "
            f"{', '.join(choose_candidate_files_to_read(analysis['candidate_files_to_read'])) or 'None'}. "
            f"Context length prepared for LLM: {len(repository_context)} characters. "
            f"Analysis mode: {analysis['analysis_mode']}."
        ),
    )
