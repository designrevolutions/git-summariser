#!/usr/bin/env python3
"""
Main FastAPI application for the Nebius assignment project.

This version of the service can now:
1. Validate a public GitHub repository URL.
2. Extract the repository owner and name.
3. Check that the repository is accessible.
4. Download the repository as a ZIP archive.
5. Extract the repository into a temporary working directory.
6. Traverse the repository while ignoring large dependency/build/cache folders.
7. Collect structural metadata such as:
   - file count
   - file extensions present
   - directory names
   - important project files
8. Infer likely technologies from file extensions and marker files.
9. Select important files to read.
10. Build structured repository context ready to be sent to an LLM.

The next step will be to send the prepared context to an LLM and return a
real summary, technology list, and structure description.

Run:
    uvicorn main:app --reload

Then open:
    http://127.0.0.1:8000/docs
"""

import re

from dotenv import load_dotenv
from openai import OpenAI

from pathlib import Path  # Object-oriented, cross-platform file path handling.
import os  # Used for recursive directory traversal with os.walk().
import shutil  # Used to remove temporary directories after processing.
import tempfile  # Used to create temporary working directories safely.
from urllib.parse import urlparse  # Used to split a URL into parts for validation.
import zipfile  # Used to extract downloaded ZIP archives.

import requests  # Third-party HTTP client for downloading repository archives.
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import logging # The app not working - need to add some logging to understand where it's failing. This is a common practice when developing APIs, as it helps you trace the flow of execution and identify any issues that arise.
import time

load_dotenv() # Load environment variables from .env file, including the API key for the LLM.

# ######++++++ Logging code
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)

logger = logging.getLogger(__name__)
# ######++++++ End of logging code


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
# These directories usually contain dependencies, caches, or build artefacts.
# Ignoring them keeps analysis faster and focuses on the real project files.

IMPORTANT_FILENAMES = {
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
# These filenames are good signals for understanding a repository because they
# often describe dependencies, build configuration, or project purpose.

EXTENSION_TO_TECHNOLOGY = {
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
# Simple extension-to-technology mapping used as one technology-detection
# signal. It is heuristic-based, so it is useful but not perfect.

FILE_TECHNOLOGY_MAP: dict[str, set[str]] = {
    "Python": {"requirements.txt", "pyproject.toml", "setup.py", "pipfile"},
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
# Marker files can reveal technologies even when source file extensions alone
# are not enough.

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
# Lower numbers mean higher priority when deciding which files to read first.

MAX_FILES_FOR_BROAD_ANALYSIS = 2000
# If a repository has more files than this after filtering, switch from broad
# analysis to selective analysis.

MAX_CANDIDATE_FILES_TO_READ = 12
# Maximum number of important files to read and include in LLM context.

MAX_CHARACTERS_PER_FILE = 4000
# Prevent a single large file from dominating the context sent to the LLM.

MAX_TOTAL_CONTEXT_CHARACTERS = 20000
# Hard limit for the total repository context we build for the LLM.

AI_MODEL_NAME = "deepseek-ai/DeepSeek-R1-0528"


class SummarizeRequest(BaseModel):
    """
    Request model for the /summarize endpoint.

    Attributes:
        github_url:
            URL of the GitHub repository to analyse.

    Example:
        {
            "github_url": "https://github.com/psf/requests"
        }
    """

    github_url: str = Field(..., description="Public GitHub repository URL")


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
    technologies: list[str]
    structure: str


class LlmRepositorySummary(BaseModel):
    """
    Structured repository summary parsed from the LLM response.
    Same as SummarizeResponse for now, but defined separately to allow for different
    parsing and validation logic later when we implement the LLM call and response parsing.

    Attributes:
    summary:
        Human-readable summary of what the project does.
    technologies:
        Main technologies, languages, and frameworks used in the project.
    structure:
        A brief overview of the project structure.
    """

    summary: str
    technologies: list[str]
    structure: str


# Main FastAPI app instance. The metadata is also used in the automatic docs.
app = FastAPI(
    title="GitHub Repository Summarizer API",
    description="API to summarize the contents of a GitHub repository.",
    version="0.3.0",
    openapi_tags=[
        {
            "name": "Repository Analysis",
            "description": "Operations related to analysing GitHub repositories.",
        }
    ],
)


def parse_github_repository_url(github_url: str) -> tuple[str, str]:
    """
    Validate and parse a GitHub repository URL.

    This function checks that:
    - the URL uses http or https
    - the hostname is github.com
    - the path contains at least two segments: owner and repository name

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
            If the URL is invalid or does not point to a GitHub repository.
    """

    parsed_url = urlparse(github_url)

    # Ensure the scheme is http or https.
    if parsed_url.scheme not in {"http", "https"}:
        raise ValueError("URL must start with http:// or https://")

    # hostname is clearer than netloc here because we only care about the domain.
    if parsed_url.hostname != "github.com":
        raise ValueError("URL must point to github.com")

    # Remove empty segments caused by leading/trailing slashes.
    path_parts = [part for part in parsed_url.path.split("/") if part]

    # A valid repository URL should at least look like /owner/repo
    if len(path_parts) < 2:
        raise ValueError("URL path must contain at least owner and repository name")

    owner, repository_name = path_parts[0], path_parts[1]

    # Handle URLs that end with .git
    if repository_name.endswith(".git"):
        repository_name = repository_name[:-4]

    # Final sanity check in case stripping .git left an empty repo name.
    if not owner or not repository_name:
        raise ValueError("Owner and repository name could not be determined")

    return owner, repository_name


def build_github_archive_urls_for_zip_download(
    owner: str,
    repository_name: str,
) -> list[str]:
    """
    Build likely GitHub ZIP download URLs for the repository archive.

    GitHub archive URLs depend on the branch name. For now, this version keeps
    things simple by trying the most common defaults: main and master.

    Args:
        owner:
            The owner of the GitHub repository.
        repository_name:
            The name of the GitHub repository.

    Returns:
        A list of candidate archive URLs.
    """

    return [
        f"https://github.com/{owner}/{repository_name}/archive/refs/heads/main.zip",
        f"https://github.com/{owner}/{repository_name}/archive/refs/heads/master.zip",
    ]


def download_github_repository_as_zip(
    owner: str,
    repository_name: str,
    working_directory: Path,
) -> Path:
    """
    Download the GitHub repository as a ZIP file into the supplied directory.

    This function tries likely archive URLs until one succeeds.

    Args:
        owner:
            The owner of the GitHub repository.
        repository_name:
            The name of the GitHub repository.
        working_directory:
            Directory where the downloaded ZIP file will be saved.

    Returns:
        Path to the downloaded ZIP file.

    Raises:
        ValueError:
            If the repository cannot be verified or the ZIP cannot be downloaded.
    """

    archive_urls = build_github_archive_urls_for_zip_download(owner, repository_name)

    # Fast existence/accessibility check so we fail quickly for bad repository URLs.
    # This is still a lightweight shortcut for now; later it could be replaced
    # with a GitHub API call to discover the actual default branch more cleanly.
    try:
        response = requests.head(
            f"https://github.com/{owner}/{repository_name}",
            timeout=10,
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

    zip_file_path = working_directory / f"{repository_name}.zip"

    for archive_url in archive_urls:
        try:
            # stream=True lets us write the file in chunks rather than loading
            # the whole archive into memory at once.
            response = requests.get(archive_url, timeout=30, stream=True)
        except requests.RequestException as error:
            raise ValueError(
                f"Failed to download GitHub repository archive from "
                f"{archive_url}: {error}"
            ) from error

        if response.status_code == 200:
            with open(zip_file_path, "wb") as zip_file:
                for chunk in response.iter_content(chunk_size=8192):
                    if chunk:
                        zip_file.write(chunk)

            return zip_file_path

        # 404 here usually means this guessed branch name was wrong, so try the
        # next candidate URL.
        if response.status_code == 404:
            continue

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
        with zipfile.ZipFile(zip_file_path, "r") as zip_file:
            zip_file.extractall(extract_to_directory)
    except zipfile.BadZipFile as error:
        raise ValueError(
            f"Failed to extract ZIP file {zip_file_path}: {error}. "
            "The downloaded file may not be a valid ZIP archive."
        ) from error

    extracted_directories = [
        path for path in extract_to_directory.iterdir() if path.is_dir()
    ]

    if not extracted_directories:
        raise ValueError(f"No repository folder found in {extract_to_directory}")

    if len(extracted_directories) == 1:
        return extracted_directories[0]

    # Fallback: choose the first directory in sorted order if multiple appear.
    # This should be rare for normal GitHub archive downloads.
    return sorted(extracted_directories)[0]


def download_and_extract_repository(
    owner: str,
    repository_name: str,
) -> tuple[Path, Path]:
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

    # Create a temporary working directory so we do not clutter the project
    # folder with downloaded archives and extracted files.
    temporary_directory = Path(tempfile.mkdtemp(prefix="repo_summarizer_"))
    print(f"Temporary directory created at: {temporary_directory}")

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
        # Always clean up the temporary directory if anything fails during
        # download or extraction, then re-raise the original exception.
        shutil.rmtree(temporary_directory, ignore_errors=True)
        raise


def infer_technologies_from_files(
    extensions: set[str],
    important_files: set[str],
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

    # Use file extensions as one signal.
    for extension in extensions:
        technology = EXTENSION_TO_TECHNOLOGY.get(extension)
        if technology:
            detected_technologies.add(technology)

    # Normalise filenames to lower case so matching is case-insensitive.
    lower_important_files = {file_name.lower() for file_name in important_files}

    # Marker files are another strong signal. For example, package.json often
    # implies Node.js even if there are not many visible .js files.
    for technology, markers in FILE_TECHNOLOGY_MAP.items():
        # & checks whether the two sets have any overlap.
        if lower_important_files & markers:
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
    candidate_files_to_read = []
    # This list contains relative paths to important files that are good
    # candidates to read later and include in the LLM context.

    for root, dirs, files in os.walk(repository_path):
        # Modify dirs in place so os.walk will not descend into ignored folders.
        dirs[:] = [directory for directory in dirs if directory not in IGNORED_DIRECTORIES]

        # Record directory names as a lightweight structure signal.
        directories_seen.add(Path(root).name)

        for file_name in files:
            file_count += 1

            suffix = Path(file_name).suffix.lower()
            if suffix:
                extensions_found.add(suffix)

            lower_file_name = file_name.lower()

            if lower_file_name in IMPORTANT_FILENAMES:
                important_files_found.add(lower_file_name)

                full_file_path = Path(root) / file_name
                relative_file_path = full_file_path.relative_to(repository_path)

                # as_posix() keeps the path readable and consistent by using
                # forward slashes even on Windows.
                candidate_files_to_read.append(relative_file_path.as_posix())

    inferred_technologies = infer_technologies_from_files(
        extensions=extensions_found,
        important_files=important_files_found,
    )

    analysis_mode = "broad"

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
    candidate_files: list[str],
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
        # Use the filename for priority lookup, and the full relative path as a
        # stable tiebreaker.
        file_name = Path(relative_path).name.lower()
        priority = FILE_PRIORITY_ORDER.get(file_name, 100)
        return priority, relative_path.lower()

    # sorted(..., key=...) is the standard Python pattern here: the key
    # function defines the sorting criteria for each item.
    sorted_files: list[str] = sorted(candidate_files, key=sort_key)

    return sorted_files[:MAX_CANDIDATE_FILES_TO_READ]


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
        # Try reading as UTF-8 text. If the file is binary or uses an
        # incompatible encoding, raise a clean ValueError instead.
        content = full_file_path.read_text(encoding="utf-8")
    except UnicodeDecodeError:
        raise ValueError(f"File {relative_file_path} could not be read as text.")
    except OSError as error:
        raise ValueError(
            f"An error occurred while reading file {relative_file_path}: {error}"
        ) from error

    # Trim large files so they do not dominate the context sent to the LLM.
    return content[:max_characters]


# The next two functions are where we build the context for the LLM and create a client to call the LLM. They are separate from the repository analysis logic to keep things modular and clear.
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

    # The candidate file list comes from analyse_repository_structure().
    chosen_files = choose_candidate_files_to_read(
        analysis["candidate_files_to_read"]
    )

    context_parts = [
        "REPOSITORY OVERVIEW",
        f"File count: {analysis['file_count']}",
        f"Analysis mode: {analysis['analysis_mode']}",
        f"Detected technologies: {', '.join(analysis['technologies'] or ['Unknown'])}",
        f"Detected file extensions: {', '.join(analysis['extensions'][:20]) or 'None'}",
        f"Important files found: {', '.join(analysis['important_files'][:20]) or 'None'}",
        "",
        "SELECTED FILE CONTENTS",
    ]
    # This creates a clearly structured prompt for the LLM.

    # Start with the characters already used by the overview/header section.
    total_characters_used = sum(len(part) for part in context_parts)

    for relative_file_path in chosen_files:
        try:
            file_content = read_repository_text_file(
                repository_path=repository_path,
                relative_file_path=relative_file_path,
            )
        except ValueError:
            # Skip unreadable files rather than failing the whole request.
            continue

        section_text = (
            f"\nFILE: {relative_file_path}\n"
            f"{'-' * 40}\n"
            f"{file_content}\n"
        )

        # Stop before exceeding the total context size budget.
        if total_characters_used + len(section_text) > MAX_TOTAL_CONTEXT_CHARACTERS:
            break

        context_parts.append(section_text)
        total_characters_used += len(section_text)

    return "\n".join(context_parts)


def parse_llm_summary_response(response_text: str) -> LlmRepositorySummary:
    """
    Parse the LLM response written in a labelled plain-text format.

    Expected format:

    SUMMARY:
    ...

    TECHNOLOGIES:
    Python, FastAPI, Requests

    STRUCTURE:
    ...

    Args:
        response_text:
            Raw text returned by the LLM.

    Returns:
        Parsed structured repository summary.

    Raises:
        ValueError:
            If the response does not contain the expected sections.
    """

    # Use regular expressions to extract the sections of the response based on the labels.
    summary_match = re.search(
        r"SUMMARY:\s*(.*?)\s*TECHNOLOGIES:",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    technologies_match = re.search(
        r"TECHNOLOGIES:\s*(.*?)\s*STRUCTURE:",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )
    structure_match = re.search(
        r"STRUCTURE:\s*(.*)",
        response_text,
        flags=re.DOTALL | re.IGNORECASE,
    )

    if not summary_match or not technologies_match or not structure_match:
        raise ValueError(
            "Nebius response did not match the expected labelled format."
        )

    summary = summary_match.group(1).strip() # Remove leading/trailing whitespace for cleaner output. group(1) gets the content of the first (and only) capturing group in the regex, which is the text between the labels.
    technologies_text = technologies_match.group(1).strip()
    structure = structure_match.group(1).strip()

    # Using list comprehension to split the technologies by comma, strip whitespace, and filter out any empty items. This allows for a flexible number of technologies in the response.
    # I don't like this version of the code as much because it's a bit dense, but it is more robust to different formatting of the technologies list in the LLM response. For example, it can handle extra spaces or missing commas more gracefully than a simple split(",") would.
    # technologies = [
    #     item.strip()
    #     for item in technologies_text.split(",")
    #     if item.strip()
    # ]


    # This version is more verbose but clearer to read and understand. It achieves the same result as the list comprehension above, but with more explicit steps. It also allows for easier debugging if needed, since you can inspect the intermediate variables.
    technologies = []

    for item in technologies_text.split(","):
        
        cleaned = item.strip()

        if cleaned:
            technologies.append(cleaned)
      

    return LlmRepositorySummary(
        summary=summary,
        technologies=technologies,
        structure=structure,
    )


def create_nebius_client() -> OpenAI:
    """
    Create an OpenAI-compatible client for Nebius Token Factory.

    Returns:
        Configured OpenAI client.

    Raises:
        ValueError:
            If the NEBIUS_API_KEY environment variable is missing.
    """

    api_key = os.environ.get("NEBIUS_API_KEY")

    if not api_key:
        raise ValueError("NEBIUS_API_KEY environment variable is not set.")

    return OpenAI(
        base_url="https://api.tokenfactory.nebius.com/v1/",
        api_key=api_key,
        timeout=60.0,
        max_retries=2,
    )


# I've now added logging statements to the generate_repository_summary_with_nebius function to help trace the flow of execution and identify where it might be failing. This is a common practice when developing APIs, as it allows you to see how far the code is getting before an error occurs, and what the internal state is at various points in the process. The logging statements will print messages to the console with timestamps and log levels, which can be very helpful for debugging and monitoring the API's behavior when it is running.
def generate_repository_summary_with_nebius(
    repository_context: str,
) -> LlmRepositorySummary:
    """
    Send repository context to Nebius and return a structured repository summary.
    """

    logger.info("Creating Nebius client")
    client = create_nebius_client()

    prompt = f"""
Analyse the following GitHub repository context and return the result in exactly this format:

SUMMARY:
<short human-readable explanation of what the project does>

TECHNOLOGIES:
<comma-separated list of the main technologies, languages, frameworks, or tools>

STRUCTURE:
<brief description of how the repository is organised>

Do not include any extra headings, notes, or commentary.

Repository context:

{repository_context}
"""

    try:
        logger.info("Calling Nebius model: %s", AI_MODEL_NAME)
        response = client.chat.completions.create(
            model=AI_MODEL_NAME,
            temperature=0.2,
            messages=[
                {"role": "user", "content": prompt}
            ],
        )
        logger.info("Nebius response received")
    except Exception as error:
        logger.exception("Nebius API request failed")
        raise ValueError(f"Nebius API request failed: {error}") from error

    content = response.choices[0].message.content

    if not content:
        raise ValueError("Nebius returned an empty response.")

    logger.info("Nebius response received: %d characters", len(content))

    return parse_llm_summary_response(content)


####################################################################################
# All the action happens below - we've declared functions and classes, and now we
# wire everything together.
# Below are what is called the route fns. These are the functions that get called when an HTTP request hits a certain endpoint.
####################################################################################


@app.get("/")
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

    # Handy debug print to confirm requests are reaching the running server.
    print("Root endpoint was called")
    return {"message": "Hello, Nebius assignment project! The API is running."}


# I added some logging to the summarize_repository function to help trace the flow of execution and identify where it might be failing. This is a common practice when developing APIs, as it allows you to see how far the code is getting before an error occurs, and what the internal state is at various points in the process.
@app.post("/summarize", response_model=SummarizeResponse)
def summarize_repository(request: SummarizeRequest) -> SummarizeResponse:
    """
    Validate the GitHub URL, download the repository, analyse it, build
    repository context, send that context to Nebius, and return the result.
    """

    temporary_directory: Path | None = None

    try:
        logger.info("Step 1: Parsing GitHub URL")
        owner, repository_name = parse_github_repository_url(request.github_url)

        logger.info("Step 2: Downloading and extracting repository %s/%s", owner, repository_name)
        temporary_directory, extracted_repository_path = download_and_extract_repository(
            owner=owner,
            repository_name=repository_name,
        )

        logger.info("Step 3: Analysing repository structure")
        analysis = analyse_repository_structure(extracted_repository_path)

        logger.info("Step 4: Building repository context")
        repository_context = build_repository_context(
            repository_path=extracted_repository_path,
            analysis=analysis,
        )

        logger.info("Step 5: Sending context to Nebius")
        llm_result = generate_repository_summary_with_nebius(
            repository_context=repository_context,
        )

        logger.info("Step 6: Returning response")
        return SummarizeResponse(
            summary=llm_result.summary,
            technologies=llm_result.technologies or analysis["technologies"] or ["Unknown"],
            structure=llm_result.structure,
        )

    except ValueError as error:
        logger.exception("Summarisation failed")
        raise HTTPException(status_code=400, detail=str(error)) from error

    finally:
        if temporary_directory is not None:
            logger.info("Cleaning up temporary directory: %s", temporary_directory)
            shutil.rmtree(temporary_directory, ignore_errors=True)