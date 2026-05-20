import os
import modal

# Create a Modal App instance
app = modal.App("ai-internship-agent")

# 1. Define the Modal Image
# This installs requirements, Playwright, and includes all project files
image = (
    modal.Image.debian_slim(python_version="3.10")
    # Install dependencies from the project
    .pip_install_from_requirements("requirements.txt")
    # Install playwright system dependencies and browsers
    .run_commands(
        "playwright install-deps",
        "playwright install chromium"
    )
    # Add all project files into the container
    .add_local_dir(".", remote_path="/root/project", ignore=[
        ".git", "__pycache__", "internai.db", "get-pip.py",
        "naukri_dom.html", ".env", "modal.env", "*.pyc"
    ])
)

# 2. Define a Persistent Volume
# This ensures that /data/internai.db is NEVER deleted
volume = modal.Volume.from_name("internai-data", create_if_missing=True)

# 3. Define the Web Endpoint
@app.function(
    image=image,
    volumes={"/data": volume},
    # This pulls your API keys from the Modal Dashboard Secrets
    secrets=[modal.Secret.from_name("internai-secrets")],
    # Give it enough timeout for deep research and heavy scraping
    timeout=600,
)
@modal.concurrent(max_inputs=100)
@modal.wsgi_app()
def flask_app():
    import sys
    # Add the project directory to Python's search path
    sys.path.insert(0, "/root/project")
    os.chdir("/root/project")

    # VERY IMPORTANT: Tell the Flask app to save data to the permanent Modal volume
    os.environ["SQLITE_DB_PATH"] = "/data/internai.db"
    os.environ["UPLOAD_DIR"] = "/data/uploads"
    os.makedirs("/data/uploads", exist_ok=True)

    # Import the Flask instance from app.py
    from app import app as web_app
    return web_app
