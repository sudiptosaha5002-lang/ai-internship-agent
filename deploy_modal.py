import modal
import os

app = modal.App("internship-agent-final-release")

# Define persistent volume for resumes
volume = modal.Volume.from_name("internai-resumes", create_if_missing=True)

# Build the image with all app dependencies and the entire source code
image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("poppler-utils")
    .pip_install_from_requirements(os.path.join(os.path.dirname(__file__), "requirements.txt"))
    .add_local_dir(os.path.dirname(__file__), remote_path="/root", ignore=[".git", "__pycache__", "*.md", "test_*.py", "naukri_dom.html"])
)

@app.function(
    image=image,
    min_containers=1,
    volumes={"/data": volume},
    env={
        "UPLOAD_DIR": "/data/uploads",
        "SQLITE_DB_PATH": "/data/internai.db"
    },
    secrets=[modal.Secret.from_dotenv()],
)
@modal.wsgi_app()
def flask_app():
    import sys
    sys.path.append("/root")
    from app import app as flask_instance
    return flask_instance
