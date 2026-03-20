"""
Script to install topiary in a google colab notebook. This requires two cells
that, at a minimum, look like: 

Cell #1:

#-------------------------------------------------------------------------------
import colab_installer
colab_installer.setup_condacolab()
#-------------------------------------------------------------------------------

Cell #2:
#-------------------------------------------------------------------------------
import colab_installer
colab_installer.install_topiary(install_raxml=True,
                                install_generax=True,
                                bin_cache="/content/gdrive/MyDrive/topiary_bin")

import topiary
import numpy as np
import pandas as pd
import os
os.chdir("/content/")
topiary._in_notebook = "colab"
colab_installer.initialize_environment()
colab_installer.mount_google_drive(google_drive_directory)
#-------------------------------------------------------------------------------

Note: this script uses condacolab to set up a Python 3.12 environment (2025/03/16). 

"""

import sys
import subprocess
import os
import shutil
import time
from tqdm.auto import tqdm

def setup_condacolab():
    """
    Install and initialize condacolab. This will restart the kernel.
    """
    try:
        import condacolab
        condacolab.check()
        print("condacolab already installed and initialized.")
    except (ImportError, Exception):
        print("Installing condacolab. This will restart the kernel. Please re-run the next cell after restart.", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "condacolab"], check=True)
        import condacolab
        condacolab.install()

def _run_cmd(cmd, description):
    print(f"Running {description}... ", end="", flush=True)
    result = subprocess.run(cmd, shell=True, capture_output=True, text=True)
    if result.returncode != 0:
        print("\nFailed!", flush=True)
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        raise RuntimeError(f"Command failed: {description}")
    print("Done.")

def install_topiary(install_raxml=True, install_generax=True, bin_cache=None, ncbi_key=None):
    """
    Install topiary and its dependencies using the official install.sh.

    install_raxml : bool, default=True
        whether to install raxml-ng
    install_generax : bool, default=True
        whether to install generax
    bin_cache : str, optional
        path to a directory (e.g. on Google Drive) to store and retrieve 
        pre-compiled binaries for raxml-ng and generax.
    ncbi_key : str, optional
        NCBI API key to set during installation.
    """

    os.chdir("/content/")

    # 1. Clone topiary
    if os.path.exists("topiary"):
        shutil.rmtree("topiary")
    _run_cmd("git clone https://github.com/harmslab/topiary.git", "Cloning topiary")

    # 2. Seed bin_cache if provided
    bin_dir = "/usr/local/bin"
    if bin_cache:
        bin_cache = os.path.abspath(bin_cache)
        if os.path.exists(os.path.join(bin_cache, "raxml-ng")):
            print("Seeding raxml-ng from cache...")
            shutil.copy(os.path.join(bin_cache, "raxml-ng"), os.path.join(bin_dir, "raxml-ng"))
            os.chmod(os.path.join(bin_dir, "raxml-ng"), 0o755)
        
        if os.path.exists(os.path.join(bin_cache, "generax")):
            print("Seeding generax from cache...")
            shutil.copy(os.path.join(bin_cache, "generax"), os.path.join(bin_dir, "generax"))
            os.chmod(os.path.join(bin_dir, "generax"), 0o755)

    # 3. Patch topiary for Colab (mpirun --allow-run-as-root)
    # We do this before installing so the patched files are used.
    print("Patching topiary for Colab...")
    files_to_patch = [
        "topiary/src/topiary/generax/_generax.py",
        "topiary/src/topiary/generax/_reconcile_bootstrap.py",
        "topiary/src/topiary/_private/mpi/mpi.py"
    ]
    for f in files_to_patch:
        if os.path.exists(f):
             _run_cmd(fr"sed -i 's/\[\"mpirun\"/\[\"mpirun\",\"--allow-run-as-root\"/g' {f}", f"Patching {os.path.basename(f)}")

    # 4. Run official install.sh
    # We use --name base to update the current condacolab environment.
    # We use --no-cluster and --yes for non-interactive mode.
    # We use --keep-existing to use the binaries we seeded from cache.
    install_cmd = "cd topiary && bash install.sh --name base --no-cluster --yes --keep-existing"
    if not install_raxml:
        install_cmd += " --no-raxml"
    if not install_generax:
        install_cmd += " --no-generax"
    if ncbi_key:
        install_cmd += f" --ncbi-key {ncbi_key}"
    
    _run_cmd(install_cmd, "Running topiary/install.sh")

    # 5. Update bin_cache after installation if needed
    if bin_cache:
        os.makedirs(bin_cache, exist_ok=True)
        if install_raxml and os.path.exists(os.path.join(bin_dir, "raxml-ng")):
            shutil.copy(os.path.join(bin_dir, "raxml-ng"), os.path.join(bin_cache, "raxml-ng"))
        if install_generax and os.path.exists(os.path.join(bin_dir, "generax")):
            shutil.copy(os.path.join(bin_dir, "generax"), os.path.join(bin_cache, "generax"))

    print("\nInstallation complete! Topiary is ready to use.")

def initialize_environment():
    """
    Initialize environment variables for topiary in Colab.
    """
    os.environ["TOPIARY_MAX_SLOTS"] = "1"
    # Ensure site-packages is in path (condacolab usually handles this after restart)
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    to_append = f'/usr/local/lib/python{py_version}/site-packages'
    if to_append not in sys.path:
        sys.path.append(to_append)
    print("Environment initialized.")

def mount_google_drive(google_drive_directory):
    """
    Mount Google Drive and change directory to a specific project folder.
    """
    google_drive_directory = google_drive_directory.strip()
    if google_drive_directory != "":
        from google.colab import drive
        drive.mount('/content/gdrive/')
        working_dir = f"/content/gdrive/MyDrive/{google_drive_directory}"
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)
    print(f"Working directory: {os.getcwd()}")