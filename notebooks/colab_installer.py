"""
Script to install topiary in a google colab notebook. This requires two cells
that, at a minimum, look like: 

Cell #1:

#-------------------------------------------------------------------------------
import os
os.chdir("/content/")

import urllib.request
urllib.request.urlretrieve(SCRIPT_URL,"colab_installer.py")

import colab_installer
colab_installer.install_topiary(install_raxml=True,
                                install_generax=True,
                                bin_cache="/content/gdrive/MyDrive/topiary_bin")
#-------------------------------------------------------------------------------

Cell #2:
#-------------------------------------------------------------------------------
import topiary
import numpy as np
import pandas as pd

import os
os.chdir("/content/")

topiary._in_notebook = "colab"
import colab_installer
colab_installer.initialize_environment()
colab_installer.mount_google_drive(google_drive_directory)
#-------------------------------------------------------------------------------


Note: this script uses condacolab to set up a Python 3.12 environment (2025/03/16). 
If colab updates from python 3.12, the site-packages path in initialize_environment
may need to be updated.

"""

from tqdm.auto import tqdm
import sys
import subprocess
import time
import os
import re
import shutil

def _run_install_cmd(bash_to_run, description):
    """
    Run an installation command.

    bash_to_run : str
        bash command as a string
    description : str
        description of what is being done
    """

    no_space = re.sub(" ", "_", description)
    status_file = f"/content/software/{no_space}.installed"

    if os.path.isfile(status_file):
        print(f"{description} already installed.")
        return

    os.makedirs("software", exist_ok=True)
    os.chdir("software")

    print(f"Installing {description}... ", end="", flush=True)
    f = open(f"{no_space}_tmp-script.sh", "w")
    f.write(bash_to_run)
    f.close()

    result = subprocess.run(["bash", f"{no_space}_tmp-script.sh"],
                             stdout=subprocess.PIPE,
                             stderr=subprocess.PIPE,
                             text=True)
    
    if result.returncode != 0:
        print("\nFailed!", flush=True)
        print(result.stdout, flush=True)
        print(result.stderr, flush=True)
        raise RuntimeError(f"Installation of {description} failed!")

    f = open(f"{no_space}_stdout.txt", "w")
    f.write(result.stdout)
    f.close()

    f = open(f"{no_space}_stderr.txt", "w")
    f.write(result.stderr)
    f.close()

    os.chdir("..")

    f = open(status_file, 'w')
    f.write("Installed\n")
    f.close()
    
    print("Complete.", flush=True)


def install_topiary(install_raxml=True, install_generax=True, bin_cache=None):
    """
    Install topiary and its dependencies.

    install_raxml : bool, default=True
        whether to install raxml-ng
    install_generax : bool, default=True
        whether to install generax
    bin_cache : str, optional
        path to a directory (e.g. on Google Drive) to store and retrieve 
        pre-compiled binaries for raxml-ng and generax.
    """

    os.chdir("/content/")

    # 1. condacolab setup
    try:
        import condacolab
        condacolab.check()
    except (ImportError, Exception):
        print("Initializing condacolab. This will restart the kernel.", flush=True)
        subprocess.run([sys.executable, "-m", "pip", "install", "-q", "condacolab"], check=True)
        import condacolab
        condacolab.install()
        return

    # 2. Clone topiary to get environment.yml and compilation scripts
    topiary_clone = """
    if [ -d topiary ]; then
        rm -rf topiary
    fi
    git clone https://github.com/harmsm/topiary.git
    """
    
    # 3. Use environment.yml to install dependencies
    # condacolab environment is 'base'
    env_install = """
    unset PYTHONPATH
    rm -f /usr/local/conda-meta/pinned
    mamba env update -n base -f topiary/environment.yml --quiet
    mamba install --yes --quiet -c conda-forge ghostscript cmake
    """

    # 4. Binary handling logic
    bin_dir = "/usr/local/bin"
    
    raxml_cmd = ""
    if install_raxml:
        target = f"{bin_dir}/raxml-ng"
        if bin_cache and os.path.exists(os.path.join(bin_cache, "raxml-ng")):
            print("Found cached raxml-ng. Using it.")
            shutil.copy(os.path.join(bin_cache, "raxml-ng"), target)
            os.chmod(target, 0o755)
        else:
            raxml_cmd = f"""
            cd topiary/dependencies
            bash compile-raxml-ng.sh
            cd ../..
            """
            if bin_cache:
                raxml_cmd += f"mkdir -p {bin_cache}\n"
                raxml_cmd += f"cp {target} {bin_cache}/raxml-ng\n"

    generax_cmd = ""
    if install_generax:
        target = f"{bin_dir}/generax"
        if bin_cache and os.path.exists(os.path.join(bin_cache, "generax")):
            print("Found cached generax. Using it.")
            shutil.copy(os.path.join(bin_cache, "generax"), target)
            os.chmod(target, 0o755)
        else:
            generax_cmd = """
            apt-get install -y flex bison libgmp3-dev
            cd topiary/dependencies
            bash compile-generax.sh
            cd ../..
            """
            if bin_cache:
                generax_cmd += f"mkdir -p {bin_cache}\n"
                generax_cmd += f"cp {target} {bin_cache}/generax\n"

    # 5. Patch topiary for Colab (mpirun --allow-run-as-root)
    # We do this before installing topiary so the patched files are installed.
    patch_cmd = r"""
    # add --allow-run-as-root to mpirun calls
    files_to_patch="topiary/src/topiary/generax/_generax.py topiary/src/topiary/generax/_reconcile_bootstrap.py topiary/src/topiary/_private/mpi/mpi.py"
    for x in ${files_to_patch}; do
        if [ -f ${x} ]; then
            sed -i 's/\[\"mpirun\"/\[\"mpirun\",\"--allow-run-as-root\"/g' ${x}
        fi
    done
    """

    # 6. Install topiary itself
    topiary_install = """
    cd topiary
    pip install . -y -vv
    cd ..
    """

    print("Setting up environment.", flush=True)

    # List of commands to run
    description_list = ["cloning topiary", "conda packages"]
    cmd_list = [topiary_clone, env_install]

    if raxml_cmd != "":
        description_list.append("compiling raxml-ng")
        cmd_list.append(raxml_cmd)
    
    if generax_cmd != "":
        description_list.append("compiling generax")
        cmd_list.append(generax_cmd)

    description_list.append("patching topiary")
    cmd_list.append(patch_cmd)

    description_list.append("installing topiary")
    cmd_list.append(topiary_install)

    # Install each package
    pbar = tqdm(range(len(cmd_list)))
    for i in pbar:
        _run_install_cmd(cmd_list[i], description_list[i])
        pbar.refresh()
        time.sleep(0.5)
        
    print("\nInstallation complete! Restarting runtime.", flush=True)
    time.sleep(2)
    os._exit(0)


def initialize_environment():
    """
    Initialize environment variables for topiary in Colab.
    """
        
    os.environ["PYTHONPATH"] = ""
    os.environ["TOPIARY_MAX_SLOTS"] = "1"

    # condacolab handles most of this, but we ensure site-packages is in path
    # just in case, and for older notebook versions. 
    py_version = f"{sys.version_info.major}.{sys.version_info.minor}"
    to_append = f'/usr/local/lib/python{py_version}/site-packages'
    if to_append not in sys.path:
        sys.path.append(to_append)

    print("Environment initialized.")

def mount_google_drive(google_drive_directory):
    """
    Mount Google Drive and change directory to a specific project folder.

    google_drive_directory : str
        path relative to 'My Drive' to use as working directory.
    """

    google_drive_directory = google_drive_directory.strip()
    if google_drive_directory != "":

        # Set up google drive
        from google.colab import drive
        drive.mount('/content/gdrive/')

        working_dir = f"/content/gdrive/MyDrive/{google_drive_directory}"
        os.makedirs(working_dir, exist_ok=True)
        os.chdir(working_dir)
    
    print(f"Working directory: {os.getcwd()}")