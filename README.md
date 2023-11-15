# triton
[PyTriton](https://triton-inference-server.github.io/pytriton/0.3.1/) server for serving model inference requests.
The inference server runs in a [Singularity](https://docs.sylabs.io/guides/3.0/user-guide) container, as we are otherwise unable to install PyTriton in the cluster environment.

## Do This First
As per the [Martinos Docs](https://www.nmr.mgh.harvard.edu/martinos/userInfo/computer/docker.php), it is important create a symlink for the `~/.singularity` folder, otherwise Singularity will generate many GBs of data that surpass the user home quota.

If you previously used Singularity, first delete the folder in your home directory:
```bash
rm -rf ~/.singularity
```

Then, create a symlink to `/space/neo/4/.singularity` (or some other location):
```bash
ln -s /space/neo/4/.singularity ~/.singularity
```

## Image Setup
**This setup only needs to be completed if building a new image.**

Images cannot be built on the cluster without `sudo` privleges. Instead, the image can be built on a development machine and uploaded to the cluster. See the [Singularity docs](https://docs.sylabs.io/guides/3.0/user-guide/installation.htm) for OS-dependent installation for your local machine; these steps are enumerated below for Windows.

The following examples may be helpful, though the `.def` file takes tips from several sources. Importantly, we do not use an Nvidia-provided base docker image because it 1) takes a long time to download and presented (potentially VM-related) issues when generating a `.sif` file and 2) PyTorch handles isstallation of CUDA and cuDNN, so we can keep the base image lighter.
- <https://github.com/sylabs/examples/blob/master/machinelearning/intel-tensorflow/Singularity.mkl.def>
- <https://github.com/bdusell/singularity-tutorial>

### Singularity Installation on Windows
If on Linux or Mac, read the Singularity docs. For Linux, you need `sudo`. You can also potentially skip this by exploring the free Singularity remote build server. The following is what worked for building the image on Windows. In a nutshell, we will use a prebuilt image to run a VM with singularity installed. From within the VM, we can then build a Singularity image from the `.def` file on Windows.

1. Install the following:
    - [Git for Windows](https://gitforwindows.org/) and/or [Cygwin](https://www.cygwin.com/) (for a Unix-like shell)
    - [VirtualBox](https://www.virtualbox.org/wiki/Downloads) (for running the VM)
    - [Vagrant](https://developer.hashicorp.com/vagrant/downloads) (for convenient command line tools for managing the VM)
    - [Vagrant Manager](https://www.vagrantmanager.com/downloads/) (suggested by Singularity docs; may not be necessary)
2. Create a folder to host your Singularly installation. E.g., `C:\Users\BRO7\vm-singularity`.
3. From within the folder you just created, initialize a `Vagrantfile`:
    ```bash
    vagrant init sylabs/singularity-3.7-ubuntu-bionic64
    ```
    (We use the above VM because at the time of writing this README, the cluster was running Singularity 3.7.)
4. You can edit the `Vagrantfile` to configure the VM. In particular, we want to bind local directory containing inference server code to `/triton` by adding the following line (with the first argument reflecting the correct path on your machine):
    ```bash
    config.vm.synced_folder "C:\\Users\\BRO7\\Documents\\triton", "/triton"
    ```
    You may want to increase the resources allocated to the VM:
    ```bash
    # Increase size of the primary drive
    # config.vm.disk :disk, size: "40GB", primary: true
    config.vm.provider "virtualbox" do |vb|
        # Customize the amount of memory on the VM:
        vb.memory = "2048"
        # Allocate additional CPUs
        vb.cpus = "2"
    end
    ```
5. Now, power up the VM and log in: (If asked for a password, it will be `vagrant`)
    ```bash
    vagrant up
    vagrant ssh
    ```
6. When first logged into the vagrant container, check the Singularity vesion with `singularity version`.
7. See the steps below for building the image while logging into the VM.
8. You can exit the guest OS with `exit` and then (if done) bring down the VM with `vagrant halt`.
9. If you find yourself wanting to delete the VM, you can run `vagrant destroy`. The next time you run `vagrant up` it will reconstuct the VM based on your `Vagrantfile`.

### Building the Image
The container running the inference server is defined by the `python_3.10.def` file. The purpose of the container is primarily to provide the build tools and environment necessary to install and run the `nvidia-triton` Python package (and any other packages we want). The `def` file may need to be modified (and the container rebuilt) if additional tools need to be installed to support new or updated models. Documentation for the def files can be found at <https://docs.sylabs.io/guides/3.0/user-guide/definition_files.html>.

Python libraries are handled separately (on a per-project basis) by Pipenv. This 1) reduces the image size and 2) allows for multiple applications (e.g., including client scripts) to use the same image.

PyTriton docs state that it is most rigorously tested on Ubuntu 22.04, so we choose that OS for the base image. We rely on PyTorch to install CUDA support via pip. (Singularity also provides a `--nv` flag when running the container.)

To build the image, make sure you are logged into the VM (if on Windows) and in the `/triton` directory you previously bound to the codebase on the local file system. (If on Linux with `sudo`, just `cd` to the codebase.)
You can then run:
```bash
sudo singularity build python_3.10.sif python_3.10.def
```

### Uploading the Image
Upload the built image (`python_3.10.sif`) to the cluster, replacing your username below:
```bash
rsync --info=progress1 python_3.10.sif bro7@neo.nmr.mgh.harvard.edu:/space/neo/4/sif/python_3.10.sif
```
Other tools, such as `sftp` or `scp`, may be used if you do not have `rsync` installed on your machine. Cygwin provides an `rsync` installation for Windows.

## Installing/Updating Python Libraries
**This step is only necessary if updating the Python libraries or for first-time setup of a new project.**

For this project, you can log into the cluster `cd /space/drwho/3/neurobooth/applications/triton_server/`,  then run `./install_python_libs.sh`.
The below section explains some of what is going on in this script.

**After updating the libraries, you should check the new `Pipfile` and `Pipfile.lock` files into your code repository.**

### Manual Installation / Explanation
The Python libraries needed by the image are specified in the project's `Pipfile`. 
To generate a new `Pipfile` on the cluster, first `cd` to your project directory (e.g., `/space/drwho/3/neurobooth/applications/triton_server`) and run the following command (modifying the package list as necessary):
```bash
HARG=$(pwd)  # Store the absolute path to your current directory
singularity exec -H $HARG /space/neo/4/sif/python_3.10.sif pipenv install \
numpy \
"scipy>=1.10" \
pandas \
"torch>=2.1" \
torchvision \
torchaudio \
nvidia-pytriton \
lightning \
torchmetrics \
"pydantic>=2.0" \
pyyaml \
tqdm
```
The `-H $HARG` is necessary for this to work as intended. By default, Singularity binds your home directory to the container's home directory. The `-H` option overides this behavior so that the container's home directory is bound to the specified directory on the cluster. This argument must be an absolute (e.g., no `.` or `..`) path.

You can then validate that GPU support is enabled with the following command:
```bash
singularity exec -H $HARG --nv /space/neo/4/sif/python_3.10.sif pipenv run \
python -c "import torch; print(f'GPU Enabled: {torch.cuda.is_available()}, # GPUs: {torch.cuda.device_count()}')"
```

To install local code repos (via `pip -e`), you need to make sure their location on the host file system is mounted to
the singularity container using the `--bind` argument. For example:
```bash
singularity exec \
-H $(pwd) \
--bind "/space/neo/3/neurobooth/applications/neurobooth-analysis-tools:/dep/neurobooth-analysis-tools" \
/space/neo/4/sif/python_3.10.sif \
pipenv install -e /dep/neurobooth-analysis-tools
```

## Starting the Server
To start the server, execute `./run_server.sh`.

_Note_: To run as a service the container will need to be started as a service.
See <https://docs.sylabs.io/guides/3.0/user-guide/running_services.html>

A python script can be run in the Singularity container with:
```bash
./singularity_exec.sh pipenv run python script.py
```

## Monitoring the Server
Server status can be queried via:
```bash
curl -v localhost:8000/v2/health/live
```

The stats of a model can be queried via:
```bash
curl -v localhost:8000/v2/models/model_name/ready
```