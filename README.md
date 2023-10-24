# triton
Triton server for serving model inference requests.
See: <https://triton-inference-server.github.io/pytriton/0.3.1/>

## Setup
The inference server runs in a Singularity container. Images cannot be built on the cluster without `sudo` privleges. Instead, the image can be built on a development machine and uploaded to the cluster. See the [Singularity docs](https://docs.sylabs.io/guides/3.0/user-guide/installation.htm) for OS-dependent installation on your local machine; these steps are enumerated below for Windows.

Besides the instructions below, you can also refer to this internal Martinos help page: <https://www.nmr.mgh.harvard.edu/martinos/userInfo/computer/docker.php>

### Singularity Installation on Windows
1. Install the following:
    - [Git for Windows](https://gitforwindows.org/) and/or [Cygwin](https://www.cygwin.com/) (for a Unix-like shell)
    - [VirtualBox](https://www.virtualbox.org/wiki/Downloads)
    - [Vagrant](https://developer.hashicorp.com/vagrant/downloads)
    - [Vagrant Manager](https://www.vagrantmanager.com/downloads/)
2. Create a folder to host your Singularly installation. E.g., `C:\Users\BRO7\vm-singularity`.
3. Run the following commands to install, start, and `ssh` into a virtual machine with Singularity installed. (Make sure your shell path is in the folder you just created. We use the following VM because at the time of writing this README, the cluster was running Singularity 3.7.)
    ```bash
    export VM=sylabs/singularity-3.7-ubuntu-bionic64
    vagrant init "$VM"
    vagrant up
    vagrant ssh
    ```
    If asked for a password during the `ssh`, it will be `vagrant`.
4. While logged into the vagrant container, check the Singularity vesion with `singularity version`.
5. You can exit the guest OS with `exit` and then (if done) bring down the VM with `vagrant halt`.
6. You can edit the `Vagrantfile` (while the container is down) to specify which additional directories are mounted to the guest OS as needed. In particular, the path to the inference server code on the local machine can mounted to `/triton` by adding the following line (with the first argument reflecting the correct path on your machine):
    ```bash
    config.vm.synced_folder "C:\\Users\\BRO7\\Documents\\triton", "/triton"
    ```

### Building the Image
The container running the inference server is defined by the `triton_server.def` file. The purpose of the container is primarily to provide the build tools and environment necessary to install and run the `nvidia-triton` Python package. The `def` file may need to be modified (and the container rebuilt) if additional libraries and tools need to be installed to support new or updated models. Documentation for the def files can be found at <https://docs.sylabs.io/guides/3.0/user-guide/definition_files.html>.

To build the image, you can run `sudo singularity build triton_server.sif triton_server.def`. (Note: This is from within the Singularity VM if on Windows.)

### Uploading the Image
Upload the built image to the cluster (replacing your username as appropriate):
```bash
rsync triton_server.sif bro7@billnted.nmr.mgh.harvard.edu:/space/billnted/3/neurobooth/applications/triton_server/triton_server.sif
```
(Other tools, such as sftp, may also be used if you do not have rsync installed on your machine. Cygwin provides an rsync installation for Windows.)

## Starting the Server

### First time setup.
As per the [Martinos Docs](https://www.nmr.mgh.harvard.edu/martinos/userInfo/computer/docker.php), it is important create a symlink for the `~/.singularity` folder, otherwise Singularity will generate many GBs of data that surpass the user home quota. When writing this document, the `/space/billnted/1/singularity` directory was created for this purpose.

If you previously used Singularity, first delete the folder in your home directory:
```bash
rm -rf ~/.singularity
```

Then, create a symlink to `/space/billnted/1/singularity`:
```bash
ln -s /space/billnted/1/singularity ~/.singularity
```

### Start the Inference Server as a Service

