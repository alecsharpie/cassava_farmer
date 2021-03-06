# Cassava Disease Identification
## An API exposing a Tensorflow Model which predicts plant diseases from low quality cassava leaf images

# Startup the project

The initial setup.

Create virtualenv and install the project:
```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv ~/venv ; source ~/venv/bin/activate ;\
    pip install pip -U; pip install -r requirements.txt
```

Unittest test:
```bash
make clean install test
```

Check for cassava_farmer in gitlab.com/{group}.
If your project is not set please add it:

- Create a new project on `gitlab.com/{group}/cassava_farmer`
- Then populate it:

```bash
##   e.g. if group is "{group}" and project_name is "cassava_farmer"
git remote add origin git@github.com:{group}/cassava_farmer.git
git push -u origin master
git push -u origin --tags
```

Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
cassava_farmer-run
```

# Install

Go to `https://github.com/{group}/cassava_farmer` to see the project, manage issues,
setup you ssh public key, ...

Create a python3 virtualenv and activate it:

```bash
sudo apt-get install virtualenv python-pip python-dev
deactivate; virtualenv -ppython3 ~/venv ; source ~/venv/bin/activate
```

Clone the project and install it:

```bash
git clone git@github.com:{group}/cassava_farmer.git
cd cassava_farmer
pip install -r requirements.txt
make clean install test                # install and test
```
Functionnal test with a script:

```bash
cd
mkdir tmp
cd tmp
cassava_farmer-run
```
