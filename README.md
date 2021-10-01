# Docker
Modeled after out sdc d utility.
Use ./d -h to see help
Use ./d -b to build docker
Use ./d -u to start container
Use ./d -a to attach to container
Use ./d -s to stop container
Use ./d -ua to start and attach to container

During docker build make sure that base nvidia image suits
your host nvidia driver. Otherwise cuda version may conflict
and tf will switch on cpu version.
If everything is correct then you will see your GPU when you run
```
./infra/test_tf_gpu.py
```

# Directory structure
* Lux-Design-2021 - holds competition repo
* viewer - holds official replay viewer code with few custom tweaks
* src - where bot/our team competitions code goes
* infra - put there any utility scripts non related to bot logic such as installation utils. Or put there scripts to run on the host system, such as tensorboard. TODO move Docker and requirements.txt there.

# Viewing replays
Make sure that your ssh session forwards \*:5000 port (override with --port).
Inside docker container call ./viewer.py.
You can see replays from your /tmp/replays (override with --replays) directory by clicking on the links in
localhost:5000.

# Run example
TODO
