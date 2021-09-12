# Docker
Modeled after sdc./d utility.
Use ./build.sh to build docker image.
Use ./start.sh to start and attach to the container.
Use ./attach.sh to attach to docker container from another session.
Use ./stop.sh to stop container.
Use ./remove_docker.sh to clean container and image.

During docker build make sure that base nvidia image suits
your host nvidia driver. Otherwise cuda version may conflict
and tf will switch on cpu version.
If everything is correct then you will see your GPU when you run
```
./test_tf_gpu.py
```

# Viewing replays
Make sure that your ssh session forwards \*:5000 port.
Inside docker container call ./viewer.py.
You can see replays from your ./replays folder by clicking on the links in
localhost:5000.
