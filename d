#!/usr/bin/env bash

attach() {
  docker exec -w /lux_ai \
    --user $USER \
    -i \
    -t \
    lux_ai \
    /bin/bash
}

start() {
  # 5000 is flask port
  # 6007 and 6006 for tensorboard
  # 8888 jupyter port
  # rest don't remember
  docker run \
    -d \
    -it \
    --gpus all \
    -v /tmp/lux_ai:/tmp/lux_ai \
    -v `pwd`:/lux_ai \
    -p 8888:8888 \
    -p 0.0.0.0:6006:6006 \
    -p 6007:6007 \
    -p 8000:8000 \
    -p 5000:5000 \
    -p 3000:3000 \
    --name lux_ai \
    --ipc=host \
    lux_ai
}

build() {
  docker build `dirname $(realpath $0)` \
    -t lux_ai \
    --build-arg \
    UID=$(id -u) \
    --build-arg \
    GID=$(id -g) \
    --build-arg USER=$USER
}

stop() {
  docker stop lux_ai
  docker rm lux_ai
}

remove_docker() {
  docker rmi lux_ai
}

status() {
  docker ps | grep lux_ai | cat
}

print_help() {
  # using here doc feature
  # https://stackoverflow.com/a/23930212
  cat << END
usage: d [-h] [-u] [-a] [-s] [-b] [-c] 

Script to control docker
    
optional arguments:
  -h, --help            show this help message and exit

commands:
  Various commands for ./d. Could be combined (e.g. ./d -sua)

  -u, --up              Start new docker container
  -a, --attach          Attach (start interactive shell) to running container
  -s, --stop            Stop container if running
  -b, --build           Build new docker images
  -q, --status          Query container status
  -c, --cleanup         Cleanup docker container files. Useful when you change build.
END
}

main() {
  # modeled after our sdc d utility
  # parse command line arguments
  # Combines this tutorials:
  # https://sookocheff.com/post/bash/parsing-bash-script-arguments-with-shopts/ 
  # https://wiki.bash-hackers.org/howto/getopts_tutorial
  while getopts ":huabscq" opt; do
    case ${opt} in
      h )
        print_help
        exit 0
        ;;
      u )
        echo "Start container"
        start 
        ;;
      a )
        echo "Attach to container"
        attach
        ;;
      b )
        echo "Build container"
        build
        ;;
      s )
        echo "Stop container"
        stop
        ;;
      c )
        echo "Remove docker"
        remove_docker   
        ;;
      q )
        echo "Query container status"
        status
        ;;
      \? )
        echo "Invalid Option: -${OPTARG}" 1>&2
        exit 1
        ;;
    esac
  done
}

main "$@"
