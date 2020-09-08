xhost +SI:localuser:root

localxauth_var=$(xauth list)

echo "$localxauth_var"

docker build -t research_img --build-arg local_xauth_="$localxauth_var" --build-arg local_display="$displayvar" .

docker run -it --runtime=nvidia \
       --net=host \
       --privileged \
       --env DISPLAY \
       --volume /tmp/.X11-unix:/tmp/.X11-unix \
       --volume /var/run/docker.sock:/var/run/docker.sock \
       --env XAUTHORITY=/root/.Xauthority \
       -v /playpen/john/:/app \
       -v /net/vision29/data/c/:/data \
       -v /media/:/media \
       research_img:latest bash 
       #--env NVIDIA_DRIVER_CAPABILITIES=compute,utility \
       #research_img:latest bash

#-v ${HOME}/.Xauthority:/app/.Xauthority:rw \
