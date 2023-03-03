docker build -t gstreamer_experiments - < Dockerfile
docker run --rm -it --user $(id -u):$(id -g) -v $(pwd):/playground  gstreamer_experiments
