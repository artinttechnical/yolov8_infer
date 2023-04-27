docker build -t gstreamer_experiments:20.04 - < Dockerfile
docker run --rm -it --user $(id -u):$(id -g) -v /home/artint/Projects/MachineLearning/Otus2022/Project/opencv-python:/playground  gstreamer_experiments:20.04
