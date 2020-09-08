#installs docker and nvidia docker
#https://cnvrg.io/how-to-setup-docker-and-nvidia-docker-2-0-on-ubuntu-18-04/

sudo apt-get update
sudo apt-get upgrade
sudo apt-get install apt-transport-https ca-certificates curl software-properties-common -y
curl -fsSL https://download.docker.com/linux/ubuntu/gpg | sudo apt-key add -
sudo add-apt-repository "deb [arch=amd64] https://download.docker.com/linux/ubuntu $(lsb_release -cs) stable"
sudo apt-get update -y
sudo apt-get install docker-ce -y

docker -v

sudo usermod -aG docker jlim13

#su - jlim13

# nvidia docker

curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | \
 sudo apt-key add -
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | \
 sudo tee /etc/apt/sources.list.d/nvidia-docker.list
echo "Here"

 sudo apt-get update
 sudo apt-get install -y nvidia-docker2
 sudo pkill -SIGHUP dockerd

