Development setup (on Ubuntu):

~~~~
sudo apt-get update
sudo apt-get install git
git clone https://github.com/stuhlmueller/connoisseur.git
cd connoisseur
scripts/setup-ubuntu
source ~/.bashrc
~~~~

Once the setup concludes, you can update the following dependencies:

~~~~
scripts/update-webppl
scripts/update-editor
scripts/update-viz
~~~~