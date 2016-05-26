[![Build Status](https://travis-ci.org/stuhlmueller/play.svg?branch=gh-pages)](https://travis-ci.org/stuhlmueller/play)

Development setup (on Ubuntu):

~~~~
sudo apt-get update
sudo apt-get install git
git clone https://github.com/stuhlmueller/play.git
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
