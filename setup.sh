#!/bin/bash
apt-get install python3-pip
pip3 install matplotlib numpy pandas scikit-learn scipy sklearn tabulate
wget "https://raw.githubusercontent.com/bpacke/903/master/generated_datasets.py"
wget "https://raw.githubusercontent.com/bpacke/903/master/algorithm_test.py"
mkdir test_output
clear
echo "Don't forget to pip3 install any other packages you may need"
echo "You need to wget your code"
echo "Use nano algorithm_test.py to import your algorithm"
echo "Run the test with python3 algorithm_test.py"
