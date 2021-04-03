#!/bin/bash

TESTCASE=$1  #"P100_noshuffle_resnet152"

python candle_inf.py --testcase $TESTCASE &&
python resnet_inf.py --testcase $TESTCASE &&
python vgg_inf.py --testcase $TESTCASE &&
./git.sh 

