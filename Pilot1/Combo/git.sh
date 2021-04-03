#!/bin/bash

git pull &&
git add logs/* &&
git commit -m "updated instance" &&
git push
