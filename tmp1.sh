#!/bin/bash

git pull
./rebuild.sh
cd build/AI || exit
./ai_pass_selector
cd ../..
