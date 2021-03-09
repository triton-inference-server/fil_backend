#!/bin/bash
cmake -DCMAKE_INSTALL_PREFIX:PATH="${PWD}/install" ..
make install
