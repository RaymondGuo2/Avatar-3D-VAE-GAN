#!/bin/bash
for file in data/*.obj; do ~/binvox -d 64 -cb "$file"; done


