# !/bin/bash

COLUMNS=$(tput cols) 
title=". : PYRON INSTALLATION : ."    
printf "\n\n\n\n""%*s\n\n\n\n" $(((${#title}+$COLUMNS)/2)) "$title"
sleep 1
if [[ "pip3" == *"$(which pip)"* ]]; then
    echo 'python version: 3.x found use pip3 installer'
    pip3 install .
    pip='pip3'
elif [[ "pip" == *"$(which pip)"* ]]; then
    echo 'use pip installer'
    pip2 install .
    pip='pip'
fi

# parse the path
str="$($pip show pyron)"
str="${str/*Location: /}"
p="${str/Requires:*/}"
echo 'Location: '$p

# export to paths
printf 'export... '
export PATH=$PATH:$p'/pyron'
export PYTHONPATH=$PYTHONPATH:$p
echo 'done'