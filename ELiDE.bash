#!/bin/bash
export SHELL=/bin/bash;
if [ -z "$LISE_PATH" ]; then
    LISE_PATH="`dirname "$0"`/LiSE";
    export LISE_PATH;
fi;

if [ -e "$LISE_PATH" ] && [ -f "$LISE_PATH/.installed" ]; then
    cd "$LISE_PATH";
    git pull;
    git submodule update;
    python3 setup.py install --user --upgrade;
    python3 -m ELiDE;
else
    if [ -e "$LISE_PATH" ]; then
        # clean out failed installation
        rm -rf "$LISE_PATH";
    fi;
    xterm -e 'bash -s <<EOF
echo "About to install dependencies." &&
sudo add-apt-repository -y ppa:thopiekar/pygame &&
sudo add-apt-repository -y ppa:kivy-team/kivy-daily &&
sudo apt-get -y update &&
sudo apt-get -y install git virtualenv python3-kivy python3-numpy python3-networkx;
EOF'

    cd "`dirname "$0"`";
    git clone https://github.com/LogicalDash/LiSE.git;
    cd LiSE;
    git submodule init;
    git submodule update;
    python3 setup.py install --user;

    echo "[Desktop Entry]
Comment=Development environment for LiSE
Exec=\"`dirname \"$0\"`/ELiDE\"
Name=ELiDE
Type=Application
Categories=Development;
" >$HOME/.local/share/applications;
    xdg-desktop-menu forceupdate;

    touch "$LISE_PATH/.installed";

    python3 -m ELiDE;
fi;
