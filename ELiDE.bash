#!/bin/bash
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
echo "About to install dependencies." &&
sudo apt-get -y update &&
sudo apt-get -y install git cython3 python3-setuptools python3-kivy python3-numpy python3-networkx libsdl2-dev libsdl2-image-dev libsdl2-ttf-dev libsdl2-mixer-dev;



    cd "`dirname "$0"`";
    git clone https://github.com/kivy/kivy.git
    cd kivy;
    USE_SDL2=1 python3 setup.py install --user;
    cd ..;
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
" >$HOME/.local/share/applications/ELiDE.desktop;
    xdg-desktop-menu forceupdate;

    touch "$LISE_PATH/.installed";

    python3 -m ELiDE;
fi;
