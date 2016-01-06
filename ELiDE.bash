#!/bin/bash
if [ -z "$LISE_PATH" ]; then
    LISE_PATH="`dirname "$0"`/LiSE";
fi;

if [ -e "$LISE_PATH" ] && [ -f "$LISE_PATH/.installed" ]; then
    cd "$LISE_PATH";
    git pull;
    git submodule update;
    python3 -mpip install --user --upgrade LiSE/ ELiDE/;
    python3 -m ELiDE;
else
    if [ -e "$LISE_PATH" ]; then
        # clean out failed installation
        rm -rf "$LISE_PATH";
    fi;
    if [ -n "`uname -v | grep Ubuntu`" ]; then
        # on Ubuntu-derived distros, install from PPA
        mkfifo announce;
        mkfifo addapt;
        echo '
echo "About to install dependencies. This involves setting up the kivy-daily PPA.";
echo "ppa:kivy-team/kivy-daily";
sudo add-apt-repository -y ppa:kivy-team/kivy-daily;
echo "Updating package lists.";
sudo apt-get -y update;
echo "Installing dependencies.";
sudo apt-get -y install git cython3 python3-dev python3-setuptools python3-kivy;
echo "All dependencies installed." >announce;
sleep 1;
exit;' >addapt &
        MYTERM="`which gnome-terminal` -x";
        if [ "$MYTERM" == " -x" ]; then
            MYTERM="`which xfce4-terminal` -x";
        fi;
        if [ "$MYTERM" == " -x" ]; then
            MYTERM="`which lxterminal` -e";
        fi;
        if [ "$MYTERM" == " -e" ]; then
            MYTERM="`which konsole` -e";
        fi;
        if [ "$MYTERM" == " -e" ]; then
            MYTERM="`which xterm` -e" ;
        fi;
        if [ "$MYTERM" == " -e" ]; then
            MYTERM="";
        fi;
        $MYTERM bash --rcfile addapt;

        echo <announce;
        rm addapt;
        rm announce;
    else
        VENV=""
        if [ -n "`which virtualenv`" ]; then
            VENV="virtualenv -p python3";
        fi;
        if [ -n "`which pyvenv`" ]; then
            VENV=pyvenv;
        fi;
        if [ -z "$VENV" ]; then
            echo "Couldn't find a usable virtualenv binary. Please install python3-venv and retry.";
            exit 1;
        fi;
        if [ -z "`which git`" ]; then
            echo "Couldn't find git. Please install it and retry."
            exit 1;
        fi;
        echo "Creating LiSE-virtualenv";
        $VENV LiSE-virtualenv;
        echo "Activating LiSE-virtualenv";
        source LiSE-virtualenv/bin/activate;
        echo "Installing Cython";
        python3 -mensurepip
        python3 -mpip install --user cython;
        echo "Getting latest Kivy master";
        git clone https://github.com/kivy/kivy.git kivy;
        WORKINGDIR=$PWD;
        cd kivy;
        echo "Installing Kivy with SDL2 backend.";
        echo "Assuming you have development headers for:";
        echo "libsdl2";
        echo "libsdl2-ttf";
        echo "libsdl2-image";
        echo "libsdl2-mixer";
        USE_SDL2=1 python3 -mpip install --user .;
        cd $WORKINGDIR;
    fi;

    git clone https://github.com/LogicalDash/LiSE.git "$LISE_PATH";
    cd "$LISE_PATH";
    git submodule init;
    git submodule update;
    python3 -mpip install --user LiSE/ ELiDE/;

    mkdir -p $HOME/.local/share/applications;
    echo "[Desktop Entry]
Comment=Development environment for LiSE
Exec=env LISE_PATH=\"$LISE_PATH\" \"`dirname \"$0\"`/ELiDE\"
Name=ELiDE
Type=Application
Categories=Development;
" >$HOME/.local/share/applications/ELiDE.desktop;
    xdg-desktop-menu forceupdate;

    touch "$LISE_PATH/.installed";

    python3 -m ELiDE;
fi
