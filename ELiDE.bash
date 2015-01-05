#!/bin/bash
if [ -z "$LISE_PATH" ]; then
    LISE_PATH="`dirname "$0"`/LiSE";
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

    mkfifo announce;
    mkfifo addapt;
    echo '
echo "About to install dependencies. This involves setting up two PPAs.";
echo "ppa:thopiekar/pygame";
sudo add-apt-repository -y ppa:thopiekar/pygame;
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

    git clone https://github.com/LogicalDash/LiSE.git "$LISE_PATH";
    cd "$LISE_PATH";
    git submodule init;
    git submodule update;
    python3 setup.py install --user;

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
