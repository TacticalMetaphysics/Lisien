texturestack
============
Widget for compositing textures, like virtual paper dolls.

"Virtual paper doll" is a technique for building sprite graphics used in, eg., the player avatar in Dungeon Crawl Stone Soup. Basically, you just put the sprites one on top of the other, and then move them around like they're one sprite. You might do this with plain Image widgets, but I found it awkward to manage the exact order of the various subwidgets, so I made this widget to handle them for me.

Includes two classes: TextureStack proper requires a list of kivy Texture objects; ImageStack is happy with a list of paths to loadable images. For demonstration purposes I have included [the ProcJam 2015 art pack](http://www.procjam.com/2015/09/01/procjam-art-pack-now-available/) by Marsh Davies, available under [Creative Commons By-NC](https://creativecommons.org/licenses/by-nc/4.0/)

In case of my death, I, Zachary Spector, wish for this code to be relicensed under [CC0](https://creativecommons.org/choose/zero/).
