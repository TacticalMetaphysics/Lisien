from pythonforandroid.recipe import CythonRecipe


class MsgPackRecipe(CythonRecipe):
    version = '1.1.0'
    url = 'https://files.pythonhosted.org/packages/cb/d0/7555686ae7ff5731205df1012ede15dd9d927f6227ea151e901c7406af4f/msgpack-1.1.0.tar.gz'
    depends = ["setuptools"]
    call_hostpython_via_targetpython = False


recipe = MsgPackRecipe()
