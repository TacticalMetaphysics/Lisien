from pythonforandroid.recipe import CompiledComponentsPythonRecipe


class SQLAlchemyRecipe(CompiledComponentsPythonRecipe):
    name = 'sqlalchemy'
    version = '2.0.40'
    url = 'https://files.pythonhosted.org/packages/68/c3/3f2bfa5e4dcd9938405fe2fab5b6ab94a9248a4f9536ea2fd497da20525f/sqlalchemy-2.0.40.tar.gz'
    call_hostpython_via_targetpython = False

    depends = ['setuptools', 'typing_extensions']

    patches = ['zipsafe.patch']


recipe = SQLAlchemyRecipe()
