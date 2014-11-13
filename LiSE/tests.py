import unittest
from unittest.mock import MagicMock
from LiSE.core import Engine


class BindingTestCase(unittest.TestCase):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        if 'worlddb' not in kwargs:
            kwargs['worlddb'] = ':memory:'
        if 'codedb' not in kwargs:
            kwargs['codedb'] = ':memory:'
        self.kwargs = kwargs

    def setUp(self):
        self.engine = Engine(**self.kwargs)

    def tearDown(self):
        self.engine.close()

    def test_bind_string(self):
        """Test binding to the string store, and to a particular string"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()

        # these would normally be called using decorators but I don't
        # think I can do mocks that way
        self.engine.string.listener(general)
        self.engine.string.listener(specific, 'spam')
        self.engine.string.listener(string='ham')(inert)

        self.engine.string['spam'] = 'eggs'
        general.assert_called_once_with(self.engine.string, 'spam', 'eggs')
        specific.assert_called_once_with(self.engine.string, 'spam', 'eggs')
        general = MagicMock()
        specific = MagicMock()
        self.engine.string.listener(general)
        self.engine.string.listener(string='spam')(specific)
        del self.engine.string['spam']
        general.assert_called_once_with(self.engine.string, 'spam', None)
        specific.assert_called_once_with(self.engine.string, 'spam', None)
        self.assertEqual(inert.call_count, 0)
        bound = MagicMock()
        self.engine.string.lang_listener(bound)
        self.engine.string.language = 'jpn'
        bound.assert_called_once_with(self.engine.string, 'jpn')

    def test_bind_func_store(self):
        """Test binding to the function store, and to a specific function
        name

        """
        for store in self.engine.stores:
            general = MagicMock()
            specific = MagicMock()
            inert = MagicMock()
            getattr(self.engine, store).listener(general)
            getattr(self.engine, store).listener(name='spam')(specific)
            getattr(self.engine, store).listener(name='ham')(inert)

            def nothing():
                pass
            getattr(self.engine, store)['spam'] = nothing
            general.assert_called_once_with(
                getattr(self.engine, store),
                'spam',
                nothing
            )
            specific.assert_called_once_with(
                getattr(self.engine, store),
                'spam',
                nothing
            )
            del getattr(self.engine, store)['spam']
            general.assert_called_with(
                getattr(self.engine, store),
                'spam',
                None
            )
            specific.assert_called_with(
                getattr(self.engine, store),
                'spam',
                None
            )
            self.assertEqual(general.call_count, 2)
            self.assertEqual(specific.call_count, 2)
            self.assertEqual(inert.call_count, 0)

    def test_bind_univ_var(self):
        """Test binding to the universal variable store, and to a specific
        var

        """
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        self.engine.universal.listener(general)
        self.engine.universal.listener(key='spam')(specific)
        self.engine.universal.listener(key='ham')(inert)
        self.engine.universal['spam'] = 'eggs'
        general.assert_called_once_with(
            self.engine.universal,
            'spam',
            'eggs'
        )
        specific.assert_called_once_with(
            self.engine.universal,
            'spam',
            'eggs'
        )
        self.assertEqual(inert.call_count, 0)

if __name__ == '__main__':
    unittest.main()
