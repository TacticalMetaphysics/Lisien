import unittest
from unittest.mock import MagicMock
from networkx import DiGraph
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

    def test_bind_char_map(self):
        """Test binding to the CharacterMapping, and to a specific character"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        self.engine.character.listener(general)
        self.engine.character.listener(specific, 'spam')
        self.engine.character.listener(inert, 'ham')
        self.engine.character['spam'] = DiGraph(eggs=True)
        general.assert_called_once_with(
            self.engine.character,
            'spam',
            self.engine.character['spam']
        )
        specific.assert_called_once_with(
            self.engine.character,
            'spam',
            self.engine.character['spam']
        )
        self.assertEqual(inert.call_count, 0)
        self.assertTrue(self.engine.character['spam'].stat['eggs'])

    def test_bind_char_thing(self):
        """Test binding to a character's thing mapping, and to a specific
        thing

        """
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        self.engine.character['spam'] = DiGraph(eggs=True)
        char = self.engine.character['spam']
        self.assertTrue(char.stat['eggs'])
        # I have to put the thing someplace
        char.place['plate'] = {'flat': True}
        char.thing.listener(general)
        char.thing.listener(specific, 'baked_beans')
        char.thing['baked_beans'] = {'location': 'plate'}
        th = char.thing['baked_beans']
        self.assertEqual(th['location'], 'plate')
        general.assert_called_once_with(
            char.thing,
            'baked_beans',
            th
        )
        specific.assert_called_once_with(
            char.thing,
            'baked_beans',
            th
        )
        self.assertEqual(inert.call_count, 0)

    def test_bind_char_place(self):
        """Test binding to the place mapping of a character"""
        self.engine.character['spam'] = DiGraph()
        ch = self.engine.character['spam']
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        ch.place.listener(general)
        ch.place.listener(place='plate')(specific)
        ch.place.listener(inert, 'floor')
        ch.place['plate'] = {'flat': True}
        pl = ch.place['plate']
        self.assertTrue(pl['flat'])
        general.assert_called_once_with(
            ch.place,
            'plate',
            pl
        )
        specific.assert_called_once_with(
            ch.place,
            'plate',
            pl
        )
        self.assertEqual(inert.call_count, 0)

    def test_bind_char_portal(self):
        """Test binding to character's portal mapping"""
        self.engine.character['spam'] = DiGraph()
        ch = self.engine.character['spam']
        ch.place['kitchen'] = {'smell': 'yummy'}
        ch.place['porch'] = {'rustic': True}
        nodeA = ch.place['kitchen']
        nodeB = ch.place['porch']
        generalA = MagicMock()
        specificA = MagicMock()
        inert = MagicMock()
        ch.portal.listener(generalA)
        ch.portal.listener(specificA, 'kitchen')
        ch.portal.listener(inert, 'living_room')
        generalB = MagicMock()
        specificB = MagicMock()
        ch.portal['kitchen'].listener(generalB)
        ch.portal['kitchen'].listener(specificB, 'porch')
        ch.portal['kitchen'].listener(inert, 'balcony')
        ch.portal['kitchen']['porch'] = {'locked': False}
        port = ch.portal['kitchen']['porch']
        generalB.assert_called_once_with(
            ch.portal['kitchen'],
            nodeA,
            nodeB,
            port
        )
        specificB.assert_called_once_with(
            ch.portal['kitchen'],
            nodeA,
            nodeB,
            port
        )
        self.assertFalse(port['locked'])
        generalA.assert_called_once_with(
            ch.portal,
            nodeA,
            nodeB,
            port
        )
        specificA.assert_called_once_with(
            ch.portal,
            nodeA,
            nodeB,
            port
        )
        self.assertEqual(inert.call_count, 0)

    def test_bind_char_avatar(self):
        """Test binding to ``add_avatar``"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        if 'a' not in self.engine.character:
            self.engine.add_character('a')
        if 'b' not in self.engine.character:
            self.engine.add_character('b')
        chara = self.engine.character['a']
        charb = self.engine.character['b']
        chara.avatar_listener(general)
        chara.avatar_listener(specific, 'b')
        chara.avatar_listener(inert, 'z')
        pl = charb.new_place('q')
        chara.add_avatar(pl)
        general.assert_called_once_with(
            chara,
            charb,
            pl,
            True
        )
        specific.assert_called_once_with(
            chara,
            charb,
            pl,
            True
        )
        chara.del_avatar(pl)
        general.assert_called_with(
            chara,
            charb,
            pl,
            False
        )
        specific.assert_called_with(
            chara,
            charb,
            pl,
            False
        )
        charc = self.engine.new_character('c')
        plc = charc.new_place('c')
        chara.add_avatar(plc)
        general.assert_called_with(
            chara,
            charc,
            plc,
            True
        )
        self.assertEqual(general.call_count, 3)
        self.assertEqual(specific.call_count, 2)
        self.assertEqual(inert.call_count, 0)

    def test_bind_rule(self):
        """Test binding to a rulemap and a rule therein"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        if 'a' not in self.engine.character:
            self.engine.add_character('a')
        char = self.engine.character['a']
        char.rule.listener(general)
        char.rule.listener(specific, 'spam')
        char.rule.listener(inert, 'eggs')

        @char.rule
        def spam(*args):
            pass

        general.assert_called_once_with(
            char.rule,
            spam,
            True
        )
        specific.assert_called_once_with(
            char.rule,
            spam,
            True
        )
        del char.rule['spam']
        general.assert_called_with(
            char.rule,
            spam,
            False
        )
        specific.assert_called_with(
            char.rule,
            spam,
            False
        )
        self.engine.rule.listener(general)
        self.engine.rule.listener(specific, 'ham')
        self.engine.rule.listener(inert, 'eggs')
        @self.engine.rule
        def ham(*args):
            pass
        general.assert_called_with(
            self.engine.rule,
            ham,
            True
        )
        specific.assert_called_with(
            self.engine.rule,
            ham,
            True
        )
        del self.engine.rule['ham']
        general.assert_called_with(
            self.engine.rule,
            ham,
            False
        )
        specific.assert_called_with(
            self.engine.rule,
            ham,
            False
        )
        @self.engine.rule
        def baked_beans(*args):
            pass
        general.assert_called_with(
            self.engine.rule,
            baked_beans,
            True
        )
        self.assertEqual(general.call_count, 5)
        self.assertEqual(specific.call_count, 4)
        self.assertEqual(inert.call_count, 0)

    def test_bind_rule_funlist(self):
        """Test binding to each of the function lists of a rule"""
        trig = MagicMock()
        preq = MagicMock()
        act = MagicMock()
        if 'a' not in self.engine.character:
            self.engine.add_character('a')
        ch = self.engine.character['a']

        @ch.rule
        def nothing(*args):
            pass

        nothing.triggers.listener(trig)
        nothing.prereqs.listener(preq)
        nothing.actions.listener(act)

        def something(*args):
            pass

        nothing.trigger(something)
        nothing.prereq(something)
        nothing.action(something)
        trig.assert_called_once_with(nothing.triggers)
        preq.assert_called_once_with(nothing.prereqs)
        act.assert_called_once_with(nothing.actions)

    def test_bind_place_stat(self):
        """Test binding to one of a place's stats, and to all of them"""
        general = MagicMock()
        specific = MagicMock()
        inert = MagicMock()
        char = self.engine.new_character('c')
        pl = char.new_place('p')
        pl.listener(general)
        pl.listener(stat='spam')(specific)
        pl.listener(inert, 'eggs')
        pl['spam'] = 'tasty'
        general.assert_called_once_with(
            pl,
            'spam',
            'tasty'
        )
        specific.assert_called_once_with(
            pl,
            'spam',
            'tasty'
        )
        pl['baked_beans'] = 'tastier'
        general.assert_called_with(
            pl,
            'baked_beans',
            'tastier'
        )
        del pl['spam']
        general.assert_called_with(
            pl,
            'spam',
            None
        )
        specific.assert_called_with(
            pl,
            'spam',
            None
        )
        self.assertEqual(general.call_count, 3)
        self.assertEqual(specific.call_count, 2)
        self.assertEqual(inert.call_count, 0)

if __name__ == '__main__':
    unittest.main()
