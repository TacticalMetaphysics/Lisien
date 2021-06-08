from abc import abstractmethod
from LiSE import Engine
from LiSE.examples import kobold
from LiSE.examples import polygons
from .util import idle_until, window_with_widget, ELiDEAppTest
from ..card import Card


class RuleBuilderTest(ELiDEAppTest):
    @abstractmethod
    def install(self, engine: Engine):
        raise NotImplementedError()

    @abstractmethod
    def get_selection(self):
        raise NotImplementedError()

    def setUp(self):
        super(RuleBuilderTest, self).setUp()
        with Engine(self.prefix) as eng:
            self.install(eng)
        app = self.app
        mgr = app.build()
        self.win = window_with_widget(mgr)
        screen = app.mainscreen
        idle_until(lambda: app.rules.rulesview, 100, 'Never made rules view')
        self.board = screen.graphboards['physical']
        idle_until(lambda: 'kobold' in self.board.pawn, 100,
                   'never got the pawn for the kobold')
        app.selection = self.get_selection()
        screen.charmenu.charmenu.toggle_rules()
        rules = app.rules
        rules_box = rules.children[0]
        idle_until(lambda: 'ruleslist' in rules_box.ids, 100,
                   'Never made rules list')
        self.rules_list = rules_list = rules_box.ids.ruleslist
        self.rules_view = rules_box.ids.rulesview
        idle_until(lambda: rules_list.children[0].children, 100,
                   'Never filled rules list')


class TestRuleBuilderKobold(RuleBuilderTest):
    def install(self, engine: Engine):
        kobold.inittest(engine)

    def get_selection(self):
        return self.board.pawn['kobold']

    def test_rule_builder_display_trigger(self):
        rules_list = self.rules_list
        rules_view = self.rules_view
        idle_until(lambda: 'shrubsprint' in {
            rulebut.text for rulebut in rules_list.children[0].children}, 100,
                   'Never made shrubsprint button')
        for rulebut in rules_list.children[0].children:
            if rulebut.text == 'shrubsprint':
                rulebut.state = 'down'
                break
        idle_until(lambda: rules_view.children)
        idle_until(lambda: hasattr(rules_view, '_trigger_tab'), 100,
                   'Never made trigger tab')
        builder = rules_view._trigger_builder
        idle_until(lambda: builder.children, 100,
                   'Never filled trigger builder')
        card_names = {card.headline_text for card in builder.children
                      if isinstance(card, Card)}
        assert card_names == {'standing_still', 'aware', 'uncovered',
                              'sametile', 'breakcover', 'kobold_alive'}


class TestCharRuleBuilder(ELiDEAppTest):
    def setUp(self):
        super(TestCharRuleBuilder, self).setUp()
        with Engine(self.prefix) as eng:
            polygons.install(eng)
        app = self.app
        mgr = app.build()
        self.win = window_with_widget(mgr)
        app.select_character(app.engine.character['triangle'])
        idle_until(lambda: app.character_name == 'triangle', 100,
                   'Never changed character')
        app.mainscreen.charmenu.charmenu.toggle_rules()
        idle_until(lambda: getattr(app.charrules, '_finalized', False), 100,
                   'Never finalized')

    def test_char_rule_builder_display_avatar_trigger(self):
        app = self.app
        tabitem = app.charrules._avatar_tab
        idle_until(lambda: tabitem.content, 100,
                   'avatar tab never got content')
        app.charrules._tabs.switch_to(tabitem)
        idle_until(lambda: app.charrules._tabs.current_tab == tabitem, 100,
                   'Never switched tab')
        rules_box = app.charrules._avatar_box
        idle_until(lambda: rules_box.rulesview.children, 100,
                   'Never filled rules view')
        rules_list = rules_box.ruleslist
        idle_until(lambda: rules_list.children[0].children, 100,
                   'Never filled rules list')
        idle_until(lambda: 'relocate' in {
            rulebut.text for rulebut in rules_list.children[0].children}, 100,
                   'Never made relocate button')
        for rulebut in rules_list.children[0].children:
            if rulebut.text == 'relocate':
                rulebut.state = 'down'
                break
        builder = rules_box.rulesview._trigger_builder
        idle_until(lambda: builder.children, 100,
                   'Never filled trigger builder')
        card_names = {card.headline_text for card in builder.children
                      if isinstance(card, Card)}
        assert card_names == {'similar_neighbors', 'dissimilar_neighbors'}
