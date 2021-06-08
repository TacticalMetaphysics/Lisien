from LiSE import Engine
from LiSE.examples import kobold
from LiSE.examples import polygons
from .util import idle_until, window_with_widget, ELiDEAppTest
from ..card import Card


class TestRuleBuilder(ELiDEAppTest):
    def test_rule_builder_display(self):
        with Engine(self.prefix) as eng:
            kobold.inittest(eng)
        app = self.app
        mgr = app.build()
        win = window_with_widget(mgr)
        screen = app.mainscreen
        idle_until(lambda: app.rules.rulesview, 100, 'Never made rules view')
        board = screen.graphboards['physical']
        idle_until(lambda: 'kobold' in board.pawn, 100,
                   'never got the pawn for the kobold')
        app.selection = board.pawn['kobold']
        screen.charmenu.charmenu.toggle_rules()
        rules = app.rules
        rules_box = rules.children[0]
        idle_until(lambda: 'ruleslist' in rules_box.ids, 100,
                   'Never made rules list')
        rules_list = rules_box.ids.ruleslist
        rules_view = rules_box.ids.rulesview
        idle_until(lambda: rules_list.children[0].children, 100,
                   'Never filled rules list')
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

    def test_char_rule_builder_display(self):
        with Engine(self.prefix) as eng:
            polygons.install(eng)
        app = self.app
        mgr = app.build()
        win = window_with_widget(mgr)
        app.select_character(app.engine.character['triangle'])
        idle_until(lambda: app.character_name == 'triangle', 100,
                   'Never changed character')
        app.mainscreen.charmenu.charmenu.toggle_rules()
        idle_until(lambda: getattr(app.charrules, '_finalized', False), 100,
                   'Never finalized')
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
