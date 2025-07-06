# This file is part of Elide, frontend to Lisien, a framework for life simulation games.
# Copyright (c) Zachary Spector, public@zacharyspector.com
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Affero General Public License as published by
# the Free Software Foundation, version 3.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Affero General Public License for more details.
#
# You should have received a copy of the GNU Affero General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
import json
import os
import shutil

from kivy.clock import Clock, triggered
from kivy.core.image import Image
from kivy.logger import Logger
from kivy.properties import (
	ListProperty,
	NumericProperty,
	ObjectProperty,
	StringProperty,
)
from kivy.resources import resource_find
from kivy.uix.boxlayout import BoxLayout
from kivy.uix.button import Button
from kivy.uix.filechooser import FileChooserIconView
from kivy.uix.label import Label
from kivy.uix.modalview import ModalView
from kivy.uix.screenmanager import Screen
from kivy.uix.scrollview import ScrollView
from sqlalchemy import and_, bindparam, column

from .kivygarden.texturestack import ImageStack
from .pallet import Pallet, PalletBox
from .util import logwrap, store_kv


def trigger(func):
	return triggered()(func)


class SpriteSelector(BoxLayout):
	prefix = StringProperty()
	pallets = ListProperty()
	imgpaths = ListProperty([])
	default_imgpaths = ListProperty()
	preview = ObjectProperty()

	@logwrap(section="SpriteSelector")
	def on_prefix(self, *_):
		if "textbox" not in self.ids:
			Clock.schedule_once(self.on_prefix, 0)
			return
		self.ids.textbox.text = self.prefix

	@logwrap(section="SpriteSelector")
	def on_imgpaths(self, *_):
		if not self.preview:
			Logger.debug("SpriteSelector: no preview")
			Clock.schedule_once(self.on_imgpaths, 0)
			return
		if hasattr(self, "_imgstack"):
			self._imgstack.paths = self.imgpaths
		else:
			self._imgstack = ImageStack(paths=self.imgpaths)
			self._imgstack.bind(
				pos=self._position_imgstack, size=self._position_imgstack
			)
			self.preview.add_widget(self._imgstack)

	@trigger
	@logwrap(section="SpriteSelector")
	def _position_imgstack(self, *_):
		self._imgstack.x = self.preview.center_x - self._imgstack.height / 2
		self._imgstack.y = self.preview.center_y - self._imgstack.width / 2

	@logwrap(section="SpriteSelector")
	def on_pallets(self, *_):
		for pallet in self.pallets:
			pallet.fbind("selection", self._upd_imgpaths)

	@logwrap(section="SpriteSelector")
	def _upd_imgpaths(self, *_):
		imgpaths = []
		for pallet in self.pallets:
			if pallet.selection:
				for selected in pallet.selection:
					imgpaths.append(
						"atlas://{}/{}".format(pallet.filename, selected.text)
					)
		self.imgpaths = imgpaths if imgpaths else self.default_imgpaths


class SpriteBuilder(ScrollView):
	prefix = StringProperty()
	imgpaths = ListProperty()
	default_imgpaths = ListProperty()
	data = ListProperty()
	labels = ListProperty()
	pallets = ListProperty()

	def __init__(self, **kwargs):
		super().__init__(**kwargs)
		self.bind(data=self._trigger_update)

	@logwrap(section="SpriteBuilder")
	def update(self, *args):
		if self.data is None:
			return
		if not self.canvas:
			Clock.schedule_once(self.update, 0)
			return
		if not hasattr(self, "_palbox"):
			self._palbox = PalletBox(orientation="vertical", size_hint_y=None)
			self.add_widget(self._palbox)
		else:
			self._palbox.clear_widgets()
		if hasattr(self._palbox, "_bound_width"):
			for uid in self._palbox._bound_width:
				self._palbox.unbind_uid("width", uid)
			del self._palbox._bound_width
		self.labels = []
		for pallet in self.pallets:
			if hasattr(pallet, "_bound_minimum_height"):
				pallet.unbind_uid(
					"minimum_height", pallet._bound_minimum_height
				)
				del pallet._bound_minimum_height
			if hasattr(pallet, "_bound_height"):
				pallet.unbind_uid("height", pallet._bound_height)
				del pallet._bound_height
		self.pallets = []
		for text, filename in self.data:
			label = Label(text=text, size_hint=(None, None), halign="center")
			label.texture_update()
			label.height = label.texture.height
			label.width = self._palbox.width
			pallet = Pallet(filename=filename, size_hint=(None, None))
			pallet.width = self._palbox.width
			self._palbox._bound_width = [
				self._palbox.fbind("width", label.setter("width")),
				self._palbox.fbind("width", pallet.setter("width")),
			]
			pallet.height = pallet.minimum_height
			pallet._bound_minimum_height = (
				pallet.fbind("minimum_height", pallet.setter("height")),
			)
			pallet._bound_height = pallet.fbind(
				"height", self._trigger_reheight
			)
			self.labels.append(label)
			self.pallets.append(pallet)
		n = len(self.labels)
		assert n == len(self.pallets)
		for i in range(0, n):
			self._palbox.add_widget(self.labels[i])
			self._palbox.add_widget(self.pallets[i])

	_trigger_update = trigger(update)

	@logwrap(section="SpriteBuilder")
	def reheight(self, *args):
		self._palbox.height = sum(
			wid.height for wid in self.labels + self.pallets
		)

	_trigger_reheight = trigger(reheight)


class SpriteDialog(BoxLayout):
	toggle = ObjectProperty()
	prefix = StringProperty()
	imgpaths = ListProperty()
	custom_imgs_header = StringProperty()
	custom_imgs_dir = StringProperty()
	default_imgpaths = ListProperty()
	data = ListProperty()
	pallet_box_height = NumericProperty()

	@logwrap(section="SpriteDialog")
	def pressed(self):
		self.prefix = self.ids.selector.prefix
		self.imgpaths = self.ids.selector.imgpaths
		self.toggle()

	@trigger
	@logwrap(section="SpriteDialog")
	def _choose_graphic_to_import(self, *_):
		try:
			from android.storage import primary_external_storage_path

			path = primary_external_storage_path()
			self._android = True
		except ImportError:
			path = os.getcwd()
			self._android = False
		if hasattr(self, "_graphic_modal"):
			self._file_chooser.path = path
		else:
			self._graphic_modal = ModalView()
			graphic_modal_layout = BoxLayout(orientation="vertical")
			graphic_modal_layout.add_widget(
				Label(text="Pick an image to import", size_hint_y=None)
			)
			self._file_chooser = FileChooserIconView(path=path)
			graphic_modal_layout.add_widget(self._file_chooser)
			cancel = Button(
				text="Cancel", on_release=self._graphic_modal.dismiss
			)
			ok = Button(text="Import", on_release=self._import_graphic)
			buttons_layout = BoxLayout(
				orientation="horizontal", size_hint_y=0.1
			)
			buttons_layout.add_widget(cancel)
			buttons_layout.add_widget(ok)
			graphic_modal_layout.add_widget(buttons_layout)
			self._graphic_modal.add_widget(graphic_modal_layout)
		self._graphic_modal.open()

	@logwrap(section="SpriteDialog")
	def _copy_from_shared(self, src: str):
		from android import autoclass, cast, mActivity
		from android.storage import (
			app_storage_path,
			primary_external_storage_path,
		)

		dst = os.path.join(app_storage_path(), os.path.basename(src))
		MediaStoreMediaColumns = autoclass(
			"android.provider.MediaStore$MediaColumns"
		)
		mtm = autoclass("android.webkit.MimeTypeMap").getSingleton()
		mime_type = mtm.getMimeTypeFromExtension(src[src.rindex(".") + 1 :])
		if not mime_type.lower().startswith("image"):
			raise ValueError("Not an image")
		root_uri = autoclass(
			"android.provider.MediaStore$Files"
		).getContentUri("external")
		context = mActivity.getApplicationContext()
		select_stmt = and_(
			column(MediaStoreMediaColumns.DISPLAY_NAME) == bindparam("a"),
			column(MediaStoreMediaColumns.RELATIVE_PATH) == bindparam("b"),
		)
		select_stmt.stringify_dialect = "sqlite"
		select_s = str(select_stmt)
		Logger.debug(
			f"SpriteDialog: looking for URI using the query: {select_s}"
		)
		args = [
			os.path.basename(src),
			os.path.dirname(src)
			.replace(primary_external_storage_path(), "")
			.strip("/")
			+ "/",
		]
		Logger.debug(f"SpriteDialog: with the aruments: {args}")
		cursor = context.getContentResolver().query(
			root_uri,
			None,
			select_s,
			args,
			None,
		)
		if not cursor:
			raise FileNotFoundError(src)
		while cursor.moveToNext():
			idx = cursor.getColumnIndex(MediaStoreMediaColumns.DISPLAY_NAME)
			file_name = cursor.getString(idx)
			Logger.debug(f"SpriteDialog: file #{idx}. {file_name}")
			if file_name == os.path.basename(src):
				id_ = cursor.getLong(
					cursor.getColumnIndex(MediaStoreMediaColumns._ID)
				)
				uri = autoclass("android.content.ContentUris").withAppendedId(
					root_uri, id_
				)
				break
		else:
			cursor.close()
			raise FileNotFoundError(src)
		if uri.getScheme().lower() == "file":
			shutil.copyfile(src, dst)
			return dst
		context = mActivity.getApplicationContext()
		cursor = context.getContentResolver().query(
			uri, None, None, None, None
		)
		if not cursor:
			raise FileNotFoundError(src)
		if os.path.exists(dst):
			os.remove(dst)
		FileOutputStream = autoclass("java.io.FileOutputStream")
		FileUtils = autoclass("android.os.FileUtils")
		resolver = context.getContentResolver()
		reader = resolver.openInputStream(uri)
		writer = FileOutputStream(dst)
		FileUtils.copy(reader, writer)
		reader.close()
		writer.close()
		cursor.close()
		return dst

	@trigger
	@logwrap(section="SpriteBuilder")
	def _import_graphic(self, *_):
		if not self._file_chooser.selection:
			# should display an error message briefly
			return
		file_path = os.path.join(
			self._file_chooser.path, self._file_chooser.selection[0]
		)
		if self._android:
			if not hasattr(self, "_storage"):
				from android.permissions import Permission, request_permissions

				request_permissions([Permission.READ_MEDIA_IMAGES])
			to_import: str = self._copy_from_shared(file_path)
		else:
			to_import: str = os.path.abspath(file_path)
		# We'll need to load the image before we put it in an atlas.
		# Mainly to get its size, but maybe convert from JPG or BMP or w/e.
		# Kivy seems happiest to work with PNG files.
		# Best to do it first, so that if it fails,
		# no files or dirs are made.
		img = Image.load(to_import)
		if not os.path.exists(self.custom_imgs_dir):
			os.makedirs(self.custom_imgs_dir)
		atlas_fn = os.path.join(self.custom_imgs_dir, "custom.atlas")
		if os.path.exists(atlas_fn):
			with open(atlas_fn, "rt") as inf:
				atlas = json.load(inf)
		else:
			atlas = {}
		dest_fn = f"custom-{len(atlas)}.png"
		dest_path = os.path.join(self.custom_imgs_dir, dest_fn)
		img.save(dest_path)
		atlas[dest_fn] = {
			os.path.basename(to_import): [0, 0, img.width, img.height]
		}
		with open(atlas_fn, "wt") as outf:
			json.dump(atlas, outf)
		if not resource_find(atlas_fn):
			raise FileNotFoundError(
				"The created atlas was not where we saved it", atlas_fn
			)
		self.data = [[self.custom_imgs_header, atlas_fn]] + self.data
		self._graphic_modal.dismiss()


class PawnConfigDialog(SpriteDialog):
	pass


class SpotConfigDialog(SpriteDialog):
	pass


class PawnConfigScreen(Screen):
	toggle = ObjectProperty()
	data = ListProperty()
	custom_imgs_header = "Custom pawns"
	imgpaths = ListProperty()


class SpotConfigScreen(Screen):
	toggle = ObjectProperty()
	data = ListProperty()
	custom_imgs_header = "Custom spots"
	imgpaths = ListProperty()


store_kv(
	__name__,
	"""
<SpriteDialog>:
	orientation: 'vertical'
	SpriteBuilder:
		id: builder
		prefix: root.prefix
		default_imgpaths: root.default_imgpaths
		imgpaths: root.imgpaths
		data: root.data
	SpriteSelector:
		id: selector
		textbox: textbox
		size_hint_y: 0.1
		prefix: root.prefix
		default_imgpaths: root.default_imgpaths
		imgpaths: root.imgpaths
		pallets: builder.pallets
		preview: preview
		TextInput:
			id: textbox
			multiline: False
			write_tab: False
			hint_text: 'Enter name prefix'
		Widget:
			id: preview
			canvas:
				Color:
					rgba: 1, 1, 1, 1
		Button:
			text: 'Import'
			on_release: root._choose_graphic_to_import()
		Button:
			text: 'OK'
			on_release: root.pressed()
<PawnConfigScreen>:
	name: 'pawncfg'
	imgpaths: dialog.imgpaths
	PawnConfigDialog:
		id: dialog
		toggle: root.toggle
		default_imgpaths: ['atlas://rltiles/base/unseen']
		custom_imgs_header: root.custom_imgs_header
		custom_imgs_dir: app.prefix + '/custom_pawn_imgs'
		data: root.data
<SpotConfigScreen>:
	name: 'spotcfg'
	imgpaths: dialog.imgpaths
	SpotConfigDialog:
		id: dialog
		toggle: root.toggle
		default_imgpaths: ['atlas://rltiles/floor/floor-stone']
		custom_imgs_header: root.custom_imgs_header
		custom_imgs_dir: app.prefix + '/custom_spot_imgs'
		data: root.data
""",
)
