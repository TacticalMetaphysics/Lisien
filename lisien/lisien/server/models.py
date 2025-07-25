import os

from django.db import models

from LiSE import Engine


class Game(models.Model):
	created = models.DateTimeField(auto_now_add=True)
	prefix = models.FilePathField(allow_files=False, allow_folders=True)
	title = models.CharField(max_length=1000, blank=True, default="")
	owner = models.ForeignKey(
		"auth.User", related_name="games", on_delete=models.CASCADE
	)

	class Meta:
		ordering = ["created"]
		app_label = "LiSE"

	def save(self, *args, **kwargs):
		if not os.path.exists(self.prefix):
			os.makedirs(self.prefix)
			# just make sure all the files are there
			Engine(self.prefix)
		super().save(*args, **kwargs)
