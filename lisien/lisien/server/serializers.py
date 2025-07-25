import os

from django.contrib.auth.models import User
from rest_framework import serializers

from .models import Game


class GameSerializer(serializers.Serializer):
	id = serializers.IntegerField(read_only=True)
	created = serializers.DateTimeField()
	prefix = serializers.CharField(max_length=1000, allow_blank=False)
	"""Path to the game directory
	
	Should probably be a FilePathField but that doesn't instantiate right
	
	"""
	title = serializers.CharField(
		max_length=1000, allow_blank=True, default=""
	)
	owner = serializers.ReadOnlyField(source="owner.username")

	def create(self, validated_data):
		return Game.objects.create(**validated_data)

	def update(self, instance, validated_data):
		prefix = validated_data.get("prefix", instance.prefix)
		if not os.path.exists(prefix):
			os.makedirs(prefix)
		instance.prefix = prefix
		instance.title = validated_data.get("title", instance.title)

	class Meta:
		model = Game
		fields = ["id", "created", "prefix", "title", "owner"]


class UserSerializer(serializers.ModelSerializer):
	games = serializers.PrimaryKeyRelatedField(
		many=True, queryset=Game.objects.all()
	)

	class Meta:
		model = User
		fields = ["id", "username", "games"]
