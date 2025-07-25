from django.contrib.auth.models import User

from rest_framework import generics, permissions, renderers
from rest_framework.views import APIView
from rest_framework.decorators import api_view
from rest_framework.response import Response
from rest_framework.reverse import reverse
from .permissions import IsOwnerOrReadOnly
from .models import Game
from .serializers import UserSerializer, GameSerializer


@api_view(["GET"])
def api_root(request, format=None):
	return Response(
		{
			"users": reverse("user-list", request=request, format=format),
			"snippets": reverse("game-list", request=request, format=format),
		}
	)


class GameView(generics.GenericAPIView):
	queryset = Game.objects.all()
	renderer_classes = [renderers.StaticHTMLRenderer]

	def get(self, request, *args, **kwargs):
		return Response(self.get_game_html(self.get_object()))

	@staticmethod
	def get_game_html(game) -> str:
		return "<p>I'll put game info here</p>"


class GameList(generics.ListCreateAPIView):
	"""List all games, or create a new game"""

	queryset = Game.objects.all()
	serializer_class = GameSerializer
	permission_classes = [permissions.IsAuthenticatedOrReadOnly]

	def perform_create(self, serializer):
		serializer.save(owner=self.request.user)


class GameDetail(generics.RetrieveUpdateDestroyAPIView):
	queryset = Game.objects.all()
	serializer_class = GameSerializer
	permission_classes = [IsOwnerOrReadOnly]


class CharacterAPIView(APIView):
	def get_renderer_context(self):
		ret = super().get_renderer_context()
		# add the engine to ret
		return ret


class UserList(generics.ListAPIView):
	queryset = User.objects.all()
	serializer_class = UserSerializer


class UserDetail(generics.RetrieveAPIView):
	queryset = User.objects.all()
	serializer_class = UserSerializer
