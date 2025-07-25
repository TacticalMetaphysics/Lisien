from django.urls import path
from rest_framework.urlpatterns import format_suffix_patterns

from . import views

urlpatterns = [
	path("games/", views.GameList.as_view()),
	path("games/<int:pk>/", views.GameDetail.as_view()),
	path("games/<int:pk>/view", views.GameView.as_view()),
	# path(
	# 	"api/character",
	# 	views.CharacterAPIView.as_view(),
	# 	name="character_api_endpoint",
	# ),
	path("users/", views.UserList.as_view()),
	path("users/<int:pk>/", views.UserDetail.as_view()),
	path("", views.api_root),
]
urlpatterns = format_suffix_patterns(urlpatterns)
