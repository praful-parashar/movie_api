from django.db import models

# Create your models here.

class Movie(models.Model):
    actor_1_name = models.CharField(max_length=50,null=True)
    actor_2_name = models.CharField(max_length=30,null=True)
    actor_3_name = models.CharField(max_length=30,null=True)
    budget = models.IntegerField(null=True)
    color = models.CharField(max_length=10,null=True)
    country = models.CharField(max_length=20,null=True)
    director_name = models.CharField(max_length=25)
    duration = models.IntegerField(null=True)
    genres = models.CharField(max_length=50,null=True)
    gross = models.IntegerField(null=True)
    imdb = models.FloatField(null=True)
    language = models.CharField(max_length=20,null=True)
    movie_imdb_link = models.CharField(max_length=100,null=True)
    movie_title = models.CharField(max_length=50,null=True)
    num_critic_for_reviews = models.IntegerField(null=True)
    num_user_for_reviews = models.IntegerField(null=True)
    num_voted_users = models.IntegerField(null=True)
    plot_keywords = models.CharField(max_length=100,null=True)
    year = models.IntegerField(null=True)



    def __str__(self):
        return self.movie_title
