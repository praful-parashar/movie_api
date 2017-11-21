from django.shortcuts import render
from django.contrib.auth.models import User, Group
from rest_framework import viewsets
from .models import Movie
from .serializers import MovieSerializer
from django.http import HttpResponse
from django.http import JsonResponse
import json
from rest_framework.views import APIView
from django.views.decorators.csrf import csrf_exempt
from django.conf import settings
from . import flickbit_dj
import pandas as pd

#from quickstart.serializers import UserSerializer, GroupSerializer
#import ListAPIView
'''
class UserViewSet(viewsets.ModelViewSet):

    queryset = User.objects.all().order_by('date_joined')
    serializer_class = UserSerializer


class GroupViewSet(viewsets.ModelViewSet):

    queryset = Group.objects.all()
    serializer_class = GroupSerializer
for movie in lom:
    if movie.movie_title == request.GET['movie_title']:
# Create your views here.
'''
'''
class ListOfMovies(viewsets.ModelViewSet, APIView):


    queryset = Movie.objects.all()
    serializer_class = MovieSerializer
    #return Response(serializer.data,status=status.HTTP_200_OK)
'''
class ListOfMovies(viewsets.ModelViewSet):
    queryset = Movie.objects.all()
    serializer_class = MovieSerializer
    #return JsonResponse(serializer_class, status=status.HTTP_200_OK)
    paginate_by = None

movies = Movie.objects.all()
@csrf_exempt
def search(request):

    ret_list = []
    dir_list = []
    actors_1 = []
    actors_2 = []
    actors_3 = []
    countries = []
    durations = []
    genres = []
    gross = []
    lang = []
    ncfr = []
    nufr = []
    nvu = []
    pk = []
    years = []
    movie_imdb_links = []
    imdb_rating = []
    budgets = []
    colors = []

    pickup_dict = {}
    pickup_records = []

    paginate_by = None
    if request.method == 'GET':
        query = request.GET.get('movie_title','')
        #query = query.json()
        for movie in movies:
            ret_list.append(movie.movie_title)
            dir_list.append(movie.director_name)
            actors_1.append(movie.actor_1_name)
            actors_2.append(movie.actor_2_name)
            actors_3.append(movie.actor_3_name)
            budgets.append(movie.budget)
            countries.append(movie.country)
            years.append(movie.year)
            movie_imdb_links.append(movie.movie_imdb_link)
            imdb_rating.append(movie.imdb)
            colors.append(movie.color)
            durations.append(movie.duration)
            genres.append(movie.genres)
            gross.append(movie.gross)
            lang.append(movie.language)
            ncfr.append(movie.num_critic_for_reviews)
            nufr.append(movie.num_user_for_reviews)
            nvu.append(movie.num_voted_users)
            pk.append(movie.plot_keywords)
        ret_len = len(ret_list)
        #print (ret_list[2])
        flag = 0
        for c,i in enumerate(ret_list):
            if i.lower() == query.lower():
                print("here also")
                flag = 1
                pos = c

                record = {'actor_1_name':actors_1[c],
                          'actor_2_name':actors_2[c],
                          'actor_3_name':actors_3[c],
                          'budget':budgets[c],
                          'color':colors[c],
                          'country':countries[c],
                          'director_name':dir_list[c],
                          'duration':durations[c],
                          'genres':genres[c],
                          'gorss':gross[c],
                          'imdb':imdb_rating[c],
                          'language':lang[c],
                          'movie_imdb_link':movie_imdb_links[c],
                          'movie_title':ret_list[c],
                          'num_critic_for_reviews':ncfr[c],
                          'num_user_for_reviews':nufr[c],
                          'num_voted_users':nvu[c],
                          'plot_keywords':pk[c],
                          'year':years[c]  }
                          
                #print(record)
                pickup_records.append(record)
                #print(final)
                #print(pickup_records)
                path = "{0}\\static\\book.json".format(settings.PROJECT_ROOT)
                with open(path, "w") as out:
                    json.dump(pickup_records,out)

                full = pd.read_json(path, orient='records')
                cluster = flickbit_dj.import_and_reduce(full)
                print(cluster[0])
                with open("{0}\\static\\movie_{1}.json".format(settings.PROJECT_ROOT,cluster[0]),'r') as f:
                    data = json.load(f)
                data[0] = record
                pickup_dict["results"] = data
                #print(pickup_dict)
                #pickup_dict.insert(0,"result")
                #pickup_dict.insert(0,result)
                #pickup_dict["results",0]=record
                return JsonResponse(pickup_dict)
        #if flag != 1:
        #    s = "This Movie is not available in database at this moment,we will be adding it shortly. if it exist. Sorry for the inconveineience Thanks for using our app."
        #    s = json.loads(s)
        #    return JsonResponse(s)


'''
def import_db(request):
    #loader = Movie()
    with open(r'C:\\Users\\Praful\\Desktop\\tutorial\\static\\dataset.json','r') as f:
        data = json.load(f)
    #loader.args = data
    #loader.save()
    #print(data)

        #print(data)
    #a = data[0]
    #print("data 0:",a)
    #print(a['actor_1_name'])
    for i in data:
        loader = Movie(**i)
        loader.save()
'''
