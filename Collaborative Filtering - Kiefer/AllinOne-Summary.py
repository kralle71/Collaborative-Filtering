
"""Packages in use"""
import codecs 
import timeit
import random
from math import sqrt
from math import fabs
import numpy as np
import matplotlib.pyplot as plt
import pandas


class recommender:
    """this class is based on the recommender class of Guide2DataMining"""

    def __init__(self, data, k=1, n=5, metric='pearson'):
        """initialize recommender"""

    def convertProductID2name(self, id):
        """Given product id number return product name"""

    def userRatings(self, id, n):
        """Return n top ratings for user with id"""

    def showUserTopItems(self, user, n):
        """ show top n items for user"""
            
    def loadMovieLens(self, path=''):
        """import MovieLens trainings-set"""
        
    def computeDeviations(self):
        """Precalculations for SlopeOne"""
        
    def slopeOneRecommendations(self, user, item='0'):
        """SlopeOne Recommendations"""
    
    def sloOne(self,user,item):
        """sloOne is used to speed up the test"""
            
    def computeUserAverages(self):
        """Computes user averages"""
    
    def computeItemAverages(self):
    	"""Computes item averages"""

    def computeSimilarity(self, item1, item2):
        """Computes similarity for adjcos"""      
    
    def normalizeRating(self,rating,minRating=1,maxRating=5): 
        """normalizes to range [-1,1]"""
           
    def denormalizeRating(self,rating,minRating=1,maxRating=5):
    	"""transforms range [-1,1] to range [1,5]"""

    def adjcos(self, user, diffItem):
    	"""Adjusted Cosine Similarity Recommendations"""
        
    def pearson(self, rating1, rating2):
        """computes pearson score between rating1 and rating2"""

    def computeNearestNeighbor(self, username, rateItem = '0'):
        """creates a sorted list of users based on their distance to username"""
    
    def flwa(self):
        """http://masc.cs.gmu.edu/wiki/FloydWarshall"""
        """computes maximin paths with FloydWarshall"""
    
    def computeAdjacentMatrix(self, user='0', item='0'):
        """Precalculations for FW"""
        """computes the weights (pearson score) between edges of the graph"""

    def flowar(self, user, k, item):
        """FloydWarshall Recommendations"""
        
    def usermean(self, user):
        """Usermean Recommendations"""
    
    def itemmean(self, item):
    	"""Itemmean Recommendations"""
        
    def euclDist(self, rating1, rating2):
        """computes euclid distance between rating1 and rating2"""

    def recommend(self, metric, user, k, rateItem = '0', n=30):
        """Pearson or Euclid Recommendations (based on parameter metric)"""
        

class tester:
    
    def __init__(self,data):
        """initialize tester"""
    
    def loadMovieLens(self, path=''):
    	"""Import MovieLens test-set"""
        
    def testinput(self,fct,k,a):
        """input test for parameters given"""
        
    def testit(self, fct, k=10, a=0.5):
        """toolbox for the test of different algorithms"""

        
"""get started"""
l = 1
path = '\Data'

r = recommender(0)
r.loadMovieLens(path+'\ml-L='+str(l)+'/')
r.computeUserAverages()

t = tester(0)
t.loadMovieLens(path+'\ml-L='+str(l)+'/')


# possible folders are ml-L=1  / 5, 10, 19. .
# determines how many ratings will be in testdata


