"""Packages in use"""
import codecs 
import timeit
import random
from math import sqrt
from math import fabs
import numpy as np
import matplotlib.pyplot as plt
import pandas

class recommender: #(page 27)
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
        """import MovieLens Dataset"""
                
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
   
    def computeAdjacentMatrix(self):
        """Precalculations for FW"""
        """computes the weights (euclid distance) between edges of the graph"""
       
    def floydwarshall(self):
    	"""computes shortest paths"""
        """https://jlmedina123.wordpress.com/2014/05/17/floyd-warshall-algorithm-in-python/""" 

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
 
class tester: #(page 41)
    
    def __init__(self,r,data):
        """initialize tester"""
        
    def loadMovieLens(self, path=''):
                
    def testinput(self,fct,k,a):
        """input test for parameters given"""
                
    def testit(self, fct, k=10, a=0.5):
        """toolbox for the test of different algorithms"""

"""Load MovieLens-Dataset"""
l = 1
path = 'PathtoData\Data'

r = recommender(0)
r.loadMovieLens(path+'\ml-L='+str(l)+'/')
r.computeUserAverages()

t = tester(r,0)
t.loadMovieLens(path+'\ml-L='+str(l)+'/')
