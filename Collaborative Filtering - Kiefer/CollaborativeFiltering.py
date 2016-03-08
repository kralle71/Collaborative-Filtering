
# coding: utf-8

# In[ ]:

import codecs 
import timeit
import random
from math import sqrt
from math import fabs
import numpy as np
import matplotlib.pyplot as plt
import pandas


# # recommender class

# In[ ]:





class recommender:
    """this class is based on the recommender class of Guide2DataMining"""

    def __init__(self, data, k=1, n=5, metric='pearson'):
        """initialize recommender currently, if data is dictionary the recommender is initialized to it. 
        For all other data types of data, no initialization occurs k is the k value for k nearest neighbor
        metric is which distance formula to use n is the maximum number of recommendations to make"""
            
        self.k = k
        self.n = n
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
      
        # The following variables are used for the different algorithms
        self.frequencies = {}
        self.deviations = {}
        self.adjMatrix = {}
        self.floydwarDist = {}
        self.userAvg = {}
        self.soRec={}
        self.euclMatrix = {}
        self.penalty=False
        self.normalized=False
        
        # for some reason I want to save the name of the metric
        self.metric = metric
        if self.metric == 'pearson':
            self.fn = self.pearson
        
      
        # if data is dictionary set recommender data to it      
        if type(data).__name__ == 'dict':
            self.data = data

    def convertProductID2name(self, id):
        """Given product id number return product name"""
        if id in self.productid2name:
            return self.productid2name[id]
        else:
            return id


    def userRatings(self, id, n):
        """Return n top ratings for user with id"""
        print ("Ratings for " + self.userid2name[id])
        ratings = self.data[id]
        print(len(ratings))
        ratings = list(ratings.items())[:n]
        ratings = [(self.convertProductID2name(k), v) for (k, v) in ratings]
        # finally sort and return
        ratings.sort(key=lambda artistTuple: artistTuple[1], reverse = True)      
        for rating in ratings:
            print("%s\t%i" % (rating[0], rating[1]))


    def showUserTopItems(self, user, n):
        """ show top n items for user"""
        items = list(self.data[user].items())
        items.sort(key=lambda itemTuple: itemTuple[1], reverse=True)
        for i in range(n):
            print("%s\t%i" % (self.convertProductID2name(items[i][0]), items[i][1]))
            
    def loadMovieLens(self, path=''):
        """import MovieLens Dataset"""
        self.data = {}
      
        # first load movie ratings
      
        i = 0
      
        # First load book ratings into self.data
      
        #f = codecs.open(path + "u.data", 'r', 'utf8')
        #f = codecs.open(path + "u.data", 'r', 'ascii')
        f = codecs.open(path + "ua.base", 'r', 'ascii')
        #  f = open(path + "u.data")
        for line in f:
            i += 1
             #separate line into fields
            fields = line.split('\t')
            user = fields[0]
            movie = fields[1]
            rating = int(fields[2].strip().strip('"'))
            if user in self.data:
                currentRatings = self.data[user]
            else:
                currentRatings = {}
            currentRatings[movie] = rating
            self.data[user] = currentRatings
        f.close()
      
        # Now load movie into self.productid2name
        # the file u.item contains movie id, title, release date among
        # other fields
          
        #f = codecs.open(path + "u.item", 'r', 'utf8')
        f = codecs.open(path + "u.item", 'r', 'iso8859-1', 'ignore')
        #f = open(path + "u.item")
        for line in f:
            i += 1
             #separate line into fields
            fields = line.split('|')
            mid = fields[0].strip()
            title = fields[1].strip()
            self.productid2name[mid] = title
        f.close()
      
        #  Now load user info into both self.userid2name
        #  and self.username2id
      
        #f = codecs.open(path + "u.user", 'r', 'utf8')
        f = open(path + "u.user")
        for line in f:
            i += 1
            fields = line.split('|')
            userid = fields[0].strip('"')
            self.userid2name[userid] = line
            self.username2id[line] = userid
        f.close()
        print(i)

              
        
    def computeDeviations(self):
        """Precalculations for SlopeOne"""
        # for each person in the data:
        #    get their ratings
        for ratings in self.data.values():
            # for each item & rating in that set of ratings:
            for (item, rating) in ratings.items():
                self.frequencies.setdefault(item, {})
                self.deviations.setdefault(item, {})                    
                # for each item2 & rating2 in that set of ratings:
                for (item2, rating2) in ratings.items():
                    if item != item2:
                    # add the difference between the ratings to our
                    # computation
                        self.frequencies[item].setdefault(item2, 0)
                        self.deviations[item].setdefault(item2, 0.0)
                        self.frequencies[item][item2] += 1
                        self.deviations[item][item2] += rating - rating2
        
        for (item, ratings) in self.deviations.items():
            for item2 in ratings:
                ratings[item2] /= self.frequencies[item][item2]
                
    def slopeOneRecommendations(self, user, item='0'):
        """SlopeOne Recommendations"""
        recommendations = {}
        frequencies = {}
          # for every item and rating in the user's recommendations
        for (userItem, userRating) in self.data[user].items():
             # for every item in our dataset that the user didn't rate
            for (diffItem, diffRatings) in self.deviations.items():
                if diffItem not in self.data[user] and                     userItem in self.deviations[diffItem]:
                    freq = self.frequencies[diffItem][userItem]
                    recommendations.setdefault(diffItem, 0.0)
                    frequencies.setdefault(diffItem, 0)
                    # add to the running sum representing the numerator
                    # of the formula
                    recommendations[diffItem] += (diffRatings[userItem] +
                                             userRating) * freq
                    # keep a running sum of the frequency of diffitem
                    frequencies[diffItem] += freq
        
        recommendations =  [(k, v / frequencies[k])
                          for (k, v) in recommendations.items()]
            # finally sort and return
        recommendations.sort(key=lambda artistTuple: artistTuple[1],
                           reverse = True)
        # Return some recommendations for the user, or a particular rating of an item for the user
        if item =='0':
            return recommendations[:50]
        else:
            return recommendations[item]
    
    def sloOne(self,user,item):
        """sloOne is used to speed up the test"""
        if user not in self.soRec.keys():
            # save the recommendations for any user in soRec
            self.soRec[user] = self.slopeOneRecommendations(user)
        for i in range(len(self.soRec[user])):
            # look up the specific item and return the value
            if self.soRec[user][i][0]==item:
                result = self.soRec[user][i][1] 
                return result
        return self.userAvg[user]
    
    def computeUserAverages(self):
        """Computes user averages"""
        results = {}
        for (key, ratings) in self.data.items():
            results[key] = float(sum(ratings.values())) / len(ratings.values())
        self.userAvg = results
    
    def computeItemAverages(self):
        """Computes item averages"""
        sumRating = {}
        freqRating = {}
        results = {} 
        for (key, ratings) in self.data.items():
            for (k) in ratings:
                if k not in sumRating:
                    sumRating[k] = ratings[k]
                    freqRating[k] = 1
                else:
                    sumRating[k] += ratings[k]
                    freqRating[k] += 1
        for key in sumRating:
            results[key] = float(sumRating[key])/freqRating[key]
        return results

    def computeSimilarity(self, item1, item2):
        """Computes similarity for adjcos"""
        num = 0
        dem1 = 0
        dem2 = 0
        #for every user check if they rated both items and count them for the similarity
        for (user, ratings) in self.data.items():
            if item1 in ratings and item2 in ratings:
                avg = self.userAvg[user]
                num += (ratings[item1]-avg) * (ratings[item2]-avg)
                dem1 += (ratings[item1]-avg)**2
                dem2 += (ratings[item2]-avg)**2
        if dem1*dem2!=0:
            return num / (sqrt(dem1) * sqrt(dem2))
        return -3
         
    
    def normalizeRating(self,rating,minRating=1,maxRating=5): 
        """normalizes to range [-1,1]"""
        return float((2*(rating-minRating)-(maxRating-minRating)))/(maxRating-minRating)
    
    def denormalizeRating(self,rating,minRating=1,maxRating=5):
        """transforms range [-1,1] to range [1,5]"""
        return (float((rating+1)*(maxRating-minRating))/2) +minRating
    

    def adjcos(self, user, diffItem):
        """Adjusted Cosine Similarity Recommendations"""
        # calculates the rating of diffItem for User given
        num = 0
        dem = 0
        if diffItem not in self.data[user].keys():
            for (uItem, uRating) in self.data[user].items():
                # for all items of the user calculate the similarity to the diffItem
                similarity = self.computeSimilarity(diffItem,uItem)
                if similarity != -3: # avoid dividing with 0, if noone rated both Items
                    nRating = self.normalizeRating(uRating)
                    #calculate the rating
                    num += similarity*nRating
                    dem += fabs(similarity)        
            if dem == 0:
                return self.userAvg[user]
            else:
                return float(self.denormalizeRating(float(num/dem)))
        else:
            print("already rated with %10.3f") % (self.data[user][diffItem])
            

     
        
    def pearson(self, rating1, rating2):
        """computes pearson score between rating1 and rating2"""
        sum_xy = 0
        sum_x = 0
        sum_y = 0
        sum_x2 = 0
        sum_y2 = 0
        n = 0
        for key in rating1:
            if key in rating2:
                n += 1
                x = rating1[key]
                y = rating2[key]
                sum_xy += x * y
                sum_x += x
                sum_y += y
                sum_x2 += pow(x, 2)
                sum_y2 += pow(y, 2)
        if n == 0:
            return 0
        # now compute denominator
        denominator = sqrt(sum_x2 - pow(sum_x, 2) / n) *                     sqrt(sum_y2 - pow(sum_y, 2) / n)
        if denominator == 0:
            return 0
        else:
            return (sum_xy - (sum_x * sum_y) / n) / denominator


    def computeNearestNeighbor(self, username, rateItem = '0'):
        """creates a sorted list of users based on their distance to username"""
        distances = []
        for instance in self.data:
            if instance != username:
                if rateItem == '0':
                    #compute nearest neighbors
                    distance = self.fn(self.data[username],
                               self.data[instance])
                    distances.append((instance, distance))
                else:
                    #compute nearest neighbors, who rated the given item
                    if rateItem in self.data[instance]:
                        distance = self.fn(self.data[username],
                               self.data[instance])
                        distances.append((instance, distance))
        # sort based on distance -- closest first
        if self.fn == self.pearson:
            distances.sort(key=lambda artistTuple: artistTuple[1],
                         reverse=True)
        elif self.fn == self.euclDist:
            distances.sort(key=lambda artistTuple: artistTuple[1])
        return distances
    
    def computeAdjacentMatrix(self):
        """Precalculations for FW"""
        """computes the weights (euclid distance) between edges of the graph"""
        distlist = []
        (r.penalty,r.normalized)=(True,False)
        self.fn = self.euclDist
        for (user) in self.data:
            self.adjMatrix[user]={}
            distlist = self.computeNearestNeighbor(user)
            for (u,i) in distlist:
                self.adjMatrix[user][u]=i
    
    def floydwarshall(self):
        """https://jlmedina123.wordpress.com/2014/05/17/floyd-warshall-algorithm-in-python/""" 
        # Initialize dist and pred:
        # copy graph into dist, but add infinite where there is
        # no edge, and 0 in the diagonal
        self.computeAdjacentMatrix()
        graph = self.adjMatrix
        dist = {}
        pred = {}
        for u in graph:
            dist[u] = {}
            pred[u] = {}
            for v in graph:
                dist[u][v] = 1000
                pred[u][v] = -1
            dist[u][u] = 0
            for neighbor in graph[u]:
                dist[u][neighbor] = graph[u][neighbor]
                pred[u][neighbor] = u
 
        for t in graph:
            # given dist u to v, check if path u - t - v is shorter
            for u in graph:
                for v in graph:
                    newdist = dist[u][t] + dist[t][v]
                    if newdist < dist[u][v]:
                        dist[u][v] = newdist
                        pred[u][v] = pred[t][v] # route new path through t
 
        self.floydwarDist = dist

    def flowar(self, user, k, item):
        """FloydWarshall Recommendations"""
        self.k = k
        # find nearest neighbors
        dist = list(r.floydwarDist[user].items())
        dist.sort(key=lambda ks: ks[1],reverse = True)
        n = 0
        nearest=[]
        for (distItem) in dist:
            if item in r.data[distItem[0]]:
                nearest.append((distItem))
                n += 1
            if n == k:
                break            
        self.k = min(len(nearest), k)
        if self.k ==0:
            return self.userAvg[user]
        recommendation = 0
        totalDistance = 0.0
        for i in range(self.k):
            if item in self.data[nearest[i][0]]:
                totalDistance += 1
        # now iterate through the k nearest neighbors
        # accumulating their ratings
        if totalDistance == 0:
            return self.userAvg[user]
        for i in range(self.k):
            if item in self.data[nearest[i][0]]:
            # compute slice of pie 
                weight = 1 / totalDistance
            # get the name of the person
                name = nearest[i][0]
            # get the ratings for this person
                neighborRatings = self.data[name]
            # get the name of the person
            # now find bands neighbor rated that user didn't
                recommendation += neighborRatings[item] * weight
        return recommendation
    
    def usermean(self, user):
        """Usermean Recommendations"""
        self.computeUserAverages()
        return self.userAvg[user]
    
    def itemmean(self, item):
        """Itemmean Recommendations"""
        avg = self.computeItemAverages()
        if item not in avg:
            return 0
        else:
            return avg[item]
        
    def euclDist(self, rating1, rating2):
        """computes euclid distance between rating1 and rating2"""
        sum = 0
        freq = 0
        nooverlap = True
        for key in rating1:
            if key in rating2:
                sum += (rating1[key]-rating2[key])**2
                freq +=1
                nooverlap = False
            elif self.penalty:
                #add penalty for items in rating1 but not in rating2
                sum += ((rating1[key]-3))**2
                freq +=1
        if freq == 0 or nooverlap:
            return 1000
        else:
            result = sqrt(sum)
            if self.normalized:
                result /= freq
            return result
        

    def recommend(self, metric, user, k, rateItem = '0', n=30):
        """Pearson or Euclid Recommendations (based on parameter metric)"""
        """Give list of recommendations. rateItem.default is set to '0', which means we will get the n best 
        recommendations. If the parameter is given, it will rate just this item."""
        
        self.k = k
        self.n = n
        recommendations = {}
        recommendation = 0
        if metric == 'pearson':
            self.fn = self.pearson
        elif metric == 'euclid':
            self.fn = self.euclDist
            
        # first get list of users  ordered by nearness
        if rateItem == '0':
            nearest = self.computeNearestNeighbor(user)
        else:
            nearest = self.computeNearestNeighbor(user, rateItem)
        # if fewer neighbors than k rated the item
        self.k = min(len(nearest), k)
        
        #
        # now get the ratings for the user
        #
        userRatings = self.data[user]
        #
        # determine the total distance
        totalDistance = 0.0
        for i in range(self.k):
            if self.fn == self.pearson:
                totalDistance += nearest[i][1] #weighted mean
            elif self.fn == self.euclDist:
                totalDistance += 1 #classic mean
        # now iterate through the k nearest neighbors
        # accumulating their ratings
        if totalDistance == 0:
            return self.userAvg[user];
        for i in range(self.k):
            # compute slice of pie
            if self.fn == self.pearson:
                weight = nearest[i][1] / totalDistance
            elif self.fn == self.euclDist:
                weight = 1 / totalDistance
            # get the name of the person
            name = nearest[i][0]
            # get the ratings for this person
            neighborRatings = self.data[name]
            # get the name of the person
            # now find bands neighbor rated that user didn't
            if rateItem == '0':
                for artist in neighborRatings:
                    if not artist in userRatings:
                        if artist not in recommendations:
                            recommendations[artist] = neighborRatings[artist] * weight
                        else:
                            recommendations[artist] = recommendations[artist] +                                             neighborRatings[artist] * weight
            else:
                recommendation += neighborRatings[rateItem] * weight
        
        if rateItem == '0':
        # now make list from dictionary and only get the first n items
            recommendations = list(recommendations.items())[:self.n]
        #recommendations = [(self.convertProductID2name(k), v)
            recommendations = [(k, v)
                             for (k, v) in recommendations]
        # finally sort and return
            recommendations.sort(key=lambda artistTuple: artistTuple[1],
                               reverse = True)
            return recommendations
        else:
            return recommendation



# # test class

# In[ ]:

class tester:
    
    def __init__(self,r,data):
        """initialize tester"""
        
        self.username2id = {}
        self.userid2name = {}
        self.productid2name = {}
        
        if type(data).__name__ == 'dict':
            self.data = data
      
    
    def loadMovieLens(self, path=''):
        self.data = {}
      
        # first load movie ratings
      
        i = 0
      
        # First load book ratings into self.data
      
        #f = codecs.open(path + "u.data", 'r', 'utf8')
        #f = codecs.open(path + "u.data", 'r', 'ascii')
        f = codecs.open(path + "ua.test", 'r', 'ascii')
        #  f = open(path + "u.data")
        for line in f:
            i += 1
             #separate line into fields
            fields = line.split('\t')
            user = fields[0]
            movie = fields[1]
            rating = int(fields[2].strip().strip('"'))
            if user in self.data:
                currentRatings = self.data[user]
            else:
                currentRatings = {}
            currentRatings[movie] = rating
            self.data[user] = currentRatings
        f.close()
      
        # Now load movie into self.productid2name
        # the file u.item contains movie id, title, release date among
        # other fields
          
        #f = codecs.open(path + "u.item", 'r', 'utf8')
        f = codecs.open(path + "u.item", 'r', 'iso8859-1', 'ignore')
        #f = open(path + "u.item")
        for line in f:
            i += 1
             #separate line into fields
            fields = line.split('|')
            mid = fields[0].strip()
            title = fields[1].strip()
            self.productid2name[mid] = title
        f.close()
      
        #  Now load user info into both self.userid2name
        #  and self.username2id
      
        #f = codecs.open(path + "u.user", 'r', 'utf8')
        f = open(path + "u.user")
        for line in f:
            i += 1
            fields = line.split('|')
            userid = fields[0].strip('"')
            self.userid2name[userid] = line
            self.username2id[line] = userid
        f.close()
        print(i)
        
    def testinput(self,fct,k,a):
        """input test for parameters given"""
        
        fctlist = ['adjcos','hybrid','floydwarshall','pearson','slopeone','euclid','usermean','itemmean','random']
        
        try: #k is integer and positiv
            k = int(k)
            if k<0:
                print("\nPlease insert a positive k amount of neighbors")
                return False
        except ValueError:
            print("\nPlease only use integers")
            return False
        
        try: #a is float and in range [0,1]
            a = float(a)
            if a<0 or a>1:
                print("\nPlease use the paramater a in range [0,1]")
                return False
        except ValueError:
            print("\nPlease only use integers")
            return False
        
        if fct not in fctlist: #fct is in fctlist
            print("\nPlease use a function like ['%s']" % "', '".join(map(str, fctlist)))
            return False
        
        return True

        
    def testit(self, fct, k=10, a=0.5):
        """toolbox for the test of different algorithms"""
        if not self.testinput(fct,k,a):
            return
        
        sumall = 0
        lenall = 0
        avgall = 0
        
        #precalculations
        if fct == 'slopeone' or fct == 'hybrid':
            r.computeDeviations()
        if fct == 'floydwarshall':
            r.floydwarshall()
        
            
        for user in self.data.keys():
            sum = 0
            len = 0
            avg = 0
            for (item) in self.data[user]:
            #calculate the error between predicted rating and rating from testdata    
                if fct == 'adjcos':
                    sum += fabs(float(self.data[user][item])-r.adjcos(user, item))
                elif fct == 'hybrid':
                    sum += fabs(float(self.data[user][item])-                             ((1-a)*r.recommend('euclid',user, k, item)+a*r.sloOne(user,item)))
                elif fct == 'floydwarshall':
                    sum += fabs(float(self.data[user][item])-r.flowar(user,k,item))
                elif fct == 'slopeone':
                    sum += fabs(float(self.data[user][item])-r.sloOne(user, item))
                elif fct == 'pearson':
                    sum += fabs(float(self.data[user][item])-r.recommend('pearson', user, k, item))
                elif fct == 'euclid':
                    sum += fabs(float(self.data[user][item])-r.recommend('euclid', user, k, item))
                elif fct =='usermean':
                    sum += fabs(float(self.data[user][item])-r.usermean(user))
                elif fct =='itemmean':
                    sum += fabs(float(self.data[user][item])-r.itemmean(item)) 
                elif fct =='random':
                    sum += fabs(float(self.data[user][item])-random.uniform(1,5))
                len += 1
            avg = float(sum)/len #avg error for user
            sumall += avg
            lenall += 1
        avgall = float(sumall)/lenall #avg error for all user
        return (avgall)
        


# # initialising the classes and import data

# In[ ]:

l = 1
path = 'PathtoData\Data'

r = recommender(0)
r.loadMovieLens(path+'\ml-L='+str(l)+'/')
r.computeUserAverages()

t = tester(r,0)
t.loadMovieLens(path+'\ml-L='+str(l)+'/')


# possible folders are ml-L=1  / 5, 10, 19. .
# determines how many ratings per user will be in testdata

# # ToyData low rank matrix
#             

# In[ ]:

"""creating low rank matrix"""
#setting parameters
l = 1
o = 30
m = 943
n = 1682


matrixdensity = 0.063
#matrixdensity = 0.30
normalize=True


#create random low rank matrix with rank o
A1 = np.random.uniform(1,5,(m,o))
A2 = np.random.uniform(1,5,(o,n))

B = np.dot(A1,A2)
#np.linalg.matrix_rank(B)

if normalize:
    # normalize to range [1,5]
    Dmax = np.max(B)
    Dmin = np.min(B)
    for (i,j), value in np.ndenumerate(B):
        B[i,j] = round(r.denormalizeRating(r.normalizeRating(B[i,j],Dmin,Dmax),0.51,5.49),0)

# killing matrixdensity of the entries
C = np.random.rand(m,n)

for (i,j), value in np.ndenumerate(C):
    if C[i][j]>matrixdensity:
        C[i,j]=0
    else: C[i,j]=1 

D = B*C


np.linalg.matrix_rank(D)

#convert to dictionaries Test and Train(l items of each user)
traindata = {}
testdata = {}
testcount={}

for (i,j), value in np.ndenumerate(D):
    istr = str(i+1)
    traindata.setdefault(istr,{})
    testdata.setdefault(istr,{})
    testcount.setdefault(istr,0)

r = recommender(0)
for (i,j), value in np.ndenumerate(D):
    istr = str(i+1)
    jstr = str(j+1)    
    if D[i,j]>0:
        if testcount[istr]<l:
            #testdata[istr][jstr] = round(r.denormalizeRating(r.normalizeRating(D[i,j],Dmin,Dmax),0.51,5.49),0)
            testdata[istr][jstr] = D[i,j]
            testcount[istr]+=1
        else:
            traindata[istr][jstr]=D[i,j]
        
r = recommender(traindata)
r.computeUserAverages()
t = tester(r,testdata)


# ### Create histograms to compare the randomized low rank matrix to the MovieLens data

# In[ ]:

#convert the MovieLens-Dataset to a dataframe to create the histogram
df = pandas.DataFrame.from_dict(r.data, orient='index').fillna(0)
histML, binsML = np.histogram(df, bins=(1,2,3,4,5,6))
histLR, binsLR = np.histogram(D, bins=(1,2,3,4,5,6))
#plot both histograms together
centerML = (binsML[:-1] + binsML[1:]) / 2
centerLR = (binsLR[:-1] + binsLR[1:]) / 2
plt.bar(centerLR-0.1, histLR, color='r', align='center', width=0.7, alpha=0.7, label='low rank matrix')
plt.bar(centerML+0.1, histML, color='b',align='center', width=0.7, alpha=0.7, label='MovieLens Data')
fig1 = plt.figure(1)
rect = fig1.patch
rect.set_facecolor('white')
plt.title('Histogram')
plt.xlabel('Rating')
plt.ylabel('Frequency')
plt.legend()
plt.show()


# ## precalculations for floydwarshall for test purposes

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'r.floydwarshall()\nam = r.adjMatrix\nfw = r.floydwarDist')


# In[ ]:

r.adjMatrix = am
r.floydwarDist = fw


# # Parameter test for Euclid, Pearson, FloydWarshall and Hybrid

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'(r.penalty,r.normalized)=(True,False)\nk=15\nalist=[0,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9,1]\nfor a in alist:\n    print ("%3.1f: %10.5f" %  (a,t.testit(\'hybrid\',k,a)))')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'klist = [1,25,50,100,200,500]\nfor k in klist:\n    print ("%i: %10.5f" %  (k,t.testit(\'floydwarshall\',k)))')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'klist = [1,5,10,15,20,25,50,100,200]\nfor k in klist:\n    print ("%i: %10.5f" %  (k,t.testit(\'pearson\',k)))')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'klist = [1,5,10,15,20,25,50,100,200]\n#(r.penalty,r.normalized)=(False,False)\n(r.penalty,r.normalized)=(True,False)\n#(r.penalty,r.normalized)=(False,True)\nfor k in klist:\n    print ("%i: %10.5f" %  (k,t.testit(\'euclid\',k)))')


# # Calculation of the error for a given algorithm

# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'print "Hybrid"\nprint "%10.5f" % t.testit(\'hybrid\',15,0.3)')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'print "FloydWarshall"\nprint "%10.5f" % t.testit(\'floydwarshall\',10)')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'#(r.penalty,r.normalized)=(False,False)\n(r.penalty,r.normalized)=(True,False)\n#(r.penalty,r.normalized)=(False,True)\nprint "Euclid"\nprint "%10.5f" % t.testit(\'euclid\',15)')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'print "Pearson"\nprint "%10.5f" % t.testit(\'pearson\',10)')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'print "Adjcos"\nprint "%10.5f" % t.testit(\'adjcos\')')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'print "SlopeOne"\nprint "%10.5f" % t.testit(\'slopeone\')')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'print "UserMean"  \nprint "%10.5f" % t.testit(\'usermean\')')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'print "ItemMean"\nprint "%10.5f" % t.testit(\'itemmean\')')


# In[ ]:

get_ipython().run_cell_magic(u'time', u'', u'print "Random"\nprint "%10.5f" % t.testit(\'random\')')

