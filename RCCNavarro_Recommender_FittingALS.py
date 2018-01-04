
# coding: utf-8

# # Advanced Machine Learning (MScA, 32017)
# 
# # Project Recommending Music with Audioscrobbler Data
# 
# ### Yuri Balasanov, Mihail Tselishchev, &copy; iLykei 2017
# 
# ## Fitting ALS model to Audioscrobbler (LastFM) data

# In[2]:


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, Row
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as func
import random
import time
from datetime import datetime


# In[3]:


spark = SparkSession.builder.getOrCreate()
spark


# ## Data
# 
# Create paths to the data files. Add path to file with predictions for the test that will be calculated at the end of this notebook.  

# In[4]:


# paths to files
artistdata_path = './data/artist_data.csv'
userartist_path = './data/clean_15_5.csv'
test_path = './data/LastFM_Test_Sample.csv'


# In[5]:


# defining schemas
artistdata_struct = StructType([StructField('artistId', IntegerType()),                                 StructField('name', StringType())])
userartist_struct = StructType([StructField('userId', IntegerType()),                                 StructField('artistId', IntegerType()),                                 StructField('count', IntegerType())])


# In[6]:


# read artist names data
artistdata_df = spark.read.csv(artistdata_path, sep = '\t',schema = artistdata_struct)
artistdata_df.cache()
artistdata_df.show(10)


# In[7]:


# read user-artist data
userartist_df = spark.read.csv(userartist_path, sep = ',', header=True,schema = userartist_struct)
userartist_df.cache()
userartist_df.show(10)


# In[8]:


# split data:
(training, test) = userartist_df.randomSplit([0.9, 0.1], seed=0)
training.cache()
# remove 'count' column from test:
test = test.drop('count')
test.cache()
test.show(10)


# ## Fitting model
# 
# Fit the ALS model. <br>
# Hyperparameters to specify: <br>
# 
# -  `rank` between 5 and 40; default 10; the number of latent factors in the model
# -  `regParam` between 0.01 and 8; default 0.1; regularization parameter $\lambda$
# -  `alpha` between 1 and 40; default 1; parameter $\alpha$ appears in the expression for confidence $$c_{u,i}=1+\alpha r_{u,i}$$ or $$c_{u,i}=1+\alpha \ln(1+\frac{r_{u,i}}{\epsilon}).$$ If $\alpha=0$  confidence is always 1 regardless of rating$r_{u,i}$. As $\alpha=0$ grows we pay more and more attention to how many times user $u$ consumed item $i$. Thus $\alpha$ controls the relative weight of observed versus unobserved ratings. 
# 
# Search for hyperparameters on the grid of 4-5 values in each range.

# In[9]:


# broadcast all artist ids
allItemIDs = userartist_df.select('artistId').distinct().rdd.map(lambda x: x[0]).collect()
bAllItemIDs = spark.sparkContext.broadcast(allItemIDs)


# In[10]:


# broadcast 10000 most popular artist ids
artists = userartist_df.groupBy('artistId') .agg(func.count(func.lit(1)).alias('num_of_users'))

artists.cache()
top_artists = artists. orderBy('num_of_users', ascending=False).limit(10000). rdd.map(lambda x: x['artistId']).collect()

bTopItemIDs = spark.sparkContext.broadcast(top_artists)


# Calculation of AUC is described in the book Advanced Analytics with Spark.
# 
# In the calculation below parameter `positiveData` has the meaning of "positive" or "good" artist for the user. Parameter `predictFunction` is a function that takes user-item pairs and predicts estimated strength of interactions between them.

# In[11]:


# define meanAUC logic according to 'Advanced Analytics with Spark'

def areaUnderCurve(positiveData, bAllItemIDs, predictFunction):
    positivePredictions = predictFunction(positiveData.select("userId", "artistId"))        .withColumnRenamed("prediction", "positivePrediction")
        
    negativeData = positiveData.select("userId", "artistId").rdd                    .groupByKey()                    .mapPartitions(lambda userIDAndPosItemIDs: 
                                   createNegativeItemSet(userIDAndPosItemIDs, 
                                                         bAllItemIDs))\
                    .flatMap(lambda x: x).map(lambda x: Row(userId=x[0], artistId=x[1])) \
                .toDF()
    
    negativePredictions = predictFunction(negativeData)        .withColumnRenamed("prediction", "negativePrediction")

    joinedPredictions = positivePredictions.join(negativePredictions, "userId")        .select("userId", "positivePrediction", "negativePrediction").cache()
        
    allCounts = joinedPredictions        .groupBy("userId").agg(func.count(func.lit("1")).alias("total"))        .select("userId", "total")
    correctCounts = joinedPredictions        .where(joinedPredictions.positivePrediction > joinedPredictions.negativePrediction)        .groupBy("userId").agg(func.count("userId").alias("correct"))        .select("userId", "correct")

    joinedCounts = allCounts.join(correctCounts, "userId")
    meanAUC = joinedCounts.select("userId", (joinedCounts.correct / joinedCounts.total).                                   alias("auc"))        .agg(func.mean("auc")).first()

    joinedPredictions.unpersist()

    return meanAUC[0]


def createNegativeItemSet(userIDAndPosItemIDs, bAllItemIDs):
    allItemIDs = bAllItemIDs.value
    return map(lambda x: getNegativeItemsForSingleUser(x[0], x[1], allItemIDs), 
               userIDAndPosItemIDs)


def getNegativeItemsForSingleUser(userID, posItemIDs, allItemIDs):
    posItemIDSet = set(posItemIDs)
    negative = []
    i = 0
    # Keep about as many negative examples per user as positive.
    # Duplicates are OK
    while i < len(allItemIDs) and len(negative) < len(posItemIDSet):
        itemID = random.choice(allItemIDs) 
        if itemID not in posItemIDSet:
            negative.append(itemID)
        i += 1
    # Result is a collection of (user,negative-item) tuples
    return map(lambda itemID: (userID, itemID), negative)


# In[15]:


# building a model
# Note that there are some hyperparameters, that should be fitted during cross-validation 
# (here we use default values for all hyperparameters but rank) 
for regParam in (.01,.5):
    for alpha in (5,40,20,30,10):
        for rank in (5,10,15,20,25,30,35,40):
            model = ALS(implicitPrefs=True, userCol="userId", itemCol="artistId", ratingCol="count", 
                    rank=rank, alpha=alpha,regParam=regParam).fit(training)
            # predict test data
            predictions = model.transform(test)
            predictions.cache()
            predictions.take(3)
            print('r=',rank,'a=',alpha,'rp=',regParam,'meanAUC =', areaUnderCurve(test, bTopItemIDs, model.transform), 'for ALS-PREDICTION')

