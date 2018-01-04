
# coding: utf-8

# # Advanced Machine Learning (MScA, 32017)
# 
# # Project Recommending Music with Audioscrobbler Data
# 
# ### Yuri Balasanov, Mihail Tselishchev, &copy; iLykei 2017
# 
# ## Fitting ALS model to Audioscrobbler (LastFM) data

# In[1]:


from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DoubleType, IntegerType, StringType, Row
from pyspark.ml.recommendation import ALS
import pyspark.sql.functions as func
import random
import time
from datetime import datetime


# In[2]:


spark = SparkSession.builder.getOrCreate()
spark


# ## Data
# 
# Create paths to the data files. Add path to file with predictions for the test that will be calculated at the end of this notebook.  

# In[3]:


# paths to files
artistdata_path = './data/artist_data.csv'
userartist_path = './data/clean_15_5.csv'
test_path = './data/LastFM_Test_Sample.csv'


# In[4]:


# defining schemas
artistdata_struct = StructType([StructField('artistId', IntegerType()),                                 StructField('name', StringType())])
userartist_struct = StructType([StructField('userId', IntegerType()),                                 StructField('artistId', IntegerType()),                                 StructField('count', IntegerType())])


# In[5]:


# read artist names data
artistdata_df = spark.read.csv(artistdata_path, sep = '\t', schema = artistdata_struct)
artistdata_df.cache()
artistdata_df.show(10)


# In[6]:


# read user-artist data
userartist_df = spark.read.csv(userartist_path, sep = ',', header=True, schema = userartist_struct)
userartist_df.cache()
userartist_df.show(10)


# In[7]:


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

# In[8]:


# building a model
# Note that there are some hyperparameters, that should be fitted during cross-validation 
# (here we use default values for all hyperparameters but rank) 
t1 = time.perf_counter()
model = ALS(implicitPrefs=True, userCol="userId", itemCol="artistId", ratingCol="count", 
            rank=40, alpha=5,regParam=0.05).fit(training)
t2 = time.perf_counter()
print('Fitting time:', t2-t1)


# ## Predict test data
# 
# From the test shiny download your test sample.
# 
# Use it in the following cell to predict ratings, save the results as csv file and upload back to the test shiny for scoring.
# 
# Of course, predictions obtained without tuning hyperparameters and using small sample are not expected to be good.

# In[22]:


# reading test file
test_struct = StructType([StructField('userId', IntegerType()),                           StructField('artistId', IntegerType())])
test_df = spark.read.csv(test_path, sep = '\t', schema = test_struct)
test_df.show(10)


# In[23]:


# Note that many predictions are NaN since some users and artists might be out of 
# small train-data
# Full train file has to be used to avoid this.
# However, even using full train file, some users might be new. 
# What artists should we propose to them?
predictions = model.transform(test_df)
predictions.show(10)
assert predictions.count() == test_df.count()


# In[24]:


# Save test predictions to CSV-file
#timestamp = datetime.now().isoformat(sep='T', timespec='seconds')
#predictions.coalesce(1).write.csv('./data/test_predictions_{}.csv'.format(timestamp), 
#                                  sep = '\t')
predictions.coalesce(1).write.csv('test_predictions.csv', sep='\t')


# Check saved results in `./data` directory. <br>
# Solution is saved as a folder with multiple files. <br>
# There should be only one file .csv. Upload it in the test shiny.
