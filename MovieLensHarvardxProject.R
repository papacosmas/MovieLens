
#--------------------------------------------Rating recommender system through machine learning training --------------------------------------------------------------


# The purpose of this R project is to create a  rating recommender system through machine learning training. That recommender system will be able to predict 
# a users rating into a new movie. For training and testing our ML models, we will use the 10M (millions) row rating dataset named MovieLens created by the 
# University of Minnesota. It was released at 1/2009 so our newest movies are until 2008. In order to find a pattern and behavior of the data, the data sets where 
# enhanced by many new features (dimensions). As validation of the models we wil use RMSE. During the project are given more explanations. Many algorithms and 
# ML models where used in order to achieve the lowest RMSE. Such us: Matrix Factorization with parallel stochastic gradient descent, H2o stacked ensembles of 
# (GBM,GLM,DRF,NN). Also they where used H2o Auto ML models
# More details are below and also during the project. In case you dont want to wait and train the models, you can download them from my github and load them.
#
# There are 2 types of recommender systems: Content filtering (based on the description of the item - also called meta data or side information) 
# 
# And collaborative Filtering: Those techniques are calculating the similarity measures of the target ITEMS and finding the minimum (Euclidean distance,
# or Cosine distance, or other metric, it depends on the algorithm). This is done by filtering the interests of a user, by collecting preferences from many
# users (collaborating). The underlying assumption is that if a person X has the same opinion as a person Y then the recommendation system should be based 
# on preferences of person Y (similarity).
# 
# We will enhance the collaborative filtering with the help of Matrix factorization. MF is a class of collaborative filtering algorithms used in recommender 
# systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.
# This family of methods became widely known during the Netflix prize challenge due to its effectiveness as reported by Simon Funk in his 2006 blog post, where he
# shared his findings with the research community LINK https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)
#
# We will apply Matrix Factorization with parallel stochastic gradient descent. With the help of "recosystem" package it is an R wrapper of the LIBMF library 
# which creates a  Recommender System by Using Parallel Matrix Factorization.
# The main task of recommender system is to predict unknown entries in the rating matrix based on observed values.
#
#The main purpose is to calculate the matrix Rm?n by the product of the two matrixes of the lower dimension, Pk?m and Qk?n : RQ
# More info on the recosystem package and the techniques LINK https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html

# If you dont wish to train the models you can download them from git and load them.
# GITHUB LINK:https://github.com/papacosmas/MovieLens

#----------------------------------------------------------------------------------------------------------------------------------------------------------




#--------------------------------------------------------------------------------------------------------------
#--------Creation of the training = edx set, and the testing set = validation set, and submission file---------
#---------------------- Installation / loading of required packages, if needed---------------------------------
#--------------------------------------------------------------------------------------------------------------


# Note: this process could take a couple of minutes!!!

if(!require(tidyverse)) install.packages("tidyverse")
if(!require(caret)) install.packages("caret")
if(!require(kableExtra)) install.packages("kableExtra")
if(!require(dplyr)) install.packages("dplyr")
if(!require(ggplot2)) install.packages("ggplot2")
if(!require(lubridate)) install.packages("lubridate")
if(!require(h2o)) install.packages("h2o")
if(!require(stringr)) install.packages("stringr")
if(!require(recosystem)) install.packages("recosystem")
if(!require(tidyr)) install.packages("tidyr")
if(!require(wordcloud)) install.packages("wordcloud")
if(!require(GGally)) install.packages("GGally")
if(!require(ggthemes)) install.packages("ggthemes")
if(!require(stringr)) install.packages("stringr")
if(!require(lattice)) install.packages("lattice")
if(!require(rpart)) install.part("rpart")


require(tidyverse)
require(caret)
require(rmarkdown)
require(rpart)
require(lattice)
require(ggthemes)
require(GGally)
require(knitr)
require(tidyr)
require(wordcloud)
require(kableExtra)
require(dplyr)
require(ggplot2)
require(lubridate)
require(h2o)
require(stringr)
require(recosystem)


# Download the MovieLens 10M dataset:
# https://grouplens.org/datasets/movielens/10m/
# http://files.grouplens.org/datasets/movielens/ml-10m.zip

dl <- tempfile()
download.file("http://files.grouplens.org/datasets/movielens/ml-10m.zip", dl)

ratings <- read.table(text = gsub("::", "\t", readLines(unzip(dl, "ml-10M100K/ratings.dat"))),
                      col.names = c("userId", "movieId", "rating", "timestamp"))

movies <- str_split_fixed(readLines(unzip(dl, "ml-10M100K/movies.dat")), "\\::", 3)
colnames(movies) <- c("movieId", "title", "genres")
movies <- as.data.frame(movies) %>% mutate(movieId = as.numeric(levels(movieId))[movieId],
                                           title = as.character(title),
                                           genres = as.character(genres))

movielens <- left_join(ratings, movies, by = "movieId")

# Validation set will be 10% of MovieLens data

set.seed(1)
test_index <- createDataPartition(y = movielens$rating, times = 1, p = 0.1, list = FALSE)
edx <- movielens[-test_index,]
temp <- movielens[test_index,]

# Make sure userId and movieId in validation set are also in edx set

validation <- temp %>% 
  semi_join(edx, by = "movieId") %>%
  semi_join(edx, by = "userId")

# Add rows removed from validation set back into edx set

removed <- anti_join(temp, validation)
edx <- rbind(edx, removed)

rm(dl, ratings, movies, test_index, temp, movielens, removed)




#--------------------------------------------------------------------------------------------------------------
#------------------------------ Data Observation Part----------------------------------------------------------
#--------------------------------------------------------------------------------------------------------------



#First we check all columns in our training & test datasets for NA's and for Inf values (it will take some time!)
apply(edx, 2, function(x) any(is.na(x) | is.infinite(x)));

apply(validation, 2, function(x) any(is.na(x) | is.infinite(x)));
# userId   movieId    rating timestamp     title    genres 
# FALSE     FALSE     FALSE     FALSE     FALSE     FALSE 
# both sets dont contain NA's .and inf's


str(edx) 
# Its class'data.frame': with 	9000047 obs. of  6 variables:
# We have 6 features. It looks we have to transform the timestamp which should be the rating date, because the release date is on the title,
# which we also should extract it. Also we will create a new matrix with more dimensions containing every genre separately  as factor.
# Keep in mind that one movie might belong to more than one genre. That means that when we will create the new data frame with discrete  genres the sum of the movies will be greater than in edx data set
head(edx)
# userId movieId rating timestamp                         title                        genres
# 1      1     122      5 838985046              Boomerang (1992)                Comedy|Romance
# 2      1     185      5 838983525               Net, The (1995)         Action|Crime|Thriller

summary(edx)
# userId         movieId          rating        timestamp            title              genres         
# Min.   :    1   Min.   :    1   Min.   :0.500   Min.   :7.897e+08   Length:9000047     Length:9000047    
# 1st Qu.:18124   1st Qu.:  648   1st Qu.:3.000   1st Qu.:9.468e+08   Class :character   Class :character  
# Median :35738   Median : 1834   Median :4.000   Median :1.035e+09   Mode  :character   Mode  :character  
# Mean   :35870   Mean   : 4122   Mean   :3.512   Mean   :1.033e+09                                        
# 3rd Qu.:53607   3rd Qu.: 3626   3rd Qu.:4.000   3rd Qu.:1.127e+09                                        
# Max.   :71567   Max.   :65133   Max.   :5.000   Max.   :1.231e+09    
# the rating mean shows that users are rating above average rating (3.512)
# The rating (our dependent variable y) has 10 continuous values from 0 until 5. Its row has one given rating by one user for one movie. Rating is our dependent (target variable) y
# userId, movieId, timestamp (date&time) are: quantitative - Discrete unique numbers.
# Title and genres are: qualitative and not unique.

str(validation)
#Its class 'data.frame': With	999999 obs. of  6 variables: Its exactly the 10% of our training set. And has the same 6 features

head(validation)
# Same as our training set. So we will perform the same data transformation  on both training and test datasets

# See the distinct number of users and distinct number of movies in our train set
edx %>%
  summarize(distinct_users = n_distinct(userId),
            distinct_movies = n_distinct(movieId))
# distinct_users distinct_movies
# 1          69878           10669


# We create a new df with all the useful  metrics in order to understand better our dataset
# Identify  outliers. Packages used (tidyr),(dplyr)
edx_movies_metrics <- edx %>%
  separate_rows(genres,
                sep = "\\|") %>%
  group_by(genres) %>%
  summarize(Ratings_perGenre_Sum = n(),
            Ratings_perGenre_Mean = mean(rating),
            Movies_perGenre_Sum = n_distinct(movieId),
            Users_perGenre_Sum = n_distinct(userId));


edx_movies_metrics;
# We notice that the rating mean is not rounded  so we will fix it. Also we identify in our new edx movies metrics df that there is one movie without genres.
# We will treat it as an outlier and delete it from all our datasets, since it doesnt add any value. We also have 19 distinct genres.       
edx_movies_metrics$Ratings_perGenre_Mean <- round(edx_movies_metrics$Ratings_perGenre_Mean, digits = 2);

edx_movies_metrics <- subset(edx_movies_metrics, genres!="(no genres listed)");

edx_movies_metrics[order(-edx_movies_metrics$Movies_perGenre_Sum),]
# We see that most movies per genre are : 1) Drama , 2) Comedy 3) Thriller. Keep in mind this is not unique movies. Because as we show earlier one movie might
# belong to more than one genre
# genres          Ratings_perGenre_Sum    Ratings_perGenre_Mean   Movies_perGenre_Sum Users_perGenre_Sum
# <chr>                      <int>                 <dbl>               <int>              <int>
# 1 Drama                    3910124                  3.67                5333              69866
# 2 Comedy                   3540928                  3.44                3701              69864
# 3 Thriller                 2325897                  3.51                1703              69567

edx_movies_metrics[order(-edx_movies_metrics$Ratings_perGenre_Sum),]
# genres      Ratings_perGenre_Sum      Ratings_perGenre_Mean   Movies_perGenre_Sum   Users_perGenre_Sum
# 1 Drama                    3910124                  3.67                5333              69866
# 2 Comedy                   3540928                  3.44                3701              69864
# 3 Action                   2560544                  3.42                1472              69607
# Here we see that the top 3 genres with the most ratings are Drama, Comedy and Thriller
# Some genres have exponential low sum of ratings so probably they will be also treated as outliers in the data frame that we will create with all genres as factors

edx_movies_metrics[order(-edx_movies_metrics$Ratings_perGenre_Mean),]
# genres      Ratings_perGenre_Sum      Ratings_perGenre_Mean      Movies_perGenre_Sum      Users_perGenre_Sum
# <chr>                      <int>                 <dbl>               <int>              <int>
# 1 Film-Noir                 118541                  4.01                 148              31270
# 2 Documentary                93064                  3.78                 479              24295
# 3 War                       511147                  3.78                 510              64892
# 4 IMAX                        8181                  3.77                  29               6393
# 5 Mystery                   568332                  3.68                 509              61845
# Here we can notice that genres with low sum of ratings have higher rating mean. This is one more indicator that should be treated as outliers.
# Also movies with low sum of ratings will be removed from the training set for the same reasons


# We create the same df with the metrics for the validation dataset (test data) to see if it is representative of our training data
validation_movies_metrics <- validation %>%
  separate_rows(genres,
                sep = "\\|") %>%
  group_by(genres) %>%
  summarize(Ratings_perGenre_Sum = n(),
            Ratings_perGenre_Mean = mean(rating),
            Movies_perGenre_Sum = n_distinct(movieId),
            Users_perGenre_Sum = n_distinct(userId))


# We notice that the rating mean is not rounded so we will fix it. Also we identify in our new edx movies metrics df that there is one movie without genres.
# We will treat it as an outlier and delete it from all our datasets, since it doesnt add any value            
validation_movies_metrics$Ratings_perGenre_Mean <- round(validation_movies_metrics$Ratings_perGenre_Mean, digits = 2);
validation_movies_metrics <- subset(validation_movies_metrics, genres!="(no genres listed)");

validation_movies_metrics[order(-validation_movies_metrics$Movies_perGenre_Sum),]
# We can see that our validation set is representative to our training set

invisible(invisible(gc()))#we clear unused  memory from R. (We will do that frequently)

# We will create a new dataframe to analyze  the ratings distribution
ratings_distribution <- edx %>%
  group_by(rating) %>%
  summarise(ratings_distribution_sum = n()) %>%
  arrange(desc(ratings_distribution_sum));


ratings_distribution;
# `edx$rating` ratings_distribution_sum
# 1          4                    2588429
# 2          3                    2121238
# 3          5                    1390114
# 4          3.5                   791623
# 5          2                     711420
# 6          4.5                   526736
# 7          1                     345679
# 8          2.5                   333009
# 9          1.5                   106426
# 10          0.5                    85373

# We can see that the highest sum of ratings are on rating 4. So audience tend not to rate very strict.
# We also noticed that there is not a movie with rating 0


#display the ratings distribution results with the help of kableExtra package
kable(ratings_distribution) %>%
  kable_styling(bootstrap_options = "striped" , full_width = F , position = "center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold = T ,color = "red", background = "lightgrey" )

# We use the below option in our r session in order to prevent the scientific notation in our plots " A penalty to be applied when deciding to print
#numeric values in fixed or exponential notation.  Positive values bias towards fixed and negative towards scientific notation: fixed notation will be preferred unless it is more  than 'scipen' digits wider
# this step is optional
options(scipen=999)

#create object with the ratings mean
ratings_mean <- mean(edx$rating)

# We will also plot a histogram of the ratings distribution to have it as a visualization 
ggplot(edx,aes(rating, fill=cut(rating, 100))) +
  geom_histogram(color = "blue",binwidth = 0.2) +
  scale_x_continuous(breaks = seq(0.5, 5, 0.5)) +
  geom_vline(xintercept = ratings_mean,
             col = "red",
             linetype = "dashed") +
  labs(title = "Distribution of ratings",
       x = "Ratings Scale",
       y = "Sum Of Rating") +
  theme(plot.title = element_text(size = 13, color = "darkblue", hjust = 0.5))


invisible(invisible(gc()))#we clear unused  memory from R. (We will do that frequently)


# For the training of our ML algorithms we want to penalized  movies rated by low number of users.So inn order to put more weight on movies that have been rated by more people, we will add 2 more features in our data sets. 
# Number of users per movie, and number of movies per user( how many movies had that user rated ). So movies and users that have not many rates will be penalized during the ML training
#RUN Scripts one by one!
edx <- edx %>%
  group_by(userId) %>%
  mutate(number_movies_byUser = n());

#RUN Scripts one by one!
edx <- edx %>%
  group_by(movieId) %>%
  mutate(number_users_byMovie = n()); 

#RUN Scripts one by one!
validation <- validation %>%
  group_by(userId) %>%
  mutate(number_movies_byUser = n());

#RUN Scripts one by one!
validation <- validation %>%
  group_by(movieId) %>%
  mutate(number_users_byMovie = n()); 

#We will plot the most rated movies. Only those that have been rated over 20.000 times (takes time)
edx %>% filter(number_users_byMovie >= 20000) %>%
  ggplot(aes(reorder(title, number_users_byMovie), number_users_byMovie, fill = number_users_byMovie)) +
  geom_bar(stat = "identity") + coord_flip() + scale_fill_distiller(palette = "PuBuGn") + xlab("Movie Title") +ylab('Number of Ratings') +
  theme_solarized(light=FALSE) +
  ggtitle("Most rated movies")

invisible(invisible(gc()))#we clear unused  memory from R. (We will do that frequently)

# We will plot a bar graph, in order to understand the distribution of the sum of ratings per genres,and movies per genres
ggplot(data=edx_movies_metrics, aes(x= reorder(genres, -Ratings_perGenre_Sum),
                                    y = Ratings_perGenre_Sum,colour= "red", fill=genres, label= Ratings_perGenre_Sum)) + 
  xlab("Movies genres") + ylab("Sum of ratings in Mio") +
  ggtitle("Sum of ratings per genre") +
  scale_y_continuous(labels = scales::comma) +
  geom_text( vjust =-1,color = "white",check_overlap = T) +
  theme_solarized(light=FALSE) +
  geom_col()

invisible(invisible(gc()))#we clear unused  memory from R. (We will do that frequently)

# In the bar graph is also clear that the distribution is not equal in all genres. After the fantasy genre, we can see the exponential growth in the number of ratings.
# That is an indication that probably we wil penalize the genres with low number of ratings, in our model

# We plot a word count cloud with the most rated genres
wordcloud(words = edx_movies_metrics$genres, freq = edx_movies_metrics$Ratings_perGenre_Sum, min.freq = 10,
          max.words=10, random.order=FALSE,random.color=FALSE, rot.per=0.35,scale=c(5,.2),font = 4, 
          colors=brewer.pal(8, "Dark2"),
          main = "Most rated genres")


#We will plot a bar graph to see the distribution of ratings mean per genres
ggplot(data=edx_movies_metrics, aes(x= reorder(genres, - Ratings_perGenre_Mean),
                                    y = Ratings_perGenre_Mean,colour= Ratings_perGenre_Sum, fill=genres, label= Ratings_perGenre_Mean)) + 
  xlab("Movies genres") + ylab("Mean of ratings") +
  ggtitle("Mean of ratings per genre") +
  geom_text( vjust =-1,color = "black",check_overlap = TRUE) + 
  theme(plot.title = element_text(hjust = 4)) +
  geom_col()

# As we mentioned earlier we could add features in our datasets in order to analyse for correlations, and if they exist they would help our ML models. By adding a feature
# of release year, and rating year. So we will create 2 new dataframes with those 2 new features for train and test sets.
# Extract the rating year from timestamp and add it as a new column (year_rated). The same for the release year. Packages stringr and lubridate
# Step1 (Takes Time)
edx_matrix <- edx  %>%  
  mutate( year_rated = year(as_datetime(timestamp)))  %>%
  mutate(release_year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$"))

# for validation set (Takes Time)
validation_matrix <- validation  %>%  
  mutate( year_rated = year(as_datetime(timestamp)))  %>%
  mutate(release_year = as.numeric(str_extract(str_extract(title, "[/(]\\d{4}[/)]$"), regex("\\d{4}"))),title = str_remove(title, "[/(]\\d{4}[/)]$"))

# Step 2 extract the genres as unique factors ONCE for both sets.
edx_matrix$genres %>% str_split(pattern = "\\|") -> genres ;

# Step 3 Create the unique genres value
genres %>% unlist() %>% unique() -> genres_unique 

# Step 4 add the unique genres into the edx_matrix as factors
#num of cols
ncol(edx_matrix)

# create value with num columns
cols <- ncol(edx_matrix)

#add the genres as new dimensions (takes time)
for(i in seq_along(genres_unique)){
  id <- grepl(pattern = genres_unique[i] , edx_matrix$genres)
  edx_matrix[[cols + i]] <- 0
  edx_matrix[[cols + i]][id] <- 1
}

# Put the correct genre names into the new dimension
names(edx_matrix)[(cols+1):ncol(edx_matrix)] <- genres_unique




#We also add the unique genres into the validation_matrix as factors 
#num of cols
ncol(validation_matrix)
# create value with num columns
cols <- ncol(validation_matrix)

#add the genres as new dimensions 

for(i in seq_along(genres_unique)){
  id <- grepl(pattern = genres_unique[i] , validation_matrix$genres)
  validation_matrix[[cols + i]] <- 0
  validation_matrix[[cols + i]][id] <- 1
}

# Put the correct genre names into the new dimension
names(validation_matrix)[(cols+1):ncol(validation_matrix)] <- genres_unique

invisible(invisible(gc()))#we clear unused  memory from R. (We will do that frequently)

# Instead of having the year released we will create a new feature with how old is every movie and drop the year released
edx_matrix <- edx_matrix %>% mutate(age_of_movie = 2019 - release_year);
# And for the test set
validation_matrix <- validation_matrix %>% mutate(age_of_movie = 2019 - release_year)

# Check both data sets if the schema has changed
str(edx_matrix)
str(validation_matrix)

# To save RAM we drob the objects we dont need anymore
rm(genres,genres_unique,i,id,cols)
invisible(invisible(gc()))#clean RAM

# We noticed a column with no genres that contains only one movie in edx matrix, we will delete that column (outlier)
# also we will delete the columns (genres) with low sum of ratings. The reasons for this is because the train set is already big 9000047 rows
# so we dont want to have many dimensions during the models building. The second reason is to prevent overfidding

edx_matrix <- edx_matrix %>% select(-release_year, -title, -timestamp, -genres, -Western, -IMAX, -Documentary,-`Film-Noir`, -Musical,-Animation,-War,-Mystery,-Horror,-Children,-Fantasy, -"(no genres listed)") 
# Also for test set
validation_matrix <- validation_matrix %>% select(-release_year, -title, -timestamp, -genres, -Western, -IMAX, -Documentary,-`Film-Noir`, -Musical,-Animation,-War,-Mystery,-Horror,-Children,-Fantasy, -"(no genres listed)") 

# In our data sets the variation of the age of the movies starts from 11 years (that means 2008- this is the last year we have movies until 104 then are the oldest movies we have).
# We will check if there is a correlation between age of movie and ratings. First we will create a new object,in order to observe easier, and also
# plot faster and with less memory!

avg_rating_per_oldness <- edx_matrix %>% 
  group_by(age_of_movie) %>% 
  summarize(avg_rating_by_age = mean(rating)) 


# We will examine if there is a correlation between age of movie and rating. With the help of package GGally 
ggpairs(avg_rating_per_oldness, 
        mapping= aes(color = "age_of_movie"),    
        title="Age of Movie VS Rating correlation", 
        upper = list(continuous = wrap("cor", 
                                       size = 10)), 
        lower = list(continuous = "smooth"))

# We can clearly notice that there is a positive  trend. The oldest the movie the highest  the ratings it receives.
# This is due to 2 reasons. First the oldest the movie the more ratings it has. Second usually  the old movies are consider classics and they are rated better by the audience.
# It looks that the age of movie will have low p value in or ML models training. We create one more plot to demonstrate it

avg_rating_per_oldness %>%
  ggplot(aes(x = age_of_movie, y = avg_rating_by_age)) +
  ggtitle("Age of Movies vs Rating") +
  geom_line(color="yellow") +
  theme_solarized(light=FALSE) +
  geom_smooth(color="grey")


# We will also check if there is a correlation between the year that the film was rated and the rating
# The films rated year varies from 1995 until 2009.
avg_rating_per_yearRated <- edx_matrix %>% 
  group_by(year_rated) %>% 
  summarize(avg_rating = mean(rating)) 

# We will plot to check for correlation. 
avg_rating_per_yearRated %>%
  ggplot(aes(x = year_rated, y = avg_rating)) +
  ggtitle("Age of Movies vs Rating") +
  geom_line(color="yellow") +
  theme_solarized(light=FALSE) +
  geom_smooth(color="grey")
# We notice that the oldest the movie was rated (1995) the highest was the rating. Same correlation as the age of the movie

# Lets now check if there is also a correlation between the year rated and rating
ggpairs(edx_matrix, columns= 3:5,
        ggplot2::aes(color = "red"),    
        title="Age of Movie VS Rating correlation", 
        upper = list(continuous = wrap("cor", 
                                       size = 10)), 
        lower = list(continuous = "smooth"))


# We will also check the other features for correlations. We will use GGally package
ggcorr(edx_matrix, 
       label = TRUE, 
       label_alpha = TRUE)
# We can see a strong correlation on the genres with the highest number of ratings ()

invisible(gc())#free ram




#------------------------------------------------------------------------------------------------------------------------------------

#                         Matrix Factorization with parallel stochastic gradient descent
#                                Model Building Training and Validation Step

#------------------------------------------------------------------------------------------------------------------------------------


#------------------------------------------------------------------------------------------------------------------------------------
# There are 2 types of recommender systems: Content filtering (based on the description of the item - also called meta data or side information) 
# 
# And collaborative Filtering: Those techniques are calculating the similarity measures of the target ITEMS and finding the minimum (Euclidean distance,
# or Cosine distance, or other metric, it depends on the algorithm). This is done by filtering the interests of a user, by collecting preferences from many
# users (collaborating). The underlying assumption is that if a person X has the same opinion as a person Y then the recommendation system should be based 
# on preferences of person Y (similarity).
# 
# We will enhance the collaborative filtering with the help of Matrix factorization. MF is a class of collaborative filtering algorithms used in recommender 
# systems. Matrix factorization algorithms work by decomposing the user-item interaction matrix into the product of two lower dimensionality rectangular matrices.
# This family of methods became widely known during the Netflix prize challenge due to its effectiveness as reported by Simon Funk in his 2006 blog post, where he
# shared his findings with the research community LINK https://en.wikipedia.org/wiki/Matrix_factorization_(recommender_systems)
#
# We will apply Matrix Factorization with parallel stochastic gradient descent. With the help of "recosystem" package it is an R wrapper of the LIBMF library 
# which creates a  Recommender System by Using Parallel Matrix Factorization.
# The main task of recommender system is to predict unknown entries in the rating matrix based on observed values.
#
#The main purpose is to calculate the matrix Rm?n by the product of the two matrixes of the lower dimension, Pk?m and Qk?n : RQ
# More info on the recosystem package and the techniques LINK https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html
#------------------------------------------------------------------------------------------------------------------------------------


#Before we proceed with the model building,training and validation we define the RMSE function 
RMSE <- function(true_ratings, predicted_ratings){
  sqrt(mean((true_ratings - predicted_ratings)^2))
}

#The data file for training set needs to be arranged in sparse matrix triplet form, i.e., each line in the file contains three numbers
# in order to use recosystem package we create 2 new matrices (our train and our validation set) with the below 3 features 
edx_factorization <- edx %>% select(movieId,userId,rating)
validation_factorization <- validation %>% select(movieId,userId,rating)

# we also need to convert them as matrix
edx_factorization <- as.matrix(edx_factorization)
validation_factorization <- as.matrix(validation_factorization)

#recosystem needs to save the edx_factorization and validation_factorization tables on hard disk, recosystem package needed.
write.table(edx_factorization , file = "trainingset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)
write.table(validation_factorization, file = "validationset.txt" , sep = " " , row.names = FALSE, col.names = FALSE)

#  We use the function data_file(): This function specifies data, stored into the hard disk 
set.seed(1)
training_dataset <- data_file( "trainingset.txt")
validation_dataset <- data_file( "validationset.txt")

# We reate a model object (a Reference Class object in R) by calling the function Reco().
r = Reco()

# This step is optional. We call the $tune() method to select the best tuning parameters (along a set of candidate values).
opts = r$tune(training_dataset, opts = list(dim = c(10, 20, 30), lrate = c(0.1, 0.2),
                                            costp_l1 = 0, costq_l1 = 0,
                                            nthread = 1, niter = 10))
opts

# Now we train the model by calling the $train() method. A number of parameters can be set inside the function, 
# coming from the result of the previous step - $tune().
r$train(training_dataset, opts = c(opts$min, nthread = 1, niter = 20))

# We write predictions to a tempfile on HDisk
stored_prediction = tempfile() 

#With the $predict() method we will make  predictions on validation set and will calculate RMSE:
r$predict(validation_dataset, out_file(stored_prediction))  

#We display the 20 first predictions
print(scan(stored_prediction, n = 20))
#[1] 4.09517 4.92754 4.74056 3.34579 4.40629 2.82392 4.09782 4.35618 4.34707 3.38473 3.56202 3.88267 3.34036 4.06874 3.50337
#[16] 4.53853 3.76697 2.94648 3.92723 3.84097


# We read the validation data set as a file in table format and we create a data frame from it
real_ratings <- read.table("validationset.txt", header = FALSE, sep = " ")$V3
pred_ratings <- scan(stored_prediction) # We create an object from the stored predictions in order to evaluate our model

#  We calculate the standard deviation of the residuals (prediction errors) RMSE. Between the predicted ratings and the real ratings 
rmse_of_model_mf <- RMSE(real_ratings,pred_ratings)
rmse_of_model_mf  # 
#[1] 0.7825707

# We see that the RMSE is extremely low. And possibly until know the Matrix factorization with SGD
# is the best approach to create a recommender system. I would like to thank Yu-Chin Juan, Wei-Sheng Chin, Yong Zhuang 
# for creating the LIMBF library but also Yixuan Qiu that created the R wrapper LINK https://cran.r-project.org/web/packages/recosystem/vignettes/introduction.html

# We will compare the first 50 predictions of the MF model with the real ratings
# First we round the predictions for visualization convenience
pred_ratings_rounded <-  pred_ratings;

pred_ratings_rounded <- round(pred_ratings_rounded/0.5) *0.5;

MF_first50_pred <- data.frame(real_ratings[1:50],pred_ratings_rounded[1:50]);

names(MF_first50_pred) <- c("real_ratings","predicted_ratings");

MF_first50_pred <- MF_first50_pred %>%
  mutate(correct_predicted = as.numeric(real_ratings == predicted_ratings))


# We will plot the 50 first predicted ratings of the MF model. The blue are the correct predictions
ggplot(data=MF_first50_pred, aes(x=real_ratings, y = predicted_ratings,colour=correct_predicted)) + 
  xlab("Real Ratings") + ylab("Predicted Ratings") +
  ggtitle("Real vs Predicted Ratings") +
  theme(plot.title = element_text(size = 12, color = "darkblue", hjust = 0.5)) +
  geom_jitter()

# We will also display the 50 first predictions
kable(MF_first50_pred) %>%
  kable_styling(bootstrap_options = "bordered" , full_width = F , position = "center") %>%
  column_spec(1,bold = T,background = "lightgrey" ) %>%
  column_spec(2,bold = T , background = "lightgrey" )  %>%
  column_spec(3,bold = T ,color = "red" , background = "grey" )

#Save the RMSE of various models into one DF . So in the end we will have them all stored in one file
all_models_rmse_results <- data.frame(Algorithm=c("Matrix factorization with SGD"),RMSE = c(rmse_of_model_mf))

#display the results with the help of kable package
kable(all_models_rmse_results) %>%
  kable_styling(bootstrap_options = "bordered" , full_width = T , position = "center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold = T ,color = "red", background = "lightgrey" )

# IMPORTANT! if you will not work with the above model any more you can remove the below sets to save ram
# rm(edx_factorization,validation_factorization,edx,validation)





#--------------------------------------------------------------------------------------------------------------
#---------------------- H2o ---- Machine learning training, build, validating part------------------------------
#--------------------------------------------------------------------------------------------------------------


#--------------------------------------------------------------------------------------------------------------
#--------- H2o open source machine learning and artificial intelligence platform ---------------------------
# We will try with the help of H2o to achieve a smaller RMSE if its possible. We will create a local  H2o cluster
# to run our scripts. In order to use h2o you first have to download on your machine (not in r) the h2o jar file from
# http://h2o-release.s3.amazonaws.com/h2o/rel-xu/1/index.html
# then write from cmd : cd Downloads      : the path might be different
# unzip h2o-3.22.1.1.zip      : if you will load the models from my github, you will need that specific version, 
# cd h2o-3.22.1.1              : if you train the models again, then you can use newer version
# java -jar h2o.jar
#--------------------------------------------------------------------------------------------------------------


# The below 2 commands remove any previously installed H2O packages in R.
if ("package:h2o" %in% search()) { detach("package:h2o", unload=TRUE) }
if ("h2o" %in% rownames(installed.packages())) { remove.packages("h2o") }

# download r h2o packages.
pkgs <- c("RCurl","jsonlite")
for (pkg in pkgs) {
  if (! (pkg %in% rownames(installed.packages()))) { install.packages(pkg) }
}

# download, install the H2O package for R.
install.packages("h2o", type="source", repos="http://h2o-release.s3.amazonaws.com/h2o/rel-xu/1/R")

#  load H2O lib and start up the  H2O cluster CAUTION to load my h2o models you have to install the same h2o version h2o-3.22.1.1, check above
library(h2o)
h2o.init(nthreads = -1,        ## -1: use all available threads         
         ignore_config = TRUE,
         max_mem_size = "8G") # You can change it , it depends on your machine RAM


# i create new data sets to use them with H2o clusters (need to be h2o data frames)
edx_h2o <- edx_matrix

# in h2o the dependent variables need to be factors
edx_h2o$userId <-  as.factor(edx_h2o$userId)
edx_h2o$movieId <- as.factor(edx_h2o$movieId)
edx_h2o$Drama <- as.factor(edx_h2o$Drama)
edx_h2o$Comedy <- as.factor(edx_h2o$Comedy)
edx_h2o$age_of_movie <- as.factor(edx_h2o$age_of_movie)
edx_h2o$Thriller <- as.factor(edx_h2o$Thriller)
edx_h2o$Action <- as.factor(edx_h2o$Action)
edx_h2o$Adventure <- as.factor(edx_h2o$Adventure)
edx_h2o$Crime <- as.factor(edx_h2o$Crime)
edx_h2o$`Sci-Fi` <- as.factor(edx_h2o$`Sci-Fi`)
edx_h2o$year_rated <- as.factor(edx_h2o$year_rated)

#we do the same for the validation set
validation_h2o <- validation_matrix

validation_h2o$userId <-  as.factor(validation_h2o$userId)
validation_h2o$movieId <- as.factor(validation_h2o$movieId)
validation_h2o$Drama <- as.factor(validation_h2o$Drama)
validation_h2o$Comedy <- as.factor(validation_h2o$Comedy)
validation_h2o$age_of_movie <- as.factor(validation_h2o$age_of_movie)
validation_h2o$Thriller <- as.factor(validation_h2o$Thriller)
validation_h2o$Action <- as.factor(validation_h2o$Action)
validation_h2o$Adventure <- as.factor(validation_h2o$Adventure)
validation_h2o$Crime <- as.factor(validation_h2o$Crime)
validation_h2o$`Sci-Fi` <- as.factor(validation_h2o$`Sci-Fi`)
validation_h2o$year_rated <- as.factor(validation_h2o$year_rated)


#Attempts to start and/or connect to and H2O instance 
require(h2o)

h2o.init(ignore_config=TRUE,
         nthreads=-1,        ## -1: use all available threads         
         max_mem_size = "20G") ## specify the memory size for the H2O cluster Adjust it according your machine

#After the start of the cluster optionally we can access it from browser to http://localhost:54321 # I recommend to try it. Play also with pojo saving of models

#partitioning of the training set. H2o works with h2oframes
# RMSE will be calculated on validation set and not on the below split
splits3 <- h2o.splitFrame(as.h2o(edx_h2o), # in order to test the algorithms during training, you can change the ratio if you wish
                          ratios = 0.7, 
                          seed = 1) 

train3 <- splits3[[1]] 
test3 <- splits3[[2]] # The split of our training set. (not the real testing set = test_validation)

test_validation <- as.h2o(validation_h2o)# We also create an h2o frame for the REAL validation set )

h2o.clusterInfo() # verify that h2o cluster is running

#IN CASE YOU DONT WISH TO RUN THE MODELS you can download them from github and load them with the h2o command # model_name <- h2o.loadModel(model_path)
# Ex. type YOUR full path # h2o_glm = h2oensemble2("C:/Users/npapaco/Documents/R_Projects/h2oensemble2")
# CAUTION YOU NEED TO DO FIRST THE STEPS ABOVE. Create the Splits 3, the train3, test3 and test_validation SETS



#----------------------------------------------------------------------------------------------------------------------------
# We start with h2o automl - H2O's AutoML can be used for automating the machine learning workflow, which includes automatic training 
# and tuning of many models within a user-specified time-limit. Stacked Ensembles - one based on all previously trained models, another 
# one on the best model of each family - will be automatically trained on collections of individual models to produce highly predictive 
# ensemble models which, in most cases, will be the top performing models in the AutoML Leaderboard. LINK http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

# you can download the model from github in your wd and load it  with the below command
#If you place it in your working directory, you just need to type /ModelIDname  as in my below examples

# h2oensemble2 = h2o.loadModel("/h2oensemble2")
# if above is not working type YOUR full path #h2o_glm = h2oensemble2("C:/Users/npapaco/Documents/R_Projects/h2oensemble2")

# If for some reason you need to shut down the h2o cluster
# h2o.shutdown(prompt = TRUE)

#----------------------------------------------------------------------------------------------------------------------------






#-------------------------------------H2O's AutoML---------------------------------------------------------------------------------------

# H2O's AutoML can be used for automating the machine learning workflow, which includes automatic training and tuning of many 
# models within a user-specified time-limit. Stacked Ensembles - one based on all previously trained models, another one on the
# best model of each family - will be automatically trained on collections of individual models to produce highly predictive 
# ensemble models which, in most cases, will be the top performing models in the AutoML Leaderboard LINK http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html

#----------------------------------------------------------------------------------------------------------------------------

h2oamlmodel2 <- h2o.automl ( x = c("movieId","n.users_bymovie","age_of_movie","Drama","Comedy") ,#you cyn add more features it depends on the time you wish to run the model
                             y = "rating" , 
                             training_frame = train3,#training dataset
                             leaderboard_frame = test_validation,# testing data set (in this cased i used the real testing set , and not the split of the training)
                             project_name = "movielens",#optional
                             max_models = 20,#maximum different models to try, automl will stop if max models is reached or if first max runtime is reached
                             max_runtime_secs = 3600,#  max time that automl will run, you can change it if you wish
                             nfolds = 3,#3-fold cross-validation on training data (Metrics computed for combined holdout predictions)
                             keep_cross_validation_predictions= TRUE,#  to validate the stability of our machine learning model
                             seed = 1) # for reproducibility purposes

invisible(gc())#free ram
h2oamlmodel2@leader 
# we see the model that had the lowest RMSE on the leaderboard .The (leader) of all models tested was the below.
# Model ID:  DRF_1_AutoML_20190424_155200 
# Algorithm:	Algorithm:	Distributed Random Forest

# If you want to use this model without retraining again, download it from github in your wd. Then start your
# H2o cluster (make sure to use same h2o version). And then type in r. h2oamlmodel2 = h2o.loadModel("/DRF_1_AutoML_20190424_155200")
# if above is not working type your full path #h2oamlmodel2 = h2o.loadModel("C:/Users/npapaco/Documents/R_Projects/DRF_1_AutoML_20190424_155200")
# CAUTION YOU NEED TO DO FIRST THE STEPS ABOVE. Create the Splits 3, the train3, test3 and test_validation SETS

# Model Summary: 
#   number_of_trees number_of_internal_trees model_size_in_bytes min_depth max_depth mean_depth min_leaves max_leaves mean_leaves
# 1              50                       50              185395         1        20   14.40000          2        680   160.96000
# 
# H2ORegressionMetrics: drf
# ** Reported on training data. **
#   ** Metrics reported on Out-Of-Bag training samples **
#   
#   MSE:  1.01896
# RMSE:  1.009436
# MAE:  0.8091325
# RMSLE:  0.2679033
# Mean Residual Deviance :  1.01896

h2oamlmodel2@leaderboard #we see the 6 best models from the leaderboard
# model_id mean_residual_deviance     rmse      mse       mae     rmsle
# 1              DRF_1_AutoML_20190424_155200               1.077967 1.038252 1.077967 0.8410296 0.2702502
# 2              XRT_1_AutoML_20190424_155200               1.088622 1.043370 1.088622 0.8475965 0.2717722
# 3 GLM_grid_1_AutoML_20190424_155200_model_1               1.099932 1.048777 1.099932 0.8462009 0.2689410
# 4              GBM_4_AutoML_20190424_155200               1.112490 1.054746 1.112490 0.8592683 0.2673023
# 5              GBM_2_AutoML_20190424_155200               1.116864 1.056818 1.116864 0.8620348 0.2674363
# 6 GLM_grid_1_AutoML_20190424_142708_model_1               1.117170 1.056963 1.117170 0.8571898 0.2725451

# OUTPUT - VARIABLE IMPORTANCES
# variable	relative_importance	scaled_importance	percentage
# n.users_bymovie	7164049.0	1.0	0.3019
# Drama	5418105.0	0.7563	0.2283
# movieId	5250999.0	0.7330	0.2212
# age_of_movie	5224642.5000	0.7293	0.2201
# Comedy	675853.8125	0.0943	0.0285


#We print the scoring history
h2o.scoreHistory(h2oamlmodel2@leader)


#We plot the training history, metric (rmse)
plot(h2oamlmodel2@leader, timestep = "number_of_trees", metric = "rmse")


# We plot the variables Importance (by order)
h2o.varimp_plot(h2oamlmodel2@leader, num_of_features = NULL)



# evaluate performance on test set (this is the split from training set and not the REAL validation set)
 h2o.performance(h2oamlmodel2@leader, test3) #RMSE:  1.008709
 
# If you dont re train the model and you load it from hard disk the run  
 h2o.performance(h2oamlmodel2@leader, test3)
 
 h2oamlmodel2_pred <- h2o.predict(h2oamlmodel2@leader, test_validation)  # evaluate on the validation set
 # If you dont re train the model and you load it from hard disk  h2oamlmodel2_pred <- h2o.predict(h2oamlmodel2, test_validation)

 rmse_of_model_h2oamlmodel2 <- RMSE(h2oamlmodel2_pred, as.h2o(test_validation$rating))# evaluate on the validation set
 rmse_of_model_h2oamlmodel2 

#Save the RMSE of various models into one DF . So in the end we will have them all stored in one file
all_models_rmse_results <- data.frame(Algorithm=c("Matrix factorization with SGD","H2o Auto ML model2"),RMSE = c(rmse_of_model_mf,rmse_of_model_h2oamlmodel2))

#display the RMSE results with the help of kable package
kable(all_models_rmse_results) %>%
  kable_styling(bootstrap_options = "bordered" , full_width = T , position = "center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold = T ,color = "red", background = "lightgrey" )

invisible(gc())#free ram


#--------------------------------H2o  Generalized Linear Models (GLM)  --------------------------------------------------------------------------------------------

#Because on autoML we cant tune many hyperparameters, we will also try on other models with various tunings. Then 
# we will stack those different models stackedEnsemble. Ensemble machine learning methods use multiple learning algorithms to obtain better predictive 
# performance than could be obtained from any of the constituent learning algorithms
# We start with Generalized Linear Models (GLM) estimate regression models for outcomes following exponential distributions. In addition to the Gaussian (i.e. normal) distribution, 
# these include Poisson, binomial, and gamma distributions. LINK http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/glm.html

#----------------------------------------------------------------------------------------------------------------------------

h2o_glm <- h2o.glm( x = c("movieId","userId","n.users_bymovie","Drama","age_of_movie","Comedy") ,
                    y = "rating" , 
                    training_frame = train3 ,
                    validation_frame = test3,
                    alpha = 0.5,   #0=ridge , 1 = lasso, so we leave it in the middle to use both penalizers
                    lambda = seq(0, 10, 0.25),
                    nlambdas = 30,
                    seed = 1,
                    keep_cross_validation_predictions = TRUE,
                    nfolds = 3)

#Algorithm:	Generalized Linear Modeling - MOdel ID: GLM_model_R_1555942382794_38
# If you want to use this model without retraining again, download it from github in your wd. Then start your
# H2o cluster (make sure to use same h2o version). And then type in r.  
# type your full path #h2o_glm = h2o.loadModel("C:/Users/npapaco/Documents/R_Projects/GLM_model_R_1555942382794_38")
# CAUTION YOU NEED TO DO FIRST THE STEPS ABOVE. Create the Splits 3, the train3, test3 and test_validation SETS

# output - Standardized Coefficient Magnitudes (standardized coefficient magnitudes)
# movieId
# userId
#age_of_movie


summary(h2o_glm)
#
h2o.performance(h2o_glm, test3) #performance on test set (split of training not validation set) #0.92
#
pred.ratings.h2o_glm <- h2o.predict(h2o_glm,as.h2o(test_validation))# predict ratings on validation set and evaluate RMSE
#
rmse_of_model_h2o_glm <- RMSE(pred.ratings.h2o_glm, as.h2o(test_validation$rating))# predict ratings on validation set and evaluate RMSE
#
rmse_of_model_h2o_glm # [1] 1.01631

#Save the RMSE of various models into one DF . So in the end we will have them all stored in one file
all_models_rmse_results <- data.frame(Algorithm=c("Matrix factorization with SGD","H2o Auto ML model","H2o GLM model"),RMSE = c(rmse_of_model_mf,rmse_of_model_h2oamlmodel2,rmse_of_model_h2o_glm))

#display the RMSE results with the help of kable package
kable(all_models_rmse_results) %>%
  kable_styling(bootstrap_options = "bordered" , full_width = T , position = "center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold = T ,color = "red", background = "lightgrey" )


invisible(gc())#free ram

#-------------------------H2o - Gradient Boosting Machine model ---------------------------------------------------------------------------------------------------------------------------------------------------

# Gradient Boosting Machine (for Regression and Classification) is a forward learning ensemble method. The guiding heuristic is that good predictive results can be obtained through
# increasingly refined approximations. H2O's GBM sequentially builds regression trees on all the features of the dataset in a fully distributed way - each tree is built in parallel.
# http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm.html
#third gbm model :  ntrees = 50, max depth = 5, learn rate = 0.1 , nfolds = 3 

#----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------

h2o_gbm_model <- h2o.gbm( x = c("movieId","userId","n.movies_byUser","n.users_bymovie","Drama","Comedy","age_of_movie") ,
                          y = "rating" , 
                          training_frame = train3 ,
                          ntrees = 50,
                          validation_frame = test3,
                          nfolds = 3,
                          seed=1,
                          keep_cross_validation_predictions = TRUE,
                          fold_assignment = "AUTO") 

# Algorithm :Gradient Boosting Machine  Model ID: GBM_model_R_1555942382794_37
# you can download the model from github in your wd and load it  with the below command
#If you place it in your working directory, #  And then you can type your full path # h2o_gbm_model = h2o.loadModel("C:/Users/npapaco/Documents/R_Projects/GBM_model_R_1555942382794_37")
# CAUTION YOU NEED TO DO FIRST THE PREPARATION OF DATA SETS - STEPS ABOVE. Create the Splits 3, the train3, test3 and test_validation SETS

# variable	relative_importance	scaled_importance	percentage
# age_of_movie	1759746.6250	1.0	0.3140
# n.users_bymovie	1684104.0	0.9570	0.3005
# movieId	883807.2500	0.5022	0.1577
# Drama	881101.2500	0.5007	0.1572
# n.movies_byUser	354473.7813	0.2014	0.0632
# userId	26781.1113

# Plot scoring history
plot(h2o_gbm_model)

# Plot the variables importance
h2o.varimp_plot(h2o_gbm_model, num_of_features = NULL)


# Print the scoring history
h2o.scoreHistory(h2o_gbm_model)


h2o.performance(h2o_gbm_model, test3) #performance on test set (split of training not validation set)
#[1] 1.01631
pred.ratings.h2o_gbm_model <- h2o.predict(h2o_gbm_model,as.h2o(test_validation))# predict ratings on validation set and evaluate RMSE
#
rmse_of_model_h2o_gbm_model <- RMSE(pred.ratings.h2o_gbm_model, as.h2o(test_validation$rating))# predict ratings on validation set and evaluate RMSE
#
rmse_of_model_h2o_gbm_model # [1] 1.035326

#Save the RMSE of various models into one DF . So in the end we will have them all stored in one file
all_models_rmse_results <- data.frame(Algorithm=c("Matrix factorization with SGD","H2o Auto ML model","H2o GLM model","H2o GBM model"),RMSE = c(rmse_of_model_mf,rmse_of_model_h2oamlmodel2,rmse_of_model_h2o_glm,rmse_of_model_h2o_gbm_model))

# Display results
kable(all_models_rmse_results) %>%
  kable_styling(bootstrap_options = "bordered" , full_width = T , position = "center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold = T ,color = "red", background = "lightgrey" )


#-------------- Distributed Random Forest ----------------------------------------------------------------------------------------------
#
# Distributed Random Forest (DRF) is a powerful classification and regression tool. When 
# given a set of data, DRF generates a forest of classification (or regression) trees, rather
# than a single classification (or regression) tree. Each of these trees is a weak learner built 
# on a subset of rows and columns
#--------------------------------------------------------------------------------------------------------------------------------------

h2orf1 <- h2o.randomForest( x = c("movieId","userId","n.movies_byUser","n.users_bymovie","Drama", ## h2o.randomForest function
                                  "Comedy","age_of_movie","year_rated") ,
                            y = "rating" ,         # Dependent var
                            training_frame = train3,        ## the H2O frame for training
                            validation_frame = test3, # the testing frame NOT the real validation!!
                            model_id = "h2orf1",    ## name the model ID so you can load it afterwards in R and in h2o
                            ntrees = 50,                  ## numb of  maximum trees to use.The default is 50
                            max_depth = 30,
                            keep_cross_validation_predictions= TRUE, # i recommend to use cross validation
                            min_rows = 100, # min rows during training
                            #  You can add the early stopping criteria decide when to stop fitting new trees when 
                            score_each_iteration = T,      ## Predict against training and validation for each tree. Default will skip several.
                            nfolds = 3,
                            fold_assignment = "AUTO",
                            seed = 1) 

# variable	relative_importance	scaled_importance	percentage
# n.users_bymovie	26496842.0	1.0	0.2790
# age_of_movie	22493868.0	0.8489	0.2369
# movieId	21997428.0	0.8302	0.2317
# Drama	9633707.0	0.3636	0.1015



summary(h2orf1)
#summary(h2orf1) RMSE:  h2orf_model 0.9425651

#Plot variables importance
h2o.varimp_plot(h2orf1, num_of_features = NULL)




# you can download the model from github in your wd and load it  with the below command
#If you place it in your working directory, you just need to type /ModelIDname  as in my below examples
# h2orf1 = h2o.loadModel("/h2orf1")

#We print the variable importance
h2o.varimp_plot(h2orf1, num_of_features = NULL)


# We print the scoring history
h2o.scoreHistory(h2o_glm)


# i evaluate performance on test set
h2o.performance(h2orf1, test3) #RMSE: (Extract with `h2o.rmse`) 0.7640699

#i predict ratings on validation set and evaluate RMSE
pred.ratings.h2orf1 <- h2o.predict(h2orf1,as.h2o(test_validation))
#
rmse_of_h2orf1_model <- RMSE(pred.ratings.h2orf1, as.h2o(test_validation$rating))
#
rmse_of_h2orf1_model 


#Save the RMSE of various models into one DF . So in the end we will have them all stored in one file
all_models_rmse_results <- data.frame(Algorithm=c("Matrix factorization with SGD","H2o Auto ML model","H2o GLM model","H2o GBM model","H2o RF model"),RMSE = c(rmse_of_model_mf,rmse_of_model_h2oamlmodel2,rmse_of_model_h2o_glm,rmse_of_model_h2o_gbm_model,rmse_of_h2orf1_model))

#display the RMSE results with the help of kable package
kable(all_models_rmse_results) %>%
  kable_styling(bootstrap_options = "bordered" , full_width = T , position = "center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold = T ,color = "red", background = "lightgrey" )



#------------------------ H2o Stacked Ensembles ---------------------------------------------------------------------------------------

# Ensemble machine learning methods use multiple learning algorithms to obtain better predictive performance than 
# could be obtained from any of the constituent learning algorithms. H2O's Stacked Ensemble method is supervised ensemble machine 
# learning algorithm that finds the optimal combination of a collection of prediction algorithms using a process called stacking. 
# This method currently supports regression and binary classification . Stacking, also called Super Learning or Stacked Regression, 
# is a class of algorithms that involves training a second-level "metalearner" to find the optimal combination of the base learners. 
# Unlike bagging and boosting, the goal in stacking is to ensemble strong, diverse sets of learners together.
# https://h2o-release.s3.amazonaws.com/h2o/rel-ueno/2/docs-website/h2o-docs/data-science/stacked-ensembles.html

#----------------------------------------------------------------------------------------------------------------------------------------

# We will stack the previous 3 models: Generalized Linear Models, Gradient Boosting Machine and the Distributed Random Forest (h2o_glm, h2o_gbm_model, h2orf_model)
h2oensemble2 <- h2o.stackedEnsemble(x = c("movieId","userId","n.movies_byUser","n.users_bymovie","Drama",
                                          "Comedy","age_of_movie") ,
                                    y = "rating" , 
                                    training_frame = train3,
                                    validation_frame = test3,
                                    model_id = "h2oensemble2",
                                    seed = 1,
                                    base_models = list(h2o_glm, h2o_gbm_model,h2orf1))# the previous models we trained

# Algorithm:	Stacked Ensemble
# Model ID:	h2oensemble2
# If you want to use this model without retraining again, download it from github in your wd. Then start your
# H2o cluster (make sure to use same h2o version). Then train the datasets!! And then type in r. h2oensemble2 = h2o.loadModel("C:/Users/npapaco/Documents/R_Projects/h2oensemble2")


h2oensemble2 # RMSE:  0.9194296
#
pred.ratings.h2oensemble2 <- h2o.predict(h2oensemble2,as.h2o(test_validation))
#
rmse_of_model_h2oensemble2 <- RMSE(pred.ratings.h2oensemble2, as.h2o(test_validation$rating))
#
rmse_of_model_h2oensemble2 #[1] 1.003683

#Save the RMSE of various models into one DF . So in the end we will have them all stored in one file
all_models_rmse_results <- data.frame(methods=c("Matrix factorization with SGD","H2o Auto ML model","H2o GLM model","H2o GBM model","H2o RF model","H2o Ensemble model"),rmse = c(rmse_of_model_mf,rmse_of_model_h2oamlmodel2,rmse_of_model_h2o_glm,rmse_of_model_h2o_gbm_model,rmse_of_h2orf1_model,rmse_of_model_h2oensemble2))

#display the RMSE results with the help of kable package
kable(all_models_rmse_results) %>%
  kable_styling(bootstrap_options = "bordered" , full_width = T , position = "center") %>%
  column_spec(1,bold = T ) %>%
  column_spec(2,bold = T ,color = "red", background = "lightgrey" )


all_models_rmse_results
# 1 Matrix factorization with SGD 0.7825707
# 2             H2o Auto ML model 1.0382518
# 3                 H2o GLM model 1.0163097
# 4                 H2o GBM model 1.0353257
# 5                  H2o RF model 1.0298481
# 6            H2o Ensemble model 1.0036833




# OUTPUT - VARIABLE IMPORTANCES
# variable	relative_importance	scaled_importance	percentage
# n.users_bymovie	7164049.0	1.0	0.3019
# Drama	5418105.0	0.7563	0.2283
# movieId	5250999.0	0.7330	0.2212
# age_of_movie	5224642.5000	0.7293	0.2201
# Comedy	675853.8125	0.0943	0.0285




#------------------------   Conclusions  --------------------------------------------------------------------------------
# 
# the best RMSE was achieved only with 2 Features user ID and Movie ID in Matrix factorization with SGD (RMSE 0.78).
# The h2o ensemble model (RMSE 1.003) stacked the below 3 models (GLM,GBM,DRF)
# And had by far the lowest RMSE of the H2o Models. With more hyper parameters tuning even lower RMSE can be achieved. But not so low like with MF models.
# I wouldn't recommend the auto ML model seems you can't tune the hyper parameters of the models, that results not into the lowest RMSE or higher accuracy on classification models.
# I also trained the same models with scaled values but the RMSE was in all models higher. It seems that the most important features where
# (Number of users rated the movie, age of movie = more ratings and higher mean rating, the movie id, and Drama = genre with the most ratings)
# The other features didn't had low p value and didn't improve the model but the overfired it. 
# We could with web scrab add more features like budget of movie, critics rating and duration of movie and compare the RMSE's

# In H2o Auto ML model (RMSE 1.03) the most important features where the below
# OUTPUT - VARIABLE IMPORTANCES
# variable	relative_importance	scaled_importance	percentage
# n.users_bymovie	7164049.0	1.0	0.3019
# Drama	5418105.0	0.7563	0.2283
# movieId	5250999.0	0.7330	0.2212


# The GLM Model (RMSE 1.01 ) has similar evaluation process with Matrix factorization with SGD. Thats why put more weight on the below features
# output - Standardized Coefficient Magnitudes (standardized coefficient magnitudes)
# movieId
# userId
# age_of_movie
# But the RMSE 1.01 was not so low like MF model.

# The Gradient Boosting Machine  Model (RMSE 1.03)  put more weight on the below features
# variable	relative_importance	scaled_importance	percentage
# age_of_movie	1759746.6250	1.0	0.3140
# n.users_bymovie	1684104.0	0.9570	0.3005
# movieId	883807.2500	0.5022	0.1577

# The distributed Random Forest Model (RMSE 1.02)  put more weight on the below features

# variable	relative_importance	scaled_importance	percentage
# n.users_bymovie	26496842.0	1.0	0.2790
# age_of_movie	22493868.0	0.8489	0.2369
# movieId	21997428.0	0.8302	0.2317
# Drama	9633707.0	0.3636	0.1015

# The h2o ensemble model (RMSE 1.003) stacked the below 3 models (GLM,GBM,DRF)
# And had by far the lowest RMSE of the H2o Models. With more hyper parameters tuning even lower RMSE can be achieved. But not so low like with MF models.
# I wouldn't recommend the auto ML model seems you can't tune the hyper parameters of the models, that results not into the lowest RMSE or higher accuracy on classification models.

#--------------------------------------------------------------------------------------------------------------------------------------------


