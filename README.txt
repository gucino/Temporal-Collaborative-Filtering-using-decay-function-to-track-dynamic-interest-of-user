#purpose
This code aim to demonstrate the importance of incorperating
time information into recommendation system by simple 
temporal collaborative filtering using time decay function.
This is because in real world the preference of user change over time.

#detail of dataset
The Movielens dataset contains 10000 one-to-five-scale rating along with 
timestamp for 1682 movies and 943 users. 
Each movie is associated with genre where one movie can have more than one genre.
Each user comes with ages, sex, and occupation.
Acknoledgement 
F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: 
History and Context. ACM Transactions on Interactive Intelligent 
Systems (TiiS) 5, 4, Article 19 (December 2015), 19 pages. 
DOI=http://dx.doi.org/10.1145/2827872 

#hyper-parameter selection
alpha : determine how much we want the effect of time to be
large alpha : user preference change alot with time
small alpha : user preference change very little with time
alpha=0 : user preference remain the same over time
The optimal value of alpha is 1.7 as shown in "choosing_alpha.png"

#result 
As shown in "compare_btw_time_and_no_time.png", by taking into
account the time information, this has significantly
improve the performance of recommendation system in terms of mean absolute error.

