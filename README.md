# Approach for Will-Bill-Solve-It, India_Hacks-Machine Learning
Solution which placed 15th on LeaderBoard.

https://www.hackerearth.com/machine-learning-india-hacks-2016/machine-learning/will-bill-solve-it/

## Data Sets:
Both training and testing dataset consist of 3 files :-

### 1) User File:  
With Attributes of a User:<br />
<br />
user_id - the user id <br />
skills - all his skills separated by the delimiter '|' <br />
solved_count - number of problems solved by the user <br />
attempts - total number of incorrect submissions done by the user <br />
user_type : type of user (S - Student, W - Working, NA - No Information Available)<br />
  
### 2) Problem File:
Attribute related to a Problem : <br />
<br />
problem_id - the id of the problem<br />
level - difficulty of the problem (Very-Easy, Easy, Easy-Medium, Medium, Medium-Hard, Hard)<br />
accuracy - the accuracy score for the problem<br />
solved_count - number of people who have solved it<br />
error_count - number of people who have solved it incorrectly<br />
rating - star (quality) rating of the problem on scale of 0-5<br />
tag1 - tag of the problem representing the type e.g. Data Structures<br />
tag2 - tag of the problem<br />
tag3 - tag of the problem<br />
tag4 - tag of the problem<br />
tag5 - tag of the problem<br />
  
### 3) Submissions File:
Problem User interaction and final results for each attempt a user made to a solve a particular problem.<br />
<br />
user_id - the id of the user who made a submission<br />
problem_id - the id of the problem that was attempted<br />
solved_status - indicates whether the submission was correct (SO : Solved or Correct solution, AT : Attempted or Incorrect solution )<br />
result - result of the code execution (PAC: Partially Accepted, AC : Accepted, TLE : Time limit exceeded, CE : Compilation Error, RE : Runtime Error, WA : Wrong Answer)<br />
language_used - the lang used by user to code the solution <br />
execution_time - the execution time of the solution<br />

## Approach

count - A custom feature.
Weighs are assigned to user's skills and based on the skills that a user has, the value is calculated.

Used Ensemble of RandomForestClassifier, ExtraTreesClassifier and GradientBoostingClassifier.



### Installations Required#####

python
sklearn >=0.16
numpy
pandas


## Instructions to run the code ###

python model.py


