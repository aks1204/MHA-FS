
# Arrhythmia Feature Selection using Meta-Heuristic Algorithms

Arrhythmia refers to an irregular heartbeat or an abnormal heart rhythm. The heart normally beats in a regular pattern, but in individuals with arrhythmia, the heart may beat too quickly, too slowly, or with an irregular pattern. Some arrhythmias may not cause any noticeable symptoms and may be harmless, while others can be more severe and potentially life-threatening.

There are several parameters/features that we might consider while classifying Arrhythmia into different types.

However, considering all features for Arrhythmia classification is computationally expensive while dealing with large datasets.

We have tried to optimize the problem of Arrhythmia classification by deciding upon which all features to select using Meta-Heuristic Algorithms.
We have used Random Forest Classfier to report for the accuracy of selected features in the Arrhythmia Classification Task.

To learn more about Random Forest refer to the link given below.

[Link](https://www.ibm.com/topics/random-forest)

## Dataset

Our dataset consists of 452 samples with 279 features and each of the sample has been assigned any one of the 16 classes shown below.

Table below shows the class with its corresponding labels.

| Class label | Class                                      |
|------------|--------------------------------------------|
| 1         | Normal                                     |
| 2         | Ischemic changes (Coronary Artery Disease) |
| 3         | Old Anterior Myocardial Infarction         |
| 4         | Old Inferior Myocardial Infarction         |
| 5         | Sinus tachycardy                           |
| 6         | Sinus bradycardy                           |
| 7         | Ventricular Premature Contraction (PVC)    |
| 8         | Supraventricular Premature Contraction     |
| 9         | Left bundle branch block                    |
| 10         | Right bundle branch block                   |
| 11         | 1. degree AtrioVentricular block            |
| 12         | 2. degree AV block                         |
| 13         | 3. degree AV block                         |
| 14         | Left ventricule hypertrophy                 |
| 15         | Atrial Fibrillation or Flutter              |
| 16         | Others                                     |


We have split the samples in our dataset to 339 for training and 113 for testing.

## Dependencies 

First and foremost you need to install Python.

1. Mealpy (https://github.com/thieu1995/mealpy)
2. Permetrics (https://github.com/thieu1995/permetrics)
3. Scikit-learn (https://scikit-learn.org/stable/index.html)
4. Pandas (https://pandas.pydata.org/)
5. Matplotlib (https://matplotlib.org/)

## Setup environment

### Pip 

```code 
pip install -r requirements.txt
```

## How to run

```code
python -m src.models.mha_fs
```

##

We need to select the best features in dataset.

Our solution is a 1-D vector, each dimension representing an index of column in the dataset.
- If it has value 1, meaning this column is selected for the model.
- If it has value 0, meaning this column is not selected for the model.

So for each of the dimension we need to convert real value back to either 0 or 1.
- The lower bound is 0 for all dimensions(floor of 0 is 0).
- The upper bound is 1.99 for all dimensions(floor of 1.99 is 1).

Also if no column is selected we randomly choose any one column.

We have set our population size to 50 and have run our algorithms for 100 epochs.

### Fitness Function

$$ \text{Fitness} = \text{Accuracy} $$

$$ \text{Accuracy} = {\text{Number  of Correctly Classified Instances} \over \text{Total Number of Instances}} $$

## Differential Evolution

To learn more about Differential Evolution refer to the link given below.

[Link](https://doi.org/10.1016/j.swevo.2018.10.006)

We have used DE/rand/2/bin strategy for our Differenetial Evolution algorithm.

Our weighing factor is 0.8 and crossover rate/probability is 0.9.

The graphs below depict global best fitness value and local best fitness value for Differential Evolution algorithm as a function of number of epochs/iterations.

![lbfc](https://github.com/aks1204/MHA-FS/assets/57048028/8e2cf1e8-faaf-4296-a44d-0986d6edfdbe)

![gbfc](https://github.com/aks1204/MHA-FS/assets/57048028/13305b97-cbe6-4e22-8988-d8bba2324980)

The above graphs would be plotted and saved in graphs/DE folder after our code has been run for Differential Evolution algorithm for 100 epochs.

## Genetic Algorithm

To learn more about Genetic Algorithm refer to the link given below.

[Link](https://www.analyticsvidhya.com/blog/2017/07/introduction-to-genetic-algorithm/)

We have chosen tournament selection method for selection. In tournament selection we have run tournament amongst 10 inviduals in our population. 

We have chosen uniform crossover method with 0.95 as our crossover probability.

We have chosen multiple points for mutation with 0.025 as our mutation probability. We have chosen flip method for mutation wherein we have replaced 1 with 0 or vice versa for a particular dimension in our dataset.

The graphs below depict global best fitness value and local best fitness value for Genetic Algorithm as a function of number of epochs/iterations.

![lbfc (1)](https://github.com/aks1204/MHA-FS/assets/57048028/f4b5c3c3-7d16-4549-aa71-7be1f539e1aa)

![gbfc (1)](https://github.com/aks1204/MHA-FS/assets/57048028/adf50ae5-d10e-48f4-afb9-73084377e17c)

The above graphs would be plotted and saved in graphs/GA folder after our code has been run for Genetic Algorithm for 100 epochs.

## Artificial Bee Colony

To learn more about Artificial Bee Colony refer to the link given below.

[Link](https://www.sciencedirect.com/topics/computer-science/artificial-bee-colony)

In Artificial Bee Colony we have set the limit for Number of trials before abandoning a food source(solution) as 5.

The graphs below depict global best fitness value and local best fitness value for Artificial Bee Colony algorithm as a function of number of epochs/iterations.

![lbfc (2)](https://github.com/aks1204/MHA-FS/assets/57048028/aa5c7ca9-9583-4e1f-a237-c9c975cd4916)

![gbfc (2)](https://github.com/aks1204/MHA-FS/assets/57048028/b763e69b-de19-41e6-8cd9-46d2ddbe358b)

The above graphs would be plotted and saved in graphs/ABC folder after our code has been run for Artificial Bee Colony algorithm for 100 epochs.

## Particle Swarm Optimization

### References

J. Kennedy and R. Eberhart, "Particle swarm optimization," Proceedings of ICNN'95 - International Conference on Neural Networks, Perth, WA, Australia, 1995, pp. 1942-1948 vol.4, doi: 10.1109/ICNN.1995.488968.

###

The graphs below depict global best fitness value and local best fitness value for Particle Swarm Optimization algorithm as a function of number of epochs/iterations.

![lbfc (3)](https://github.com/aks1204/MHA-FS/assets/57048028/327464e7-7838-475b-bd03-fb600e0500f3)

![gbfc (3)](https://github.com/aks1204/MHA-FS/assets/57048028/6b0a960c-bb7e-4f3e-ad92-3ab14699f5d1)

The above graphs would be plotted and saved in graphs/PSO folder after our code has been run for Particle Swarm Optimization algorithm for 100 epochs.
