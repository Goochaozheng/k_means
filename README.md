# K-means
2031561 Chaozheng Guo  
Python implementation of k-means.  

### Packages reqired
```numpy``` ```argparse```

### Usage
```python kmeans.py```  
By default, this will load iris dataset and perform k-means clustering. Clustering result will be displayed.

Some options can be specified for the program:  

```-dataset``` str, Dataset to be used, ```breast_cancer``` or ```iris```. Default: ```iris```  
```-k``` int, Number of clusters. Default: 2     
```-max_it``` int, Maximum iteration time. Default: 20

Example:
```python kmeans.py -dataset=breast_cancer -k=3 -max_it=10```  





