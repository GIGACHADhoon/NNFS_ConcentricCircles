import numpy as np
import math

class CCData:

    def __init__(self,no_circles,category_size,data_per_category):

        self.circlesX = np.empty((0,2))
        # One extra Category for outside of concentric Circles
        self.labels = np.concatenate([np.full(data_per_category, i) for i in range(no_circles+1)])
        self.max_size = no_circles*category_size
        self.no_circles = no_circles
        self.category_size = category_size
        self.data_per_category = data_per_category

    def generate(self):
        # Generate data for points inside concentric circles
        for low, high in self.get_boundaries():
            radius = np.random.uniform(low, high, size=self.data_per_category)
            theta = np.random.uniform(0, 2 * np.pi, size=self.data_per_category)
            x = radius * np.cos(theta)
            y = radius * np.sin(theta)
            self.circlesX = np.vstack([self.circlesX, np.column_stack((x, y))])  # Stack the (x, y) coordinates
        
        #data points for the outside of concentric circles.
        outside_circle = []
        while len(outside_circle) < self.data_per_category:
            x = np.random.uniform(-self.max_size, self.max_size)
            y = np.random.uniform(-self.max_size, self.max_size)
            if math.sqrt(x**2+y**2)>self.max_size:
                outside_circle.append([x,y]) 

        # Combine the data from inside and outside the circles
        self.circlesX = np.vstack([self.circlesX, np.array(outside_circle)])
        return self.circlesX, self.labels
    
    def get_boundaries(self):
        return [(i*self.category_size,i*self.category_size+self.category_size) for i in range(self.no_circles)]