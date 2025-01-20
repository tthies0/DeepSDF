# Task 1:
Explore various ways of improving the generalization ability across different categories, e.g., adding class embedding, class-label text conditioning, etc. 
During inference, provided with similar partial observations from different categories, the model should be able to generate shapes that align well with observations while keeping class-specific properties, and the shape interpolation should show gradual changes of shapes in-between categories.

## Idea 1:
* Training, one dataset, inference on multiple ones
* Training, multiple dataset, inference on multiple ones
* Training, multiple dataset, inference on multiple ones, add class-id to (x,y,z) input -> class-ids in array -> array position (0;1)
* 