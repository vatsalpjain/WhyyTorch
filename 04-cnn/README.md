### CNN

1. What problem does this solve?

-> In CNN we use filters than learn shapes cause they can correlate close by pixels and Also The Parameter count is significantly less , Cause Shapes are used over full images
Fully Connected : 28 x 28 image = 784 * 128 neurons = ~100,480 parameters
CNN             :  32 filters of 3 x 3 + =  320 parameters

2. What are the trade-offs?

-> Reusing the same filter , like if a ball is in one corner and one more ball is in other corner the filter will learn ball and identify both with it's own weights , no need to take every pixels weights

![1785245842181](image/README/1785245842181.png)


**Mental model:** forward = one weight, many uses. backward = collect blame from every use, then update once.
