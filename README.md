# Ray-Tracer
Ray tracer created with C++ and SDL2


![Uploading image.png…]()



This ray tracer uses the path tracing technique to send many rays for each pixel and taking the average to give photo-realistic results. It has a sphere class and three material classes - metallic, diffused and dielectric. Further shapes and materials are easy to add due to the object oriented nature of this project.

It also has a camera which can be moved by :
(i)WASD to move in camera's horizontal plane
(ii)Spacebar and Left-Ctrl to move in the y axis
(iii)NUMPAD 8 and 2 to rotate vertically
(iv)NUMPAD 4 and 6 to rotate horizontally

There are two sampling functions:
(i) Sampling() - does sampling altogether and the puts buffer to screen and is useful for animations
(ii) progSampling() - can be used to move about the world with camera since it does sampling in steps and thus giving better frame rate 

I created this ray tracer with knowledge from the following resources:
(i)https://www.scratchapixel.com/ 
(ii)Ray tracing in a weekend by Peter Shirley
(iii)https://blog.scottlogic.com/
