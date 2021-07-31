# Ray-Tracer
Ray tracer created with C++ and SDL2


![image](https://user-images.githubusercontent.com/77842809/127741005-8fbba984-0270-41a2-8e46-67f5f6bf4fa1.png)


<div>
This ray tracer uses the path tracing technique to send many rays for each pixel and taking the average to give photo-realistic results. It has a sphere class and three material classes - metallic, diffused and dielectric. Further shapes and materials are easy to add due to the object oriented nature of this project.
</div>

<div>
It also has a camera which can be moved by :
<ul>
  <li>WASD to move in camera's horizontal plane</li>
  <li>Spacebar and Left-Ctrl to move in the y axis</li>
  <li>NUMPAD 8 and 2 to rotate vertically</li>
  <li>NUMPAD 4 and 6 to rotate horizontally</li>
</div>
  
<div>  
There are two sampling functions:
<ol>
<li>Sampling() - does sampling altogether and the puts buffer to screen and is useful for animations</li>
<li>progSampling() - can be used to move about the world with camera since it does sampling in steps and thus giving better frame rate</li>
</ol>
</div>
  
<div>
I created this ray tracer with knowledge from the following resources:
<ul>  
  <li>https://www.scratchapixel.com/</li> 
  <li>Ray tracing in a weekend by Peter Shirley</li>
  <li>https://blog.scottlogic.com/</li>
</ul>
</div>
