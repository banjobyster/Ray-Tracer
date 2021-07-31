/*******************************************************************************************************************************************************************/
/* # Ray-Tracer
Ray tracer created with C++ and SDL2

This ray tracer uses the path tracing technique to send many rays for each pixel and taking the average to give photo-realistic results. It has a sphere class and three material classes - metallic, diffused and dielectric. Further shapes and materials are easy to add due to the object oriented nature of this project.

It also has a camera which can be moved by :
WASD to move in camera's horizontal plane
Spacebar and Left-Ctrl to move in the y axis
NUMPAD 8 and 2 to rotate vertically
NUMPAD 4 and 6 to rotate horizontally

There are two sampling functions:
(i) Sampling() - does sampling altogether and the puts buffer to screen and is useful for animations
(ii) progSampling() - can be used to move about the world with camera since it does sampling in steps and thus giving better frame rate 

I created this ray tracer with knowledge from the following resources:
->https://www.scratchapixel.com/ 
->Ray tracing in a weekend by Peter Shirley
->https://blog.scottlogic.com/
*/
/*******************************************************************************************************************************************************************/

#include <cstdlib>
#include <cstdio>
#include <cmath>
#include <fstream>
#include <vector>
#include <iostream>
#include <cassert>
#include<float.h>
#include <SDL.h>
#include<time.h>
#include<map>
using namespace std;

#define M_PI 3.141592653589793
#define INFINITY 1e8

const int width = 600;
const int height = 400;

//class declarations

//standard vector 3D class template
template<typename T>
class Vec3
{
public:
    T x, y, z;
    Vec3() : x(T(0)), y(T(0)), z(T(0)) {}
    Vec3(T xx) : x(xx), y(xx), z(xx) {}
    Vec3(T xx, T yy, T zz) : x(xx), y(yy), z(zz) {}

    Vec3& normalize()
    {
        //nor2 is normal squared, similarly length2 is length squared
        T nor2 = length2();
        if (nor2 > 0)
        {
            T invNor = 1 / sqrt(nor2);
            x *= invNor, y *= invNor, z *= invNor;
        }
        return *this;
    }
    Vec3<T> operator * (const T& f) const
    {
        return Vec3<T>(x * f, y * f, z * f);
    }
    Vec3<T> operator / (const T& f) const
    {
        return Vec3<T>(x / f, y / f, z / f);
    }
    Vec3<T> operator * (const Vec3<T>& v) const
    {
        return Vec3<T>(x * v.x, y * v.y, z * v.z);
    }
    T dot(const Vec3<T>& v) const
    {
        return x * v.x + y * v.y + z * v.z;
    }
    Vec3<T> cross(const Vec3<T>& v) const
    {
        return Vec3<T>((y * v.z - z * v.y), -(x * v.z - z * v.x), (x * v.y - y * v.x));
    }
    Vec3<T> operator - (const Vec3<T>& v) const
    {
        return Vec3<T>(x - v.x, y - v.y, z - v.z);
    }
    Vec3<T> operator + (const Vec3<T>& v) const
    {
        return Vec3<T>(x + v.x, y + v.y, z + v.z);
    }
    Vec3<T>& operator += (const Vec3<T>& v)
    {
        x += v.x, y += v.y, z += v.z;
        return *this;
    }
    Vec3<T>& operator -= (const Vec3<T>& v)
    {
        x -= v.x, y -= v.y, z -= v.z;
        return *this;
    }
    Vec3<T>& operator *= (const Vec3<T>& v)
    {
        x *= v.x, y *= v.y, z *= v.z;
        return *this;
    }
    Vec3<T>& operator /= (const Vec3<T>& v)
    {
        x /= v.x, y /= v.y, z /= v.z;
        return *this;
    }
    Vec3<T> operator - () const
    {
        return Vec3<T>(-x, -y, -z);
    }
    T length2() const
    {
        return x * x + y * y + z * z;
    }
    T length() const
    {
        return sqrt(length2());
    }
};


typedef Vec3<float> Vec3f;

vector<vector<Vec3f>> buffer(height, vector<Vec3f>(width, Vec3f(0, 0, 0))); // current frame is hold in this

void clrBuffer() {
    for (int i = 0; i < height; i++) {
        for (int j = 0; j < width; j++) {
            buffer[i][j] = Vec3f(0, 0, 0);
        }
    }
}

class ray {
    //ray of form a+kb where a is origin, b is direction vector and k is parameter
public:
    Vec3f a;
    Vec3f b;
    ray() {}
    ray(const Vec3f& a, const Vec3f& b) {
        this->a = a;
        this->b = b;
    }
    Vec3f origin() const {
        return a;
    }
    Vec3f direction() const {
        return b;
    }
    Vec3f point_at_parameter(float k) const {
        return a + b * k;
    }
};

class camera {
private:
    float theta;
    float half_height;
    float half_width;
    Vec3f vup = Vec3f(0, 1, 0);
    Vec3f u, v;
public:
    Vec3f origin;
    Vec3f direction;
    Vec3f corner;//lower left corner
    Vec3f horizontal;
    Vec3f vertical;
    Vec3f lookat;
    //vfov is vertical field of view (in degrees)
    //taking canvas size to be of 2 high and 4 wide at a distance of 1 in the negative x axis with origin at 0
    //lookfrom is the origin of the cam, lookat is the point the camera is looking at, vup is the vertical axis of the camera plane
    camera(Vec3f lookfrom, Vec3f dir, Vec3f vup, float vfov, float aspect) {
        //Vec3f w; //u is the horizontal direction, v is vertical direction, w is the front direction (since we looking at -z, we are opposite to w)
        theta = vfov * M_PI / 180;
        half_height = tan(theta / 2); // since canvas is at a distance of 1
        half_width = aspect * half_height;
        origin = lookfrom;
        direction = dir.normalize();
        u = vup.cross(direction).normalize();
        v = u.cross(direction);
        corner = origin - u * half_width - v * half_height + direction;
        horizontal = u * 2 * half_width;
        vertical = v * 2 * half_height;
    }
    ray get_ray(float s, float t) {
        return ray(origin, corner + horizontal * s + vertical * t - origin);
    }

    //for movement of cam
    void updCam() {
        u = vup.cross(direction).normalize();
        v = u.cross(direction);
        corner = origin - u * half_width - v * half_height + direction;
        horizontal = u * 2 * half_width;
        vertical = v * 2 * half_height;
    }
    void MovFront(bool i) {
        if (i) {
            origin += Vec3f(direction.x,0,direction.z).normalize() * 0.04;
        }
        else {
            origin -= Vec3f(direction.x, 0, direction.z).normalize() * 0.04;
        }
        updCam();
    }
    void MovRight(bool i) {
        if (i) {
            origin += Vec3f(u.x, 0, u.z).normalize() * 0.04;
        }
        else {
            origin -= Vec3f(u.x, 0, u.z).normalize() * 0.04;
        }
        updCam();
    }
    void MovUp(bool i) {
        if (i) {
            origin.y += 0.04;
        }
        else {
            origin.y -= 0.04;
        }
        updCam();
    }
    void rotVert(bool i) {
        //cheaper than doing vector rotation //but has limitation as well can only look between ~1.3 rad to ~-1.3 radian
        if (i) {
            direction.y += 0.01;
        }
        else
        {
            direction.y -= 0.01;
        }
        direction.normalize();
        updCam();
    }
    void rotHor(bool i) {
        float ang = 0.0172665;
        float cosA, sinA;
        if (i) {
            cosA = cos(ang);
            sinA = sin(ang);
        }
        else {
            cosA = cos(-1.0 * ang);
            sinA = sin(-1.0 * ang);

        }
        direction.x = direction.x * cosA + direction.z * sinA;
        direction.z = -1.0 * direction.x * sinA + direction.z * cosA;
        direction.normalize();
        updCam();
    }
};

class material;//for hit_record 

/*
* //In the words of Peter Shirley (whose book I am following cause my homemade ray tracer was just a reflecting piece of ****)
When a ray hits a surface (a particular sphere for example), the material pointer in the
hit_record will be set to point at the material pointer the sphere was given when it was set up in
main() when we start. When the color() routine gets the hit_record it can call member
functions of the material pointer to find out what ray, if any, is scattered.
*/

/*
The following two classes will hold the record of hit and if hitable or not, here, t is the parameter of the eqn of line which we have discussed later on
t_min and t_max are just for the near and far plane of the camera
*/
class hit_record {
public:
    float t;//the parameter
    Vec3f phit;//point of hit
    Vec3f nhit;//normal at point of hit
    material* mat_ptr;
};

class hitable {
public:
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& record) const = 0;
};

//holds list of objects which are hitable
class hitable_list : public hitable {
public:
    hitable** list;
    int list_size;
    hitable_list() {}
    hitable_list(hitable** l, int n) { list = l; list_size = n; }
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& record) const;
};

//this function helps if ray hits an hitable object and if that object is closer than all
bool hitable_list::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
    hit_record temp;
    bool hit_anything = false;
    double closest_so_far = t_max;
    for (int i = 0; i < list_size; i++) {
        //this will call the hit function of the objects's class type
        if (list[i]->hit(r, t_min, closest_so_far, temp)) {
            hit_anything = true;
            closest_so_far = temp.t;
            record = temp;
        }
    }
    return hit_anything;
}

//in diffused material, when ray hits the surface, it bounces of in any random direction, might also get absorbed depending on color of surface
//here, this function helps to find random direction quickly
Vec3f randomPointUnitSphere() {
    Vec3f p = Vec3f(1,1,1);
    do{
        p = Vec3f((float(rand()) / (RAND_MAX + 1.0)), (float(rand()) / (RAND_MAX + 1.0)), (float(rand()) / (RAND_MAX + 1.0)));
    } while (p.length2() >= 1);
    return p;
}

//material base class // attenuation is the loss of light intensity over the distance
class material {
public:
    virtual bool scatter(const ray& r, const hit_record& rec, Vec3f& attenuation, ray& scattered) const = 0;
};

//Material lambertian(diffuse) //albedo indicated the characteristic color of an object
class Lambertian : public material {
public:
    Vec3f albedo;
    Lambertian(const Vec3f& a) : albedo(a) {};
    virtual bool scatter(const ray& r, const hit_record& rec, Vec3f& attenuation, ray& scattered) const {
        //rec.phit+rec.nhit gives point at normal at a unit distance from surface, and then random point in unit sphere will give that target in the surface of a unit sphere 
        //which is on the phit
        Vec3f target = rec.phit + rec.nhit + randomPointUnitSphere();
        scattered = ray(rec.phit, target - rec.phit);
        attenuation = albedo;
        return true;
    }
};

//reflection ray
Vec3f reflect(const Vec3f& incident, const Vec3f& normal) {
    return (incident -  normal * 2.0 * normal.dot(incident)).normalize(); // n is already a unit vector(nhit)
}

class metal : public material {
public:
    Vec3f albedo;
    float fuzz;
    metal(const Vec3f& a, const float& f) : albedo(a) {
        if (f < 1) {
            fuzz = f;
        }
        else {
            fuzz = 1;
        }
    };
    virtual bool scatter(const ray& r, const hit_record& rec, Vec3f& attenuation, ray& scattered) const {
        //rec.phit+rec.nhit gives point at normal at a unit distance from surface, and then random point in unit sphere will give that target in the surface of a unit sphere 
        //which is on the phit
        Vec3f reflected = reflect(r.direction(), rec.nhit);
        //makes the scattered ray from metal not perfectly straight causing fuzziness on the surface
        scattered = ray(rec.phit, reflected + randomPointUnitSphere() * fuzz);
        attenuation = albedo;
        return (rec.nhit.dot(scattered.direction())>0);
    }
};

//this is a simple polynomial approximation by Christophe Schlick for the reflection at dielectric's surface that varies with angle
float schlick(float cosine, float ref_ind) {
    float r0 = (1 - ref_ind) / (1 + ref_ind);
    r0 = r0 * r0;
    return r0 * (1 - r0) * pow((1 - cosine), 5);
}

//refraction, using snells law, and fact of total internal reflection to check for refraction
bool refract(const Vec3f& incident, const Vec3f& normal, const float& ni_over_nt, Vec3f& refracted) {
    Vec3f unit_incident = incident;
    unit_incident.normalize();
    float dt = unit_incident.dot(normal);
    float discriminant = 1.0 - ni_over_nt * ni_over_nt * (1 - dt * dt);
    if (discriminant > 0) {
        refracted = (unit_incident - normal * dt) * ni_over_nt - normal * sqrt(discriminant);
        return true;
    }
    else {
        return false;
    }
}

class dielectric : public material {
public:
    float ref_ind;
    Vec3f color;
    dielectric(const float& ref) : ref_ind(ref), color(Vec3f(1.0, 1.0, 1.0)) {};
    dielectric(const float& ref,const Vec3f& col) : ref_ind(ref), color(col) {};
    virtual bool scatter(const ray& r, const hit_record& rec, Vec3f& attenuation, ray& scattered) const {
        Vec3f outward_normal;
        Vec3f reflected = reflect(r.direction(), rec.nhit);
        float ni_over_nt;
        attenuation = color;
        Vec3f refracted;
        float reflect_prob;
        float cosine;
        //first checks if outward normal should be same direction or not depending on direction of hit and changes ref_ind acc
        if (r.direction().dot(rec.nhit) > 0) {
            outward_normal = -rec.nhit;
            ni_over_nt = ref_ind;
            cosine = ref_ind * r.direction().dot(rec.nhit) / r.direction().length();
        }
        else {
            outward_normal = rec.nhit;
            ni_over_nt = 1.0 / ref_ind;
            cosine = -1.0 * r.direction().dot(rec.nhit) / r.direction().length();
        }
        //now for refraction
        if (refract(r.direction(), outward_normal, ni_over_nt, refracted)) {
            reflect_prob = schlick(cosine, ref_ind);
        }
        else {
            reflect_prob = 1.0;
        }
        if ((rand() / (RAND_MAX + 1.0)) < reflect_prob) {
            scattered = ray(rec.phit, reflected);
        }
        else {
            scattered = ray(rec.phit, refracted);
        }
        return true;
    }
};

class sphere : public hitable {
    //sphere eqn (x-cx)^2 + (y-cy)^2 + (z-cz)^2 = r*r
    //Let p(x,y,z) be point on sphere and C(cx,cy,cz) be center of sphere
    //then dot product of (p-C) will give the the lhs of eqn of sphere // dot product of vector with itself gives magnitude of the vector
    //replacing p with eqn of point on a line, we get p = a + b * t where t is the parameter to be found
    //we get eqn of sphere as a quadratic eqn dot(((a + b * t) - C), ((a + b * t) - C)) = r * r;
    //t*t*dot(b,b) + 2*t*dot(b,a-c) + dot(a-c,a-c) - r*r = 0;
    //if D> 0,  ray intersects at two points
    //if D= 0, ray is tangent to the sphere
    //if D<0, ray doesnt hit the sphere
public:
    Vec3f center;
    float radius;
    material* mat;
    sphere() {}
    sphere(Vec3f cen, float r, material* m) : center(cen), radius(r), mat(m) {};
    virtual bool hit(const ray& r, float t_min, float t_max, hit_record& record) const;
};

bool sphere::hit(const ray& r, float t_min, float t_max, hit_record& record) const {
    Vec3f oc = r.origin() - center;
    float a = r.direction().dot(r.direction());
    float b = 2.0 * r.direction().dot(oc);
    float c = oc.dot(oc) - radius * radius;
    float discriminant = b * b - 4 * a * c;
    if (discriminant >= 0) {
        float sqrD = sqrt(discriminant);
        float temp = (-b - sqrD) / (2 * a);
        if (t_min < temp && temp < t_max) {
            record.t = temp;
            record.phit = r.point_at_parameter(temp);
            record.nhit = Vec3f(record.phit - center) / radius;
            record.mat_ptr = mat;
            return true;
        }
        temp = (-b + sqrD) / (2 * a);
        if (t_min < temp && temp < t_max) {
            record.t = temp;
            record.phit = r.point_at_parameter(temp);
            record.nhit = Vec3f(record.phit - center) / radius;
            record.mat_ptr = mat;
            return true;
        }
    }
    return false;
}

//depth is the number of times it will scatter
Vec3f color(const ray& r, hitable* world,int depth) {
    hit_record rec;
    if (world->hit(r, 0.001, FLT_MAX, rec)) {
        ray scattered;
        Vec3f attenuation;
        if (depth < 50 && rec.mat_ptr->scatter(r, rec, attenuation, scattered)) {
            //bounces ray of until depth achieved
            return attenuation*color(scattered, world, depth + 1);
        }
        else {
            return Vec3f(0, 0, 0);
        }
    }
    else {
        //direction unit vector of ray
        Vec3f direction = r.direction().normalize();
        //t gives the upness or downess in the y direction of the ray // 0<=t<=1
        float t = 0.5 * (direction.y + 1.0);
        //lerping from white as start value to light blue as end value
        Vec3f col = Vec3f(1.0) * (1 - t) + Vec3f(0.5, 0.7, 1.0) * t;
        return col;
    }
}

inline bool input();
void bufferTOscreen(const int& k=1); // draws buffer to the sdl screen
void Sampling(); //sends large number of rays for the same image // useful to create animation
void progSampling(); //samples picture and progressively gets uploaded on the screem // useful when you want to explore // does no take anymore sample once sampling done until input given
bool init();
void kill();
bool loop();

// Pointers to our window and renderer
SDL_Window* window;
SDL_Renderer* renderer;

bool EXIT = false;
camera cam(Vec3f(-2, 1, -2), Vec3f(1, 0.1, 1), Vec3f(0, 1, 0), 60, float(width) / height);
hitable* world;
const int sampling_number = 32; // number of samples taken per image
const int quick_sampling_number = 20;
const int progSamplingNumber = 10;  // does 1/progSamplingNumber samplings per sample;
bool samplingNotDone = true;

int main(int argc, char** args)
{
    srand(time(0));
    if (!init()) return 1;

    //Add objects in the hitable list, n is number of objects
    const int n = 27;
    hitable* list[n + 1];
    // creating spheres and storing them in list
    for (int i = 0; i < 3; i++) {
        for (int j = 0; j < 3; j++) {
            for (int k = 0; k < 3; k++) {
                list[i * 9 + j * 3 + k] = new sphere(Vec3f(i, j, k), 0.4, new metal(Vec3f((rand() / (RAND_MAX + 1.0)), (rand() / (RAND_MAX + 1.0)), (rand()/(RAND_MAX+1.0))),0));
            }
        }
    }
    world = new hitable_list(list, n);
    while (loop()) {
        input();
    }

    kill();
    return 0;
}


void drawPoint(const int& r = 0,const int& g = 0,const int& b = 0, const int& a = 255, const int& x = 0, const int& y = 0)
{
    SDL_SetRenderDrawColor(renderer, r, g, b, a);
    SDL_RenderDrawPoint(renderer, x, y);
}

void drawPointArea(const int& r = 0, const int& g = 0, const int& b = 0, const int& a = 255, const int& x = 0, const int& y = 0, const int& area = 1)
{
    SDL_SetRenderDrawColor(renderer, r, g, b, a);
    for (int i = 0; i < area; i++) {
        for (int j = 0; j < area; j++) {
            SDL_RenderDrawPoint(renderer, x + i, y + j);
        }
    }
}

bool loop()
{

    SDL_Event e;

    // Event loop
    while (SDL_PollEvent(&e) != 0)
    {
        switch (e.type)
        {
        case SDL_QUIT:
            EXIT = true;
        }
    }
    
    if (EXIT) {
        return false;
    }

    if (input()) {
        samplingNotDone = true;
    }

    if (samplingNotDone) {
        progSampling();  //for explroing with cam
    }

    //if want to create animation, decomment the following two and comment progSampling
    //Sampling();
    bufferTOscreen();


    // Update window
    SDL_RenderPresent(renderer);

    return true;
}

inline bool input() {
    static const unsigned char* keys = SDL_GetKeyboardState(NULL);
    if (keys[SDL_SCANCODE_W])
    {
        cam.MovFront(1);
        return true;
    }
    if (keys[SDL_SCANCODE_S])
    {
        cam.MovFront(0);
        return true;
    }
    if (keys[SDL_SCANCODE_A])
    {
        cam.MovRight(0);
        return true;
    }
    if (keys[SDL_SCANCODE_D])
    {
        cam.MovRight(1);
        return true;
    }
    if (keys[SDL_SCANCODE_SPACE])
    {
        cam.MovUp(1);
        return true;
    }
    if (keys[SDL_SCANCODE_LCTRL])
    {
        cam.MovUp(0);
        return true;
    }
    if (keys[SDL_SCANCODE_KP_8])
    {
        cam.rotVert(1);
        return true;
    }
    if (keys[SDL_SCANCODE_KP_2])
    {
        cam.rotVert(0);
        return true;
    }
    if (keys[SDL_SCANCODE_KP_6])
    {
        cam.rotHor(1);
        return true;
    }
    if (keys[SDL_SCANCODE_KP_4])
    {
        cam.rotHor(0);
        return true;
    }

    return false;
}

void bufferTOscreen(const int& k) {
    for (int y = 0; y < height; y+=k) {
        for (int x = 0; x < width; x+=k) {
            Vec3f col = buffer[y][x];
            drawPointArea(col.x * 255.99, col.y * 255.99, col.z * 255.99, 255, x, y, k);
        }
    }
}

void Sampling() {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            Vec3f col(0, 0, 0);
            for (int i = 0; i < sampling_number; i++) {
                float hor = float(x + (rand() / (RAND_MAX + 1.0))) / width;
                float vert = float(y + (rand() / (RAND_MAX + 1.0))) / height;
                ray r = cam.get_ray(hor, vert);
                col += color(r, world,0);
            }
            col /= float(sampling_number);
            col = Vec3f(sqrt(col.x), sqrt(col.y), sqrt(col.z));
            buffer[y][x] = col;
        }
    }
    cout << "sample done"<< endl;
}

//progressive sampling
void progSampling() {
    SDL_Event e;
    for (int i = 0; i < sampling_number; i++) {
        for (int lup = 0; lup < progSamplingNumber; lup++) {
            int jump = 1;
            if (i == 0) {
                jump = progSamplingNumber - lup;
                for (int y = 0; y < height; y+= jump) {
                    for (int x = 0; x < width; x += jump) {
                        float hor = float(x + (rand() / (RAND_MAX + 1.0))) / width;
                        float vert = float(y + (rand() / (RAND_MAX + 1.0))) / height;
                        ray r = cam.get_ray(hor, vert);
                        Vec3f col = color(r, world, 0);
                        col = Vec3f(sqrt(col.x), sqrt(col.y), sqrt(col.z));
                        buffer[y][x] = (buffer[y][x] * (i) + col) / (i + 1);
                    }
                }
                bufferTOscreen(progSamplingNumber - lup);
                if (jump == 1) {
                    continue;
                }
            }
            else {
                for (int y = lup; y < height; y += progSamplingNumber) {
                    for (int x = 0; x < width; x += 1) {
                        float hor = float(x + (rand() / (RAND_MAX + 1.0))) / width;
                        float vert = float(y + (rand() / (RAND_MAX + 1.0))) / height;
                        ray r = cam.get_ray(hor, vert);
                        Vec3f col = color(r, world, 0);
                        col = Vec3f(sqrt(col.x), sqrt(col.y), sqrt(col.z));
                        buffer[y][x] = (buffer[y][x] * (i - 1) + col) / (i);
                    }
                }
                bufferTOscreen(1);
            }
            SDL_RenderPresent(renderer);
            if (SDL_PollEvent(&e) != 0)
            {
                if (e.type == SDL_QUIT) {
                    EXIT = true;
                    return;
                }
                if (input()) {
                    samplingNotDone = true;
                    clrBuffer();
                    return;
                }
            }
        }
        cout <<"Sample Number :"<< i + 1 << endl;
    }
    samplingNotDone = false;
}

bool init()
{
    // See last example for comments
    if (SDL_Init(SDL_INIT_EVERYTHING) < 0)
    {
        cout << "Error initializing SDL: " << SDL_GetError() << endl;
        system("pause");
        return false;
    }

    window = SDL_CreateWindow("Example", SDL_WINDOWPOS_UNDEFINED, SDL_WINDOWPOS_UNDEFINED, width, height, SDL_WINDOW_SHOWN);
    if (!window)
    {
        cout << "Error creating window: " << SDL_GetError() << endl;
        system("pause");
        return false;
    }

    renderer = SDL_CreateRenderer(window, -1, SDL_RENDERER_ACCELERATED);
    if (!renderer)
    {
        cout << "Error creating renderer: " << SDL_GetError() << endl;
        return false;
    }

    SDL_SetRenderDrawColor(renderer, 255, 255, 255, 255);
    SDL_RenderClear(renderer);
    return true;
}

void kill()
{
    // Quit
    SDL_DestroyRenderer(renderer);
    SDL_DestroyWindow(window);
    SDL_Quit();
}
