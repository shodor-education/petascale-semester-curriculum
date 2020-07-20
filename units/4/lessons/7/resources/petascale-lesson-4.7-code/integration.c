float f(float x) {
return (x*x);
}
int main() {
     int i, SECTIONS = 1000;
     float height = 0.0;
     float area = 0.0, y = 0.0, x = 0.0;
     float dx = 1.0/(float)SECTIONS;
          
           for( i = 0; i < SECTIONS; i++){
                x = i*dx;
                y = f(x);
                area += y*dx;
              }
  printf("Area under the curve is %f\n",area);
  return (0);
}

