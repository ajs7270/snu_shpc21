#include <stdio.h>
#define NUM_OF_BIT 8
#define SIZE_OF_LONGDOUBLE 10

int main() {
  long double x;
  scanf("%Lf", &x);

  // for Little Endian
  char * c = (char*)&x + SIZE_OF_LONGDOUBLE -1;

  for (int i =0; i < SIZE_OF_LONGDOUBLE; i++){
      unsigned char temp = 1 << (NUM_OF_BIT-1);
      for(int j = 0; j < NUM_OF_BIT; j++){
          if (temp & *c){
              printf("%d",1);
          } else{
              printf("%d",0);
          }
          temp >>= 1;
      }
      c -= 1;
  }
  printf("\n");

  return 0;
}
