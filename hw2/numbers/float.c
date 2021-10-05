#include <stdio.h>
#define NUM_OF_BIT 8

int main() {
  float x;
  scanf("%f", &x);

  // for Little Endian
  char * c = (char*)&x + sizeof(float) - 1;

  for (int i =0; i < sizeof(float); i++){
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
