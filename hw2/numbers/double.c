#include <stdio.h>
#define NUM_OF_BIT 8

int main() {
  double x;
  scanf("%lf", &x);

    char * c = (char*)&x + sizeof(double) - 1;

    for (int i =0; i < sizeof(double); i++){
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
