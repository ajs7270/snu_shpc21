#include <stdio.h>

#define MAX_OF_NUM 10001

int main(){
	int primes[MAX_OF_NUM];
	int num;
	
	for(int i=2; i<MAX_OF_NUM; i++){
		primes[i] = 1;
	}

	for(int i=2; i<MAX_OF_NUM; i++){
		if(primes[i] == 1){
			for(int j = 2; i*j < MAX_OF_NUM; j++){
				primes[i*j] == 0;
			}
		}
	}

	scanf("%d",&num);

	while(num != 1){
		for(int i=2; i<MAX_OF_NUM; i++){
			if(primes[i] == 0){
				continue;
			}

			if(i > num){
				continue;
			}

			if(num % i == 0){
				printf("%d ",i);
				num /= i;
				break;
			}
		}
	}
	
	printf("\n");
	
	return 0;
}
