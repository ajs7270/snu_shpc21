#include <stdio.h>

#define MOVE_RIGHT 0
#define MOVE_DOWN 1
#define MOVE_LEFT 2
#define MOVE_UP 3
#define NUM_OF_MOVE_TYPE 4

#define TRUE 1
#define NUM_OF_ALPHABET 26

void move(int *x, int *y, int flag){
	switch(flag){
		case MOVE_RIGHT:
			*y += 1;
			break;
		case MOVE_DOWN:
			*x += 1;
			break;
		case MOVE_LEFT:
			*y -= 1;
			break;
		case MOVE_UP:
			*x -= 1;
			break;
		default:
			break;
	}
}

void revert(int *x, int *y, int flag){
	switch(flag){
		case MOVE_RIGHT:
			*y -= 1;
			break;
		case MOVE_DOWN:
			*x -= 1;
			break;
		case MOVE_LEFT:
			*y += 1;
			break;
		case MOVE_UP:
			*x += 1;
			break;
		default:
			break;
	}
}

int main(){
	int num;
	int move_flag = MOVE_RIGHT;
	int x =0, y=0;
	int alphabet_index = 0;
	scanf("%d",&num);
	char map[num][num];
	
	for(int i=0; i <num; i++){
		for(int j=0; j <num;j++){
			map[i][j] = 0;
		}
	}

	while(TRUE){
		if (map[x][y] != 0){
			break;
		}
		
		map[x][y] = 'A' + (alphabet_index++)%NUM_OF_ALPHABET;

		move(&x, &y, move_flag);
		if(x == num || y == num || map[x][y] != 0){
			revert(&x ,&y ,move_flag);
			move_flag = (move_flag + 1) % NUM_OF_MOVE_TYPE;
			move(&x, &y, move_flag);
		}
	}

		
	for(int i=0; i <num; i++){
		for(int j=0; j <num;j++){
			printf("%c", map[i][j]);
		}
		printf("\n");
	}

	return 0;
}
