#include <bits/stdc++.h>
using namespace std;

void rotate(char *charArray, int *sizeOfArray){
	char tmp = charArray[0];
	for(int i=0; i<*sizeOfArray-1; i++) charArray[i] = charArray[i+1];
	charArray[*sizeOfArray-1] = tmp;
}

const int N = 105;
int main(){
	ios::sync_with_stdio(0);
	cin.tie(0);

	char charArray[N];
	cin >> charArray;
	int sizeOfArray = (int)strlen(charArray);

	for(int i=0; i<sizeOfArray; i++){
		rotate(charArray, &sizeOfArray);
		cout << charArray << endl;
	}
	return 0;
}
