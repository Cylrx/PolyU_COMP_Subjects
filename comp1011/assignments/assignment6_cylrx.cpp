#include <cstring>
#include <iostream>
#include <random>
using namespace std;

const int N=205;
const int M=55;
const char PROMPT[] = "Enter student names and input END to finish the input:";
const char ERROR[] = "Wrong input: please input only upper-case and low-case letters with no space in between";
mt19937 rng(random_device{}());

inline void upper(char s[]){
	for(int i=0; s[i]; i++) if(s[i]>='a' && s[i]<='z') s[i] += 'A'-'a';
}

inline bool valid(char s[]){
	int i;
	for(i=0; s[i]; i++) 
		if(!(s[i]>='a' && s[i]<='z') && !(s[i]>='A' && s[i]<='Z')) return false;
	return i<=50 && i>0;
}

void ssort(char* s[], int n){
	if(n<=1) return;
	int z = rng() % n;
	char* x = s[z];
	s[z] = s[n-1];
	int i = 0, j = n-1;

	while(i<j){
		while(i<j && strcmp(s[i], x) < 0) i++;
		if(i<j) s[j--] = s[i];
		while(i<j && strcmp(s[j], x) > 0) j--;
		if(i<j) s[i++] = s[j];
	}
	
	s[i] = x;
	ssort(s, i);
	ssort(s+i+1, n-i-1);
}

int main(){
	char s[N][M];
	char* p[N];
	int n = 0;
	cout << PROMPT << endl;
	while(cin.getline(s[n], M) && strcmp(s[n], "END")!=0) {
		while(!valid(s[n])) {
			cout << ERROR << endl;
			cin.getline(s[n], M);
		}
		upper(s[n]);
		p[n] = s[n];
		n++;
	}
	ssort(p, n);
	for(int i=0; i<n; i++) cout << p[i] << endl;
	return 0;
}
