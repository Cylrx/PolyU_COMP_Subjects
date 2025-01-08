#include <bits/stdc++.h>
using namespace std;

int main(){
	int op, h, k; 
	cout << "order (1) or reverse order (2): "; cin >> op;
	cout << "please input the branch size: "; cin >> h;
	k = op==1? 0 : 25;
	for(int i=1; i<=h; i++){
		for(int j=0; j<=h-i; j++) cout << " ";
		for(int j=0; j<i*2-1; j++) {
			cout << char('A' + k);
			k = (k += op==1? 1 : -1) < 0 ? 26+k : k%26;
		}
		cout << endl;
	}
	for(int i=1; i<=h/2; i++){
		for(int j=0; j<h-1; j++) cout << " ";
		cout << char('A' + k) << " " << char('A' + k % 26) << endl;
		k = (k += op==1? 1 : -1) < 0 ? 26+k : k%26;
	}
	return 0;
}
