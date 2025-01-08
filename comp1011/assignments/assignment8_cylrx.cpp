#include <bits/stdc++.h>
#define ff first
#define ss second
using namespace std;

const int N = 1005;
const int M = 55;

int main(){
	cout << "Enter student names and ID, and input END to finish the input:" << endl;

	pair<int, char*> arr[N];
	char name[N][M];
	int id, n = -1;
	while(cin >> name[++n] && strcmp(name[n], "END")){
		cin >> id;
		arr[n] = {id, name[n]};
	}
	sort(arr, arr+n);
	for(int i=0; i<n; i++) cout << arr[i].ss << " " << arr[i].ff << endl;
	
	return 0;
}
