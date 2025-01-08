#include <bits/stdc++.h>
using namespace std;

int main(){
	cout << "Enter a sequence of integer (-999 to finish): ";
	int a[1005], n=0, ans=0;
	while(cin >> a[n++] && a[n-1] != -999);
	for(int i=0; i<n-1; i+=2) ans+=a[i];
	for(int i=1; i<n-1; i+=2) ans-=a[i];
	cout << ans << endl;
	return 0;
}
